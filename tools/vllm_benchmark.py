#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
vllm_benchmark.py — Self-contained vLLM benchmark + kernel profiling.

Bypasses InferenceX / Magpie benchmark mode. Produces a benchmark_report.json
compatible with the Apex workload_optimizer pipeline (--skip-benchmark).

Steps:
  1. Start a vLLM OpenAI-compatible server
  2. Send concurrent requests, measure throughput
  3. Collect torch profiler traces for kernel gap analysis
  4. Parse traces -> top bottleneck kernels
  5. Output benchmark_report.json

Usage:
  python3 tools/vllm_benchmark.py \\
    --model /app/models/Qwen3.5-4B \\
    --output-dir ./results/benchmark \\
    [--tp 1] [--dtype bf16] [--max-model-len 4096] \\
    [--num-prompts 50] [--input-len 512] [--output-len 128] \\
    [--concurrency 4] [--profile]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait until vLLM server is ready."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _start_vllm_server(
    model: str, port: int, tp: int, dtype: str, max_model_len: int,
    enforce_eager: bool, gpu_mem_util: float, profile: bool,
    profile_dir: Optional[str] = None,
) -> subprocess.Popen:
    """Start vLLM OpenAI server as subprocess."""
    # Use vllm binary if available, else module invocation
    vllm_bin = shutil.which("vllm")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--no-enable-log-requests",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    env = os.environ.copy()
    if profile and profile_dir:
        env["VLLM_TORCH_PROFILER_DIR"] = profile_dir
        print(f"  [vllm] Torch profiler enabled: {profile_dir}")

    print(f"  [vllm] Starting: {' '.join(cmd[:8])}...")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, start_new_session=True,
    )
    return proc


def _generate_prompts(num: int, input_len: int) -> list[str]:
    """Generate synthetic prompts of approximately input_len tokens."""
    # ~4 chars per token is a rough estimate
    base = "Explain the following topic in detail: "
    filler = "the quick brown fox jumps over the lazy dog and "
    prompts = []
    for i in range(num):
        target_chars = input_len * 4
        prompt = base + f"(topic {i}) " + filler * (target_chars // len(filler))
        prompts.append(prompt[:target_chars])
    return prompts


def _run_benchmark(
    port: int, prompts: list[str], output_len: int, concurrency: int,
    model_name: str = "",
) -> dict:
    """Send requests to vLLM and measure throughput."""
    import urllib.request
    import concurrent.futures
    import threading

    url = f"http://localhost:{port}/v1/completions"
    results = {"completed": 0, "failed": 0, "total_input_tokens": 0,
               "total_output_tokens": 0, "latencies": []}
    lock = threading.Lock()

    def _send_one(prompt: str) -> None:
        body = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            data = json.loads(resp.read())
            elapsed = time.monotonic() - t0
            usage = data.get("usage", {})
            with lock:
                results["completed"] += 1
                results["total_input_tokens"] += usage.get("prompt_tokens", 0)
                results["total_output_tokens"] += usage.get("completion_tokens", 0)
                results["latencies"].append(elapsed * 1000)
        except Exception as e:
            with lock:
                results["failed"] += 1
            print(f"    request failed: {e}", file=sys.stderr)

    print(f"  [bench] Sending {len(prompts)} requests (concurrency={concurrency})...")
    t_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send_one, p) for p in prompts]
        concurrent.futures.wait(futures)

    duration = time.monotonic() - t_start
    results["duration_seconds"] = round(duration, 2)
    results["request_throughput"] = round(results["completed"] / duration, 2) if duration > 0 else 0
    results["output_throughput"] = round(results["total_output_tokens"] / duration, 2) if duration > 0 else 0
    results["total_token_throughput"] = round(
        (results["total_input_tokens"] + results["total_output_tokens"]) / duration, 2
    ) if duration > 0 else 0

    print(f"  [bench] Completed: {results['completed']}/{len(prompts)}, "
          f"throughput: {results['output_throughput']} tok/s, "
          f"duration: {duration:.1f}s")
    return results


def _parse_torch_traces(trace_dir: str, top_k: int = 20) -> dict:
    """Parse torch profiler traces and extract top GPU kernels."""
    trace_files = list(Path(trace_dir).rglob("*.json"))
    if not trace_files:
        trace_files = list(Path(trace_dir).rglob("*.json.gz"))
    if not trace_files:
        print(f"  [profile] No trace files found in {trace_dir}")
        return {"top_kernels": [], "errors": ["no trace files"]}

    print(f"  [profile] Parsing {len(trace_files)} trace file(s)...")
    kernel_stats: dict[str, dict] = {}
    total_gpu_us = 0.0

    for tf in trace_files[:4]:  # limit to avoid OOM on large traces
        try:
            import gzip
            opener = gzip.open if str(tf).endswith(".gz") else open
            with opener(tf, "rt") as f:
                data = json.load(f)

            events = data if isinstance(data, list) else data.get("traceEvents", [])
            for ev in events:
                if ev.get("cat") not in ("kernel", "gpu_memcpy", "cuda_runtime"):
                    continue
                dur = ev.get("dur", 0)
                if dur <= 0:
                    continue
                name = ev.get("name", "unknown")
                if name not in kernel_stats:
                    kernel_stats[name] = {"calls": 0, "total_us": 0.0}
                kernel_stats[name]["calls"] += 1
                kernel_stats[name]["total_us"] += dur
                total_gpu_us += dur
        except Exception as e:
            print(f"  [profile] Error parsing {tf.name}: {e}", file=sys.stderr)

    if not kernel_stats:
        return {"top_kernels": [], "errors": ["no kernel events found"]}

    # Sort by total time descending
    sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)

    top_kernels = []
    for name, stats in sorted_kernels[:top_k]:
        pct = (stats["total_us"] / total_gpu_us * 100) if total_gpu_us > 0 else 0
        top_kernels.append({
            "name": name,
            "calls": stats["calls"],
            "self_cuda_total_us": round(stats["total_us"], 2),
            "avg_time_us": round(stats["total_us"] / stats["calls"], 2),
            "pct_total": round(pct, 2),
        })

    print(f"  [profile] Found {len(kernel_stats)} unique kernels, top {len(top_kernels)} selected")
    return {
        "total_duration_us": round(total_gpu_us, 2),
        "top_kernels": top_kernels,
        "errors": [],
    }


def _build_report(
    model: str, framework: str, bench_results: dict,
    gap_analysis: Optional[dict] = None,
) -> dict:
    """Build Magpie-compatible benchmark_report.json."""
    latencies = bench_results.get("latencies", [])
    lat_stats = {}
    if latencies:
        import statistics
        latencies.sort()
        lat_stats = {
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2) if len(latencies) > 1 else round(latencies[0], 2),
            "std_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        }

    report = {
        "success": bench_results["completed"] > 0,
        "framework": framework,
        "model": model,
        "throughput": {
            "request_throughput": bench_results.get("request_throughput", 0),
            "output_throughput": bench_results.get("output_throughput", 0),
            "total_token_throughput": bench_results.get("total_token_throughput", 0),
            "completed_requests": bench_results.get("completed", 0),
            "total_input_tokens": bench_results.get("total_input_tokens", 0),
            "total_output_tokens": bench_results.get("total_output_tokens", 0),
            "duration_seconds": bench_results.get("duration_seconds", 0),
        },
        "latency": {"e2el": lat_stats},
        "kernel_summary": [],
        "top_bottlenecks": [],
    }

    if gap_analysis and gap_analysis.get("top_kernels"):
        report["gap_analysis"] = gap_analysis
        report["top_bottlenecks"] = [k["name"] for k in gap_analysis["top_kernels"][:10]]
        report["kernel_summary"] = [
            {"name": k["name"], "time_ms": round(k["self_cuda_total_us"] / 1000, 2),
             "percent": k["pct_total"], "calls": k["calls"]}
            for k in gap_analysis["top_kernels"][:20]
        ]

    return report


def main():
    parser = argparse.ArgumentParser(description="Self-contained vLLM benchmark for Apex pipeline")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--output-dir", required=True, help="Output directory for benchmark_report.json")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler for gap analysis")
    parser.add_argument("--warmup-prompts", type=int, default=5, help="Warmup requests before benchmark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    port = _find_free_port()
    profile_dir = str(output_dir / "traces") if args.profile else None
    if profile_dir:
        Path(profile_dir).mkdir(exist_ok=True)

    # Start vLLM server
    print("\n=== Step 1: Starting vLLM server ===")
    server_proc = _start_vllm_server(
        model=args.model, port=port, tp=args.tp, dtype=args.dtype,
        max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
        gpu_mem_util=args.gpu_mem_util, profile=args.profile,
        profile_dir=profile_dir,
    )

    try:
        print(f"  [vllm] Waiting for server on port {port}...")
        if not _wait_for_server(port, timeout=300):
            print("  [vllm] ERROR: Server failed to start within 300s")
            # Dump server output for debugging
            server_proc.terminate()
            server_proc.wait(timeout=10)
            if server_proc.stdout:
                out = server_proc.stdout.read()
                print(f"  [vllm] Server output (last 2000 chars):\n{out[-2000:]}")
            sys.exit(1)
        print(f"  [vllm] Server ready on port {port}")

        # Warmup
        if args.warmup_prompts > 0:
            print(f"\n=== Step 2: Warmup ({args.warmup_prompts} requests) ===")
            warmup_prompts = _generate_prompts(args.warmup_prompts, args.input_len)
            _run_benchmark(port, warmup_prompts, args.output_len, concurrency=2,
                           model_name=args.model)

        # Benchmark
        print(f"\n=== Step 3: Benchmark ({args.num_prompts} requests) ===")
        prompts = _generate_prompts(args.num_prompts, args.input_len)
        bench_results = _run_benchmark(port, prompts, args.output_len, args.concurrency,
                                       model_name=args.model)

        # Gap analysis from traces
        gap_analysis = None
        if args.profile and profile_dir and Path(profile_dir).exists():
            print(f"\n=== Step 4: Gap Analysis (torch profiler traces) ===")
            gap_analysis = _parse_torch_traces(profile_dir)

    finally:
        print("\n  [vllm] Shutting down server...")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            server_proc.terminate()
        server_proc.wait(timeout=30)

    # Build and save report
    print(f"\n=== Step 5: Generating report ===")
    report = _build_report(
        model=args.model, framework="vllm",
        bench_results=bench_results, gap_analysis=gap_analysis,
    )

    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved: {report_path}")
    print(f"  Throughput: {report['throughput']['output_throughput']} tok/s")
    if gap_analysis:
        n = len(gap_analysis.get("top_kernels", []))
        print(f"  Gap analysis: {n} top kernels identified")

    # Print top kernels summary
    if report.get("kernel_summary"):
        print("\n  Top GPU kernels:")
        for i, k in enumerate(report["kernel_summary"][:10], 1):
            print(f"    {i:2d}. {k['percent']:5.1f}%  {k['calls']:6d} calls  {k['name'][:60]}")

    return report_path


if __name__ == "__main__":
    main()
