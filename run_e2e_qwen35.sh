#!/usr/bin/env bash
# =============================================================================
# run_e2e_qwen35.sh — End-to-end model performance optimization for Qwen3.5-4B
# =============================================================================
#
# This script runs the full Apex pipeline on AMD gfx1100 (RDNA3):
#
#   Step 1: Benchmark  — Profile Qwen3.5-4B on vLLM, collect kernel traces
#   Step 2: Identify   — Extract & classify bottleneck GPU kernels
#   Step 3: Optimize   — Agent (OpenCode) writes optimized Triton kernels
#   Step 4: Grade      — Compile + correctness + speedup scoring
#   Step 5: Integrate  — Hot-patch optimized kernels into site-packages
#   Step 6: Re-bench   — Final E2E benchmark with optimized kernels
#   Step 7: Score      — Compute trajectory reward
#   Step 8: Report     — Generate markdown report
#
# Usage:
#   bash run_e2e_qwen35.sh              # Run full pipeline
#   bash run_e2e_qwen35.sh --dry-run    # Simulate (no GPU/API calls)
#   bash run_e2e_qwen35.sh --from step  # Resume from a specific step
#                                       # (benchmark|identify|optimize|report)
#   bash run_e2e_qwen35.sh --skip-benchmark /path/to/report.json
#
# Requirements:
#   - AMD GPU (gfx1100 tested), ROCm 7.x, PyTorch, vLLM, Triton
#   - Qwen3.5-4B model at /app/models/Qwen3.5-4B
#   - opencode CLI (agent backend)
#   - Magpie at /app/Magpie
#
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="/app/models/Qwen3.5-4B"
MODEL_ID="Qwen/Qwen3.5-4B"
MAGPIE_ROOT="${MAGPIE_ROOT:-/app/Magpie}"

# Fix broken editable vllm install — add source to PYTHONPATH if needed
if ! python3 -c "from vllm import LLM" 2>/dev/null; then
  if [[ -d /app/vllm-src/vllm/vllm ]]; then
    export PYTHONPATH="/app/vllm-src/vllm${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results_qwen35_$(date +%Y%m%d_%H%M%S)}"
GPU_ARCH="gfx1100"

# Agent settings
AGENT_BACKEND="${AGENT_BACKEND:-opencode}"
AGENT_MODEL="${AGENT_MODEL:-}"          # empty = backend default
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
MAX_TURNS="${MAX_TURNS:-20}"
TOP_K="${TOP_K:-5}"
KERNEL_TYPES="${KERNEL_TYPES:-triton}"

# Parse CLI args
DRY_RUN=""
SKIP_BENCHMARK=""
FROM_STEP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)       DRY_RUN="--dry-run"; shift ;;
    --skip-benchmark) SKIP_BENCHMARK="$2"; shift 2 ;;
    --from)          FROM_STEP="$2"; shift 2 ;;
    --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
    --agent)         AGENT_BACKEND="$2"; shift 2 ;;
    --model)         AGENT_MODEL="$2"; shift 2 ;;
    --top-k)         TOP_K="$2"; shift 2 ;;
    --kernel-types)  KERNEL_TYPES="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Benchmark config (written to RESULTS_DIR) ───────────────────────────────
BENCH_CONFIG="$RESULTS_DIR/benchmark_config.yaml"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
banner() { echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"; echo -e "${BOLD} $1${NC}"; echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}\n"; }
info()   { echo -e "  ${GREEN}▸${NC} $1"; }
warn()   { echo -e "  ${YELLOW}!${NC} $1"; }

# ── Preflight checks ────────────────────────────────────────────────────────
banner "Preflight Checks"

[[ -d "$MODEL_PATH" ]] && info "Model: $MODEL_PATH" || { echo "ERROR: Model not found at $MODEL_PATH"; exit 1; }
[[ -d "$MAGPIE_ROOT/Magpie" ]] && info "Magpie: $MAGPIE_ROOT" || warn "Magpie not found at $MAGPIE_ROOT (benchmark step may fail)"
command -v python3 &>/dev/null && info "Python: $(python3 --version 2>&1)" || { echo "ERROR: python3 not found"; exit 1; }

if [[ "$AGENT_BACKEND" == "opencode" ]]; then
  command -v opencode &>/dev/null && info "Agent: opencode ($(opencode --version 2>&1 | head -1))" || { echo "ERROR: opencode not in PATH"; exit 1; }
fi

python3 -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_properties(0).gcnArchName}')" 2>/dev/null \
  && info "GPU: $GPU_ARCH (ROCm)" || warn "No GPU detected; benchmark step requires GPU"

mkdir -p "$RESULTS_DIR"
info "Results: $RESULTS_DIR"
[[ -n "$DRY_RUN" ]] && warn "DRY-RUN mode: no GPU or API calls"

# ── Generate benchmark config ────────────────────────────────────────────────
cat > "$BENCH_CONFIG" <<YAML
# Auto-generated Magpie benchmark config for Qwen3.5-4B on gfx1100
benchmark:
  framework: vllm
  model: $MODEL_PATH
  precision: bf16
  run_mode: local
  gpu_arch: $GPU_ARCH

  envs:
    TP: 1
    CONC: 4
    ISL: 512
    OSL: 128
    RANDOM_RANGE_RATIO: 0.5
    MAX_MODEL_LEN: 4096
    GPU_MEM_UTIL: 0.90
    EXTRA_VLLM_ARGS: "--enforce-eager"

  profiler:
    torch_profiler:
      enabled: true
    system_profiler:
      enabled: false

  gap_analysis:
    enabled: true
    trace_start_pct: 40
    trace_end_pct: 80
    top_k: 20

  timeout_seconds: 1800
YAML
info "Benchmark config: $BENCH_CONFIG"

# ── Helper: should we run this step? ─────────────────────────────────────────
STEPS=(benchmark identify optimize grade integrate benchmark-final score report)
should_run() {
  local step="$1"
  [[ -z "$FROM_STEP" ]] && return 0
  local found=false
  for s in "${STEPS[@]}"; do
    [[ "$s" == "$FROM_STEP" ]] && found=true
    $found && [[ "$s" == "$step" ]] && return 0
  done
  return 1
}

# ── Common args ──────────────────────────────────────────────────────────────
COMMON_ARGS=(
  -r "$RESULTS_DIR"
  --gpu "$GPU_ARCH"
  $DRY_RUN
)

AGENT_ARGS=(
  --agent-backend "$AGENT_BACKEND"
  --max-iterations "$MAX_ITERATIONS"
  --max-turns "$MAX_TURNS"
  --score-threshold 300
)
[[ -n "$AGENT_MODEL" ]] && AGENT_ARGS+=(--agent-model "$AGENT_MODEL")

cd "$SCRIPT_DIR"
export MAGPIE_ROOT

# =============================================================================
# Step 1: Benchmark
# =============================================================================
if should_run benchmark; then
  banner "Step 1/8: E2E Benchmark (Qwen3.5-4B on vLLM)"

  BENCHMARK_REPORT="$RESULTS_DIR/benchmark_report.json"

  if [[ -n "$SKIP_BENCHMARK" ]]; then
    info "Using existing benchmark report: $SKIP_BENCHMARK"
    cp "$SKIP_BENCHMARK" "$BENCHMARK_REPORT"
  elif [[ -n "$DRY_RUN" ]]; then
    info "Dry-run: using pipeline's built-in synthetic benchmark"
    python3 workload_optimizer.py benchmark \
      "${COMMON_ARGS[@]}" \
      -b "$BENCH_CONFIG" \
      --num-benchmark-runs 1 \
      2>&1 | tee "$RESULTS_DIR/step1_benchmark.log"
    info "Benchmark log: $RESULTS_DIR/step1_benchmark.log"
  else
    # Run self-contained vLLM benchmark (no InferenceX/Magpie dependency)
    info "Running vLLM benchmark directly (torch profiler + gap analysis)..."
    python3 tools/vllm_benchmark.py \
      --model "$MODEL_PATH" \
      --output-dir "$RESULTS_DIR/benchmark" \
      --tp 1 --dtype bfloat16 \
      --max-model-len 4096 \
      --num-prompts 50 \
      --input-len 512 --output-len 128 \
      --concurrency 4 \
      --profile \
      2>&1 | tee "$RESULTS_DIR/step1_benchmark.log"

    # Copy the report for --skip-benchmark usage
    if [[ -f "$RESULTS_DIR/benchmark/benchmark_report.json" ]]; then
      cp "$RESULTS_DIR/benchmark/benchmark_report.json" "$BENCHMARK_REPORT"
      info "Benchmark report: $BENCHMARK_REPORT"
    else
      echo "ERROR: Benchmark did not produce a report"; exit 1
    fi
  fi

  # Feed the report into the pipeline state via --skip-benchmark
  if [[ -f "$BENCHMARK_REPORT" ]] && [[ -z "$DRY_RUN" ]]; then
    python3 workload_optimizer.py benchmark \
      "${COMMON_ARGS[@]}" \
      -b "$BENCH_CONFIG" \
      --skip-benchmark "$BENCHMARK_REPORT" \
      2>&1 | tee -a "$RESULTS_DIR/step1_benchmark.log"
  fi
fi

# =============================================================================
# Step 2: Identify bottleneck kernels
# =============================================================================
if should_run identify; then
  banner "Step 2/8: Identify Bottleneck Kernels"

  python3 workload_optimizer.py identify \
    "${COMMON_ARGS[@]}" \
    --kernel-types "$KERNEL_TYPES" \
    --top-k "$TOP_K" \
    2>&1 | tee "$RESULTS_DIR/step2_identify.log"

  info "Identified kernels log: $RESULTS_DIR/step2_identify.log"
fi

# =============================================================================
# Step 2b: List kernels (informational)
# =============================================================================
if should_run identify; then
  banner "Step 2b: Kernel Summary"

  python3 workload_optimizer.py list-kernels \
    -r "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/step2b_kernels.log"
fi

# =============================================================================
# Step 4: Optimize (agent writes solutions)
# =============================================================================
if should_run optimize; then
  banner "Step 3/8: Kernel Optimization (Agent: $AGENT_BACKEND)"

  python3 workload_optimizer.py optimize \
    "${COMMON_ARGS[@]}" \
    "${AGENT_ARGS[@]}" \
    --kernel-types "$KERNEL_TYPES" \
    2>&1 | tee "$RESULTS_DIR/step3_optimize.log"

  info "Optimization log: $RESULTS_DIR/step3_optimize.log"
fi

# =============================================================================
# Step 5: Grade
# =============================================================================
if should_run grade; then
  banner "Step 4/8: Grade Solutions"

  python3 workload_optimizer.py grade \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$RESULTS_DIR/step4_grade.log"

  info "Grade log: $RESULTS_DIR/step4_grade.log"
fi

# =============================================================================
# Step 6: Integrate (hot-patch optimized kernels)
# =============================================================================
if should_run integrate; then
  banner "Step 5/8: Integrate Optimized Kernels"

  python3 workload_optimizer.py integrate \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$RESULTS_DIR/step5_integrate.log"

  info "Integration log: $RESULTS_DIR/step5_integrate.log"
fi

# =============================================================================
# Step 7: Final benchmark
# =============================================================================
if should_run benchmark-final; then
  banner "Step 6/8: Final E2E Benchmark"

  python3 workload_optimizer.py benchmark-final \
    "${COMMON_ARGS[@]}" \
    -b "$BENCH_CONFIG" \
    2>&1 | tee "$RESULTS_DIR/step6_final_bench.log"

  info "Final benchmark log: $RESULTS_DIR/step6_final_bench.log"
fi

# =============================================================================
# Step 8: Score
# =============================================================================
if should_run score; then
  banner "Step 7/8: Compute Trajectory Reward"

  python3 workload_optimizer.py score \
    "${COMMON_ARGS[@]}" \
    --agent-backend "$AGENT_BACKEND" \
    2>&1 | tee "$RESULTS_DIR/step7_score.log"

  info "Score log: $RESULTS_DIR/step7_score.log"
fi

# =============================================================================
# Step 9: Report
# =============================================================================
if should_run report; then
  banner "Step 8/8: Generate Report"

  python3 workload_optimizer.py report \
    "${COMMON_ARGS[@]}" \
    -b "$BENCH_CONFIG" \
    --agent-backend "$AGENT_BACKEND" \
    2>&1 | tee "$RESULTS_DIR/step8_report.log"

  info "Report log: $RESULTS_DIR/step8_report.log"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
banner "Done"
echo -e "  Results directory: ${BOLD}$RESULTS_DIR${NC}"
echo ""
echo "  Key outputs:"
[[ -f "$RESULTS_DIR/pipeline_state.json" ]] && echo "    pipeline_state.json  — full pipeline state"
[[ -f "$RESULTS_DIR/report.md" ]]           && echo "    report.md            — optimization report"
[[ -f "$RESULTS_DIR/trajectory.json" ]]     && echo "    trajectory.json      — RL trajectory"
echo "    step*.log            — per-step logs"
echo ""
echo "  To resume from a step:"
echo "    bash run_e2e_qwen35.sh --from optimize --results-dir $RESULTS_DIR"
echo ""
