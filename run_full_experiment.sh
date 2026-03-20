#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
CONFIG_FILE="${SCRIPT_DIR}/batch_scaling_config.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/batch_scaling_experiment}"

NPROBE_VALUES="128,256,512"
LLM_MODELS="llama3-8B,llama2-13B,opt-30B"
MAX_BATCH_SIZE=""
TEST_MODE=false
DRY_RUN=false

print_header() {
    echo
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo
}

print_step() {
    echo "[step] $1"
}

read_config_value() {
    local key_path="$1"
    python - "$CONFIG_FILE" "$key_path" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
key_path = sys.argv[2].split(".")
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

value = config
for key in key_path:
    value = value[key]

print(value)
PY
}

check_dependencies() {
    print_header "Checking dependencies"

    local required_cmds=("python" "nvidia-smi")
    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Missing command: $cmd"
            exit 1
        fi
    done

    python -c "import torch, transformers, vllm, faiss, yaml, pandas, matplotlib, datasets" >/dev/null 2>&1 || {
        echo "Missing Python dependencies. Run: pip install -r requirement.txt"
        exit 1
    }
}

check_gpu() {
    print_header "Checking GPU"
    nvidia-smi >/dev/null
    nvidia-smi --list-gpus
}

check_data() {
    print_header "Checking data"

    local corpus_path
    local index_path
    corpus_path="$(read_config_value retrieval.corpus_path)"
    index_path="$(read_config_value retrieval.index_path)"

    if [ -f "$corpus_path" ]; then
        echo "Found local corpus: $corpus_path"
    else
        python - "$corpus_path" <<'PY'
import sys
from datasets import load_dataset

corpus_path = sys.argv[1]
try:
    dataset = load_dataset(corpus_path)
    if "train" not in dataset:
        raise RuntimeError(f"Dataset {corpus_path} does not expose a train split.")
    print(f"Verified HuggingFace corpus: {corpus_path}")
except Exception as exc:
    raise SystemExit(f"Failed to load corpus {corpus_path}: {exc}")
PY
    fi

    if [ ! -f "$index_path" ]; then
        echo "Missing index: $index_path"
        exit 1
    fi

    python -c "from datasets import load_dataset; load_dataset('natural_questions', split='test', streaming=True)" >/dev/null 2>&1 || {
        echo "Warning: NaturalQuestions dataset was not reachable during pre-check."
        echo "The experiment script will attempt to download it on first run."
    }
}

run_single_experiment() {
    local nprobe="$1"
    local llm="$2"
    local output_dir="$3"

    local cmd=(
        python
        "${SCRIPT_DIR}/batch_scaling_experiment.py"
        --config "${CONFIG_FILE}"
        --nprobe "${nprobe}"
        --llm_model "${llm}"
        --output_dir "${output_dir}"
    )

    if [ -n "${MAX_BATCH_SIZE}" ]; then
        cmd+=(--max_batch_size "${MAX_BATCH_SIZE}")
    fi

    if [ "${TEST_MODE}" = true ]; then
        cmd+=(--test_mode)
    fi

    if [ "${DRY_RUN}" = true ]; then
        printf '[dry-run]'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    print_step "Running nprobe=${nprobe}, llm=${llm}"
    "${cmd[@]}"
}

run_all_experiments() {
    print_header "Running experiments"

    mkdir -p "${OUTPUT_DIR}"

    IFS=',' read -ra nprobes <<< "${NPROBE_VALUES}"
    IFS=',' read -ra llms <<< "${LLM_MODELS}"

    local total=$(( ${#nprobes[@]} * ${#llms[@]} ))
    local current=0

    for nprobe in "${nprobes[@]}"; do
        for llm in "${llms[@]}"; do
            current=$((current + 1))
            echo "[${current}/${total}] nprobe=${nprobe}, llm=${llm}"
            run_single_experiment "${nprobe}" "${llm}" "${OUTPUT_DIR}"
        done
    done
}

collect_results() {
    print_header "Collecting results"

    OUTPUT_DIR_ENV="${OUTPUT_DIR}" python - <<'PY'
import os
from pathlib import Path

import pandas as pd

output_dir = Path(os.environ["OUTPUT_DIR_ENV"])
frames = []

for csv_path in sorted(output_dir.glob("results_*.csv")):
    df = pd.read_csv(csv_path)
    if "nprobe" not in df.columns or "llm_model" not in df.columns:
        parts = csv_path.stem.split("_", 2)
        if len(parts) == 3:
            _, nprobe, llm_model = parts
            df["nprobe"] = int(nprobe)
            df["llm_model"] = llm_model
    frames.append(df)

if not frames:
    raise SystemExit("No results_*.csv files were found.")

summary = pd.concat(frames, ignore_index=True)
summary_path = output_dir / "summary_results.csv"
summary.to_csv(summary_path, index=False)
print(summary_path)
PY
}

show_help() {
    cat <<'HELP'
Usage: bash run_full_experiment.sh [options]

Options:
  --nprobe 128,256,512
  --llm llama3-8B,llama2-13B,opt-30B
  --max_batch_size 512
  --output_dir /path/to/results
  --test_mode
  --dry_run
  --help
HELP
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --nprobe)
                NPROBE_VALUES="$2"
                shift 2
                ;;
            --llm)
                LLM_MODELS="$2"
                shift 2
                ;;
            --max_batch_size)
                MAX_BATCH_SIZE="$2"
                shift 2
                ;;
            --output_dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --test_mode)
                TEST_MODE=true
                shift
                ;;
            --dry_run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    print_header "Batch size scaling experiment"
    echo "Config file: ${CONFIG_FILE}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "nprobe values: ${NPROBE_VALUES}"
    echo "LLM models: ${LLM_MODELS}"
    echo "Max batch size: ${MAX_BATCH_SIZE:-none}"
    echo "Test mode: ${TEST_MODE}"
    echo

    check_dependencies
    check_gpu
    check_data
    run_all_experiments
    collect_results

    print_header "Done"
    echo "Results written to ${OUTPUT_DIR}"
}

main "$@"
