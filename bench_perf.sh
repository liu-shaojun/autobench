#!/usr/bin/env bash
# Performance benchmark for vLLM server - minimal standalone version
# Usage: bash bench_perf.sh [MODEL_NAME] [MODEL_PATH] [PORT]
#   e.g. bash bench_perf.sh Qwen3.6-27B /llm/models/Qwen3.6-27B 9005
#
# Edit CONCURRENCY / INPUT_LENS / OUTPUT_LEN below to customize the matrix.

MODEL="${1:-model}"
MODEL_PATH="${2:-/llm/models/${MODEL}}"
PORT="${3:-9005}"

CONCURRENCY=(1 4)
INPUT_LENS=(1024 8192 16384 32768 65536)
OUTPUT_LEN=2048

LOG="bench_$(date +%Y%m%d_%H%M%S).log"

echo "=== vLLM Performance Benchmark ===" | tee "$LOG"
echo "Model:       $MODEL" | tee -a "$LOG"
echo "Model path:  $MODEL_PATH" | tee -a "$LOG"
echo "Port:        $PORT" | tee -a "$LOG"
echo "Concurrency: ${CONCURRENCY[*]}" | tee -a "$LOG"
echo "Input lens:  ${INPUT_LENS[*]}" | tee -a "$LOG"
echo "Output len:  $OUTPUT_LEN" | tee -a "$LOG"
echo "Log:         $LOG" | tee -a "$LOG"
echo | tee -a "$LOG"

# Warmup
echo "--- Warmup ---" | tee -a "$LOG"
vllm bench serve \
    --model="$MODEL_PATH" \
    --served-model-name="$MODEL" \
    --dataset-name=random \
    --random-input-len=128 \
    --random-output-len=16 \
    --ignore-eos \
    --num-prompt=1 \
    --trust_remote_code \
    --request-rate=inf \
    --backend=vllm \
    --port="$PORT" \
    --max-concurrency=1 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

# Benchmark matrix
for c in "${CONCURRENCY[@]}"; do
    for i in "${INPUT_LENS[@]}"; do
        echo "--- c=$c in=$i out=$OUTPUT_LEN num_prompt=$c ---" | tee -a "$LOG"
        vllm bench serve \
            --model="$MODEL_PATH" \
            --served-model-name="$MODEL" \
            --dataset-name=random \
            --random-input-len="$i" \
            --random-output-len="$OUTPUT_LEN" \
            --ignore-eos \
            --num-prompt="$c" \
            --trust_remote_code \
            --request-rate=inf \
            --backend=vllm \
            --port="$PORT" \
            --max-concurrency="$c" 2>&1 | tee -a "$LOG"
        echo | tee -a "$LOG"
    done
done

echo "=== Done. Log: $LOG ===" | tee -a "$LOG"
