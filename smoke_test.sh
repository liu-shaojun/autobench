#!/usr/bin/env bash
# Smoke test for vLLM server - minimal standalone version
# Usage: bash smoke_test.sh [MODEL_NAME] [PORT]
#   e.g. bash smoke_test.sh Qwen3.6-27B 9005

MODEL="${1:-model}"
PORT="${2:-9005}"
URL="http://localhost:${PORT}/v1"
PASS=0
FAIL=0

smoke() {
    local label="$1" mode="$2" prompt="$3"
    echo -n "  [$mode] $label ... "

    if [ "$mode" = "chat" ]; then
        resp=$(curl -s --max-time 30 -X POST "${URL}/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":150,\"temperature\":0}" \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])" 2>/dev/null)
    else
        resp=$(curl -s --max-time 30 -X POST "${URL}/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"${prompt}\",\"max_tokens\":30,\"temperature\":0}" \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'][:200])" 2>/dev/null)
    fi

    if [ -n "$resp" ]; then
        echo "OK"
        echo "    -> $resp"
        PASS=$((PASS+1))
    else
        echo "FAIL"
        FAIL=$((FAIL+1))
    fi
}

echo "=== Smoke Test ==="
echo "Model: $MODEL    URL: $URL"
echo

smoke "greeting"   chat       "Hi"
smoke "math_basic"  chat       "What is 2+2?"
smoke "math_word"   chat       "Natalia sold clips to 48 friends in April and half as many in May. How many clips altogether?"
smoke "code"        chat       "Write a Python function that checks if any two numbers in a list are within a given threshold."
smoke "knowledge"   completion "The capital of France is"

echo
echo "=== Result: $PASS passed, $FAIL failed ==="
