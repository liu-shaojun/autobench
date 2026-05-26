#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Isolated GSM8K evaluation script for vLLM serve endpoint.
"""

import argparse
import ast
import asyncio
import json
import os
import time
from collections.abc import Generator

import aiohttp
import numpy as np
import regex as re
import requests
from tqdm.asyncio import tqdm

INVALID = -9999999


def download_and_cache_file(url: str, filename: str | None = None) -> str:
    """Download and cache a file from a URL."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    return filename


def load_gsm8k_data() -> tuple[list[dict], list[dict]]:
    """Load GSM8K train and test data"""
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

    train_file = download_and_cache_file(train_url)
    test_file = download_and_cache_file(test_url)

    train_data = list(read_jsonl(train_file))
    test_data = list(read_jsonl(test_file))

    return train_data, test_data


def read_jsonl(filename: str) -> Generator[dict, None, None]:
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if not line.startswith("#"):
                yield json.loads(line)


def get_answer_value(answer_str: str) -> int:
    """Extract the numerical answer from the response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


async def call_vllm_api(
    session: aiohttp.ClientSession,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stop: list[str] | None = None,
    url: str | None = None,
    seed: int | None = None,
    extra_params: dict | None = None,
) -> tuple[str, int]:
    """Call vLLM's OpenAI-compatible chat completions endpoint.

    Returns:
        Tuple of (response_text, completion_tokens)
    """
    if extra_params is None:
        extra_params = {}
    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "chat_template_kwargs": {"enable_thinking": False},
        **extra_params,
    }
    if seed is not None:
        data["seed"] = seed

    try:
        async with session.post(f"{url}/v1/chat/completions", json=data) as response:
            response.raise_for_status()
            result = await response.json()
            text = result["choices"][0]["message"]["content"]
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            return text, completion_tokens
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return "", 0


def evaluate_gsm8k(
    num_questions: int = 1319,
    num_shots: int = 5,
    max_tokens: int = 256,
    host: str = "http://127.0.0.1",
    port: int = 8000,
    temperature: float = 0.0,
    seed: int | None = 42,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> dict[str, float | int]:
    """
    Evaluate GSM8K accuracy using vLLM serve endpoint.

    Returns dict with accuracy, invalid_rate, latency, etc.
    """
    base_url = f"{host}:{port}"

    # Load GSM8K train and test data
    train_data, test_data = load_gsm8k_data()

    # Limit to available test questions
    num_questions = min(num_questions, len(test_data))

    # Prepare test labels from test split
    labels = []
    for i in range(num_questions):
        labels.append(get_answer_value(test_data[i]["answer"]))

    assert all(label != INVALID for label in labels), "Some labels are invalid"

    # Run evaluation
    async def run_async_evaluation():
        states: list[str] = [""] * num_questions
        output_tokens: list[int] = [0] * num_questions

        async def get_answer(session: aiohttp.ClientSession, i: int) -> tuple[str, int]:
            messages = [
                {"role": "system", "content": "Solve the math problem step by step. End your answer with 'The answer is [number]'."},
            ]
            for j in range(num_shots):
                messages.append({"role": "user", "content": f"Question: {train_data[j]['question']}"})
                messages.append({"role": "assistant", "content": train_data[j]["answer"]})
            messages.append({"role": "user", "content": f"Question: {test_data[i]['question']}"})
            answer, tokens = await call_vllm_api(
                session=session,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=None,
                url=base_url,
                seed=seed,
                extra_params={
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "presence_penalty": presence_penalty,
                    "repetition_penalty": repetition_penalty,
                },
            )
            states[i] = answer
            output_tokens[i] = tokens
            return answer, tokens

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1200)
        ) as session:
            tasks = [get_answer(session, i) for i in range(num_questions)]
            await tqdm.gather(*tasks, desc="Evaluating")

        return states, output_tokens

    print(f"Running GSM8K evaluation: {num_questions} questions, {num_shots}-shot")

    tic = time.perf_counter()
    states, output_tokens = asyncio.run(run_async_evaluation())
    latency = time.perf_counter() - tic

    # Compute metrics
    preds = [get_answer_value(state) for state in states]
    accuracy = np.mean(np.array(preds) == np.array(labels))
    invalid_rate = np.mean(np.array(preds) == INVALID)
    total_output_tokens = sum(output_tokens)

    # Print up to 5 INVALID responses for debugging
    invalid_indices = [i for i, p in enumerate(preds) if p == INVALID]
    if invalid_indices:
        print(f"\n=== {len(invalid_indices)} INVALID responses, showing first 5 ===")
        for idx in invalid_indices[:5]:
            print(f"\n--- Question {idx} (expected: {labels[idx]}) ---")
            print(f"Raw response: {states[idx]!r}")
            print("---")

    # Print up to 5 WRONG (but valid) responses for debugging
    wrong_indices = [i for i, p in enumerate(preds) if p != INVALID and p != labels[i]]
    if wrong_indices:
        print(f"\n=== {len(wrong_indices)} WRONG responses, showing first 5 ===")
        for idx in wrong_indices[:5]:
            print(f"\n--- Question {idx} (expected: {labels[idx]}, got: {preds[idx]}) ---")
            print(f"Raw response: {states[idx]!r}")
            print("---")
    tokens_per_second = total_output_tokens / latency if latency > 0 else 0.0

    result = {
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "latency": latency,
        "questions_per_second": num_questions / latency,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "num_questions": num_questions,
        "num_shots": num_shots,
        "max_tokens": max_tokens,
        "timestamp": time.time(),
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K evaluation for vLLM serve")
    parser.add_argument(
        "--num-shots", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1319,
        help="Number of questions to evaluate",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens for generation"
    )
    parser.add_argument("--host", type=str, default="http://127.0.0.1", help="Host URL")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling (-1 to disable)")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    result = evaluate_gsm8k(
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    # Print results to terminal
    print("\nResults:")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Invalid responses: {result['invalid_rate']:.3f}")
    print(f"Total latency: {result['latency']:.3f} s")
    print(f"Questions per second: {result['questions_per_second']:.3f}")
    print(f"Total output tokens: {result['total_output_tokens']}")
    print(f"Output tokens per second: {result['tokens_per_second']:.3f}")

    # Optional file saving
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
