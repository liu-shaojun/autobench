import asyncio
import json
import time
import aiohttp
import numpy as np
import regex as re
import ast
from tqdm.asyncio import tqdm

INVALID = -9999999

def download_and_cache_file(url, filename=None):
    import os, urllib.request
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

def load_gsm8k_data():
    train_file = download_and_cache_file("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl")
    test_file = download_and_cache_file("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl")
    train_data = [json.loads(l) for l in open(train_file)]
    test_data = [json.loads(l) for l in open(test_file)]
    return train_data, test_data

def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID

async def call_chat_api(session, prompt, temperature, max_tokens, url, model_name="model", seed=None):
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        data["seed"] = seed
    try:
        async with session.post(f"{url}/v1/chat/completions", json=data) as response:
            response.raise_for_status()
            result = await response.json()
            text = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return text, tokens
    except Exception as e:
        print(f"Error: {e}")
        return "", 0

async def main():
    import argparse
    p = argparse.ArgumentParser(description="GSM8K evaluation (thinking disabled)")
    p.add_argument("--num-questions", type=int, default=300)
    p.add_argument("--num-shots", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--host", default="http://localhost")
    p.add_argument("--port", type=int, default=9005)
    p.add_argument("--model-name", type=str, default="model")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base_url = f"{args.host}:{args.port}"
    train_data, test_data = load_gsm8k_data()
    num_questions = min(args.num_questions, len(test_data))

    few_shot = ""
    for i in range(args.num_shots):
        few_shot += f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['answer']}\n\n"

    questions = []
    labels = []
    for i in range(num_questions):
        questions.append(f"{few_shot}Question: {test_data[i]['question']}\nAnswer:")
        labels.append(get_answer_value(test_data[i]["answer"]))

    states = [""] * num_questions
    output_tokens = [0] * num_questions

    async def get_answer(session, i):
        text, tokens = await call_chat_api(session, questions[i], args.temperature, args.max_tokens, base_url, args.model_name, args.seed)
        states[i] = text
        output_tokens[i] = tokens
        return text, tokens

    print(f"Running GSM8K evaluation (thinking disabled): {num_questions} questions, {args.num_shots}-shot")

    start = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
        tasks = [get_answer(session, i) for i in range(num_questions)]
        await tqdm.gather(*tasks)
    latency = time.time() - start

    preds = [get_answer_value(s) for s in states]
    accuracy = np.mean(np.array(preds) == np.array(labels))
    invalid_rate = np.mean(np.array(preds) == INVALID)
    total_tokens = sum(output_tokens)

    # Print up to 5 INVALID responses for debugging
    invalid_indices = [i for i, pred in enumerate(preds) if pred == INVALID]
    if invalid_indices:
        print(f"\n=== {len(invalid_indices)} INVALID responses, showing first 5 ===")
        for idx in invalid_indices[:5]:
            print(f"\n--- Question {idx} (expected: {labels[idx]}) ---")
            print(f"Raw response: {states[idx]!r}")
            print("---")

    # Print up to 5 WRONG (but valid) responses for debugging
    wrong_indices = [i for i, pred in enumerate(preds) if pred != INVALID and pred != labels[i]]
    if wrong_indices:
        print(f"\n=== {len(wrong_indices)} WRONG responses, showing first 5 ===")
        for idx in wrong_indices[:5]:
            print(f"\n--- Question {idx} (expected: {labels[idx]}, got: {preds[idx]}) ---")
            print(f"Raw response: {states[idx]!r}")
            print("---")

    print(f"\nResults (thinking disabled):")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Invalid responses: {invalid_rate:.3f}")
    print(f"Total latency: {latency:.3f} s")
    print(f"Questions per second: {num_questions/latency:.3f}")
    print(f"Total output tokens: {total_tokens}")
    print(f"Output tokens per second: {total_tokens/latency:.3f}")

asyncio.run(main())
