import anthropic
import time
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

TEST_PROMPTS = [
    {"id": "Q01", "category": "Factual-Simple", "prompt": "What is the capital of France?", "expected": "Paris"},
    {"id": "Q02", "category": "Factual-Simple", "prompt": "What planet is closest to the Sun?", "expected": "Mercury"},
    {"id": "Q03", "category": "Factual-Simple", "prompt": "Who wrote the play Romeo and Juliet?", "expected": "Shakespeare"},
    {"id": "Q04", "category": "Factual-Moderate", "prompt": "What is photosynthesis?", "expected": "Plants convert sunlight to glucose/oxygen"},
    {"id": "Q05", "category": "Factual-Moderate", "prompt": "Explain what HTTP status code 404 means.", "expected": "Not Found"},
    {"id": "Q06", "category": "Technical-CS", "prompt": "What is the difference between a stack and a queue in data structures?", "expected": "Stack=LIFO, Queue=FIFO"},
    {"id": "Q07", "category": "Technical-CS", "prompt": "What is the time complexity of binary search?", "expected": "O(log n)"},
    {"id": "Q08", "category": "Technical-ML", "prompt": "What is PCA in machine learning?", "expected": "Principal Component Analysis, dimensionality reduction"},
    {"id": "Q09", "category": "Technical-ML", "prompt": "What is overfitting in machine learning?", "expected": "Model learns training data too well, poor on new data"},
    {"id": "Q10", "category": "Reasoning-Math", "prompt": "If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?", "expected": "150 miles"},
    {"id": "Q11", "category": "Reasoning-Math", "prompt": "What is 15% of 200?", "expected": "30"},
    {"id": "Q12", "category": "Reasoning-Logic", "prompt": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?", "expected": "No"},
    {"id": "Q13", "category": "Reasoning-Logic", "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "expected": "9"},
    {"id": "Q14", "category": "Code", "prompt": "Write a Python function that takes a list of numbers and returns the largest number.", "expected": "Working Python code"},
    {"id": "Q15", "category": "Code", "prompt": "Write a Python function to check if a string is a palindrome.", "expected": "Working Python code"},
    {"id": "Q16", "category": "Summarization", "prompt": "Summarize the concept of supply and demand in economics in 2-3 sentences.", "expected": "Price relationship with supply/demand"},
    {"id": "Q17", "category": "Instruction", "prompt": "List exactly 3 programming languages that start with the letter P.", "expected": "Exactly 3 languages starting with P"},
    {"id": "Q18", "category": "Instruction", "prompt": "Explain what an API is in exactly one sentence.", "expected": "One sentence about software communication"},
    {"id": "Q19", "category": "Long-Response", "prompt": "Explain how the internet works, covering at least DNS, TCP/IP, and HTTP.", "expected": "Covers DNS, TCP/IP, HTTP"},
    {"id": "Q20", "category": "Long-Response", "prompt": "Compare and contrast supervised learning and unsupervised learning in machine learning.", "expected": "Labeled vs unlabeled data"},
]


def benchmark_anthropic(model_name="claude-haiku-4-5-20251001"):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    print(f"\n{'='*60}")
    print(f"Running Anthropic Benchmark - Model: {model_name}")
    print(f"{'='*60}\n")

    for i, test in enumerate(TEST_PROMPTS):
        print(f"  [{i+1}/20] {test['id']}: {test['category']}...", end=" ", flush=True)

        start_time = time.time()
        first_token_time = None
        token_count = 0
        full_response = ""

        try:
            with client.messages.stream(
                model=model_name,
                messages=[{"role": "user", "content": test["prompt"]}],
                temperature=0.0,
                max_tokens=2048,
            ) as stream:
                for text in stream.text_stream:
                    if text and first_token_time is None:
                        first_token_time = time.time()
                    if text:
                        token_count += len(text.split())
                        full_response += text

            end_time = time.time()

            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            total_time = (end_time - start_time) * 1000
            decode_time = (end_time - first_token_time) * 1000 if first_token_time else 0
            tokens_per_sec = token_count / (decode_time / 1000) if decode_time > 0 else 0

            est_input_tokens = len(test["prompt"]) // 4
            input_cost = est_input_tokens * 0.0000008
            output_cost = token_count * 0.000004
            query_cost = input_cost + output_cost

            results.append({
                "id": test["id"],
                "category": test["category"],
                "prompt": test["prompt"],
                "expected": test["expected"],
                "response": full_response,
                "ttft_ms": round(ttft, 0),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "token_count": token_count,
                "total_time_s": round(total_time / 1000, 2),
                "decode_time_s": round(decode_time / 1000, 2),
                "response_length": len(full_response),
                "cost_usd": round(query_cost, 6),
                "model": model_name,
                "provider": "Anthropic",
                "error": "",
            })

            print(f"TTFT: {ttft:.0f}ms | TPS: {tokens_per_sec:.1f} | Tokens: {token_count} | Cost: ${query_cost:.6f}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": test["id"], "category": test["category"],
                "prompt": test["prompt"], "expected": test["expected"],
                "response": "", "ttft_ms": 0, "tokens_per_sec": 0,
                "token_count": 0, "total_time_s": 0, "decode_time_s": 0,
                "response_length": 0, "cost_usd": 0, "model": model_name,
                "provider": "Anthropic", "error": str(e),
            })

        time.sleep(0.5)

    return results


def save_csv(results, filename):
    if not results:
        return
    fields = ["id", "category", "prompt", "expected", "response", "ttft_ms",
              "tokens_per_sec", "token_count", "total_time_s", "decode_time_s",
              "response_length", "cost_usd", "model", "provider", "error"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Saved to: {filename}")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("  Anthropic Claude Benchmark")
    print("  20 standardized queries | temp: 0.0 | max: 2048")
    print("=" * 60)

    results = benchmark_anthropic(model_name="claude-haiku-4-5-20251001")
    if results:
        valid = [r for r in results if int(r['token_count']) > 0]
        ttfts = [r['ttft_ms'] for r in valid]
        times = [r['total_time_s'] for r in valid]
        tokens = [r['token_count'] for r in valid]
        costs = [r['cost_usd'] for r in valid]

        print(f"\n{'='*60}")
        print(f"  SUMMARY - Claude Haiku 4.5")
        print(f"{'='*60}")
        print(f"  Avg TTFT:        {sum(ttfts)/len(ttfts):.0f} ms")
        print(f"  Avg Time/Query:  {sum(times)/len(times):.2f} s")
        print(f"  Avg Tokens/Query:{sum(tokens)/len(tokens):.0f}")
        print(f"  Total Cost:      ${sum(costs):.6f}")

        save_csv(results, f"benchmark_claude_haiku_{timestamp}.csv")

    print("\n  Done!")


if __name__ == "__main__":
    main()