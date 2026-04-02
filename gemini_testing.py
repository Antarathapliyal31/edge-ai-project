import time
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ========== Test Prompts (same 20 as browser benchmark) ==========
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


def benchmark_gemini(model_name="gemini-2.0-flash"):
    try:
        from google import genai
    except ImportError:
        print("ERROR: Install google-genai package first:")
        print("  pip install google-genai")
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        print("  export GEMINI_API_KEY='your-key-here'")
        return []

    client = genai.Client(api_key=api_key)
    results = []

    print(f"\n{'='*60}")
    print(f"Running Gemini Benchmark - Model: {model_name}")
    print(f"{'='*60}\n")

    for i, test in enumerate(TEST_PROMPTS):
        print(f"  [{i+1}/20] {test['id']}: {test['category']}...", end=" ", flush=True)

        start_time = time.time()
        first_token_time = None
        token_count = 0
        full_response = ""

        try:
            response = client.models.generate_content_stream(
                model=model_name,
                contents=test["prompt"],
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 256,
                },
            )

            for chunk in response:
                text = chunk.text if chunk.text else ""
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
            input_cost = est_input_tokens * 0.0000001
            output_cost = token_count * 0.0000004
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
                "provider": "Google",
                "error": None,
            })

            print(f"TTFT: {ttft:.0f}ms | TPS: {tokens_per_sec:.1f} | Tokens: {token_count} | Cost: ${query_cost:.6f}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": test["id"],
                "category": test["category"],
                "prompt": test["prompt"],
                "expected": test["expected"],
                "response": "",
                "ttft_ms": 0, "tokens_per_sec": 0, "token_count": 0,
                "total_time_s": 0, "decode_time_s": 0, "response_length": 0,
                "cost_usd": 0, "model": model_name, "provider": "Google",
                "error": str(e),
            })

        time.sleep(0.5)

    return results


def print_summary(results):
    if not results:
        return

    valid = [r for r in results if r["token_count"] > 0]
    if not valid:
        print("\n  No valid results")
        return

    ttfts = [r["ttft_ms"] for r in valid]
    tps_list = [r["tokens_per_sec"] for r in valid]
    total_tokens = sum(r["token_count"] for r in valid)
    total_cost = sum(r["cost_usd"] for r in valid)
    total_times = [r["total_time_s"] for r in valid]

    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY - {valid[0]['provider']} ({valid[0]['model']})")
    print(f"{'='*60}")
    print(f"")
    print(f"  --- LATENCY ---")
    print(f"  Avg TTFT:            {sum(ttfts)/len(ttfts):.0f} ms")
    print(f"  Min TTFT:            {min(ttfts):.0f} ms")
    print(f"  Max TTFT:            {max(ttfts):.0f} ms")
    print(f"")
    print(f"  --- THROUGHPUT ---")
    print(f"  Avg Tokens/Sec:      {sum(tps_list)/len(tps_list):.1f}")
    print(f"  Min Tokens/Sec:      {min(tps_list):.1f}")
    print(f"  Max Tokens/Sec:      {max(tps_list):.1f}")
    print(f"  Total Tokens:        {total_tokens}")
    print(f"")
    print(f"  --- TIME ---")
    print(f"  Avg Response Time:   {sum(total_times)/len(total_times):.2f} s")
    print(f"  Total Benchmark Time:{sum(total_times):.1f} s")
    print(f"")
    print(f"  --- COST ---")
    print(f"  Total Cost:          ${total_cost:.4f}")
    print(f"  Avg Cost/Query:      ${total_cost/len(valid):.6f}")
    print(f"  Cost for 10k queries:${total_cost / len(valid) * 10000:.2f}")
    print(f"")
    print(f"  --- COMPARISON ---")
    print(f"  Browser (WebGPU):    $0.00 for same 20 queries")
    print(f"  This API:            ${total_cost:.4f} for same 20 queries")
    print(f"{'='*60}\n")


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

    print(f"  Results saved to: {filename}")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("  EDGE AI PROJECT - Server-Side API Benchmark")
    print("  Running 20 standardized queries via Gemini API")
    print("  Temperature: 0.0 | Max Tokens: 256")
    print("=" * 60)

    # Run Gemini 2.5 Flash
    print("\n[1/2] Gemini 2.5 Flash")
    flash_results = benchmark_gemini(model_name="gemini-2.5-flash")
    if flash_results:
        print_summary(flash_results)
        save_csv(flash_results, f"benchmark_gemini_2_5_flash_{timestamp}.csv")

    # Run Gemini 3 Flash Preview
    print("\n[2/2] Gemini 3 Flash Preview")
    lite_results = benchmark_gemini(model_name="gemini-3-flash-preview")
    if lite_results:
        print_summary(lite_results)
        save_csv(lite_results, f"benchmark_gemini_3_flash_{timestamp}.csv")
        
    # Combined
    all_results = flash_results + lite_results
    if all_results:
        save_csv(all_results, f"benchmark_all_gemini_{timestamp}.csv")

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)
    if all_results:
        total_cost = sum(r["cost_usd"] for r in all_results)
        print(f"  Total queries: {len(all_results)}")
        print(f"  Total cost: ${total_cost:.4f}")
    print()


if __name__ == "__main__":
    main()