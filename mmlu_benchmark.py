import os
import re
import csv
import json
import time
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
from datasets import load_dataset
from google import genai
from google.genai import types as genai_types
import anthropic

load_dotenv()

QUESTIONS_PER_SUBJECT = 2
TEMPERATURE = 0.0
MAX_TOKENS = 10
SLEEP_BETWEEN_QUERIES = 0.5

LETTERS = ["A", "B", "C", "D"]


def build_prompt(question, choices):
    lines = [f"Question: {question}"]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}) {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D).")
    return "\n".join(lines)


def load_mmlu_subset():
    print("Loading MMLU dataset (cais/mmlu, 'all' config, 'test' split)...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    per_subject = defaultdict(list)
    for row in ds:
        subject = row["subject"]
        if len(per_subject[subject]) < QUESTIONS_PER_SUBJECT:
            per_subject[subject].append(row)

    questions = []
    for subject in sorted(per_subject.keys()):
        for row in per_subject[subject]:
            questions.append({
                "subject": subject,
                "question": row["question"],
                "choices": list(row["choices"]),
                "correct_answer": LETTERS[row["answer"]],
                "prompt": build_prompt(row["question"], row["choices"]),
            })

    print(f"  Loaded {len(questions)} questions across {len(per_subject)} subjects\n")
    return questions


def export_browser_prompts(questions, filename):
    payload = [
        {
            "subject": q["subject"],
            "question": q["question"],
            "choices": q["choices"],
            "correct_answer": q["correct_answer"],
            "prompt": q["prompt"],
        }
        for q in questions
    ]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Exported browser prompts to: {filename}\n")


def extract_letter(text):
    if not text:
        return ""
    match = re.search(r"[A-D]", text.upper())
    return match.group(0) if match else ""


def run_gemini(model_name, questions):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"ERROR: Set GEMINI_API_KEY to run {model_name}")
        return []

    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )

    print(f"\n{'='*60}")
    print(f"Running Gemini benchmark - Model: {model_name}")
    print(f"{'='*60}\n")

    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['subject']}...", end=" ", flush=True)

        start = time.time()
        first_token_time = None
        text_out = ""

        try:
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=q["prompt"],
                config=config,
            )
            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    if first_token_time is None:
                        first_token_time = time.time()
                    text_out += chunk_text

            end = time.time()
            ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
            total_s = end - start
            model_letter = extract_letter(text_out)
            is_correct = model_letter == q["correct_answer"]
            exact_match = text_out.strip() in LETTERS

            results.append({
                "subject": q["subject"],
                "question": q["question"],
                "choices": " | ".join(q["choices"]),
                "correct_answer": q["correct_answer"],
                "model_answer": model_letter,
                "is_correct": is_correct,
                "exact_match": exact_match,
                "ttft_ms": round(ttft_ms, 0),
                "total_time_s": round(total_s, 3),
                "response_text": text_out.strip(),
                "model": model_name,
                "error": "",
            })

            mark = "OK " if is_correct else "X  "
            print(f"{mark} ans={model_letter} truth={q['correct_answer']} | TTFT: {ttft_ms:.0f}ms | {total_s:.2f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "subject": q["subject"],
                "question": q["question"],
                "choices": " | ".join(q["choices"]),
                "correct_answer": q["correct_answer"],
                "model_answer": "",
                "is_correct": False,
                "exact_match": False,
                "ttft_ms": 0,
                "total_time_s": 0,
                "response_text": "",
                "model": model_name,
                "error": str(e),
            })

        time.sleep(SLEEP_BETWEEN_QUERIES)

    return results


def run_anthropic(model_name, questions):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"ERROR: Set ANTHROPIC_API_KEY to run {model_name}")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    print(f"\n{'='*60}")
    print(f"Running Anthropic benchmark - Model: {model_name}")
    print(f"{'='*60}\n")

    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['subject']}...", end=" ", flush=True)

        start = time.time()
        first_token_time = None
        text_out = ""

        try:
            with client.messages.stream(
                model=model_name,
                messages=[{"role": "user", "content": q["prompt"]}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        if first_token_time is None:
                            first_token_time = time.time()
                        text_out += text

            end = time.time()
            ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
            total_s = end - start
            model_letter = extract_letter(text_out)
            is_correct = model_letter == q["correct_answer"]
            exact_match = text_out.strip() in LETTERS

            results.append({
                "subject": q["subject"],
                "question": q["question"],
                "choices": " | ".join(q["choices"]),
                "correct_answer": q["correct_answer"],
                "model_answer": model_letter,
                "is_correct": is_correct,
                "exact_match": exact_match,
                "ttft_ms": round(ttft_ms, 0),
                "total_time_s": round(total_s, 3),
                "response_text": text_out.strip(),
                "model": model_name,
                "error": "",
            })

            mark = "OK " if is_correct else "X  "
            print(f"{mark} ans={model_letter} truth={q['correct_answer']} | TTFT: {ttft_ms:.0f}ms | {total_s:.2f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "subject": q["subject"],
                "question": q["question"],
                "choices": " | ".join(q["choices"]),
                "correct_answer": q["correct_answer"],
                "model_answer": "",
                "is_correct": False,
                "exact_match": False,
                "ttft_ms": 0,
                "total_time_s": 0,
                "response_text": "",
                "model": model_name,
                "error": str(e),
            })

        time.sleep(SLEEP_BETWEEN_QUERIES)

    return results


def save_csv(results, filename):
    if not results:
        return
    fields = ["subject", "question", "choices", "correct_answer", "model_answer",
              "is_correct", "exact_match", "ttft_ms", "total_time_s",
              "response_text", "model", "error"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Saved to: {filename}")


def compute_metrics(results):
    valid = [r for r in results if not r["error"]]
    if not valid:
        return None

    correct = sum(1 for r in valid if r["is_correct"])
    exact = sum(1 for r in valid if r["exact_match"])
    ttfts = [r["ttft_ms"] for r in valid if r["ttft_ms"] > 0]
    times = [r["total_time_s"] for r in valid if r["total_time_s"] > 0]

    per_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in valid:
        per_subject[r["subject"]]["total"] += 1
        if r["is_correct"]:
            per_subject[r["subject"]]["correct"] += 1

    return {
        "accuracy": correct / len(valid) if valid else 0,
        "exact_match_rate": exact / len(valid) if valid else 0,
        "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
        "min_ttft_ms": min(ttfts) if ttfts else 0,
        "max_ttft_ms": max(ttfts) if ttfts else 0,
        "avg_total_time_s": sum(times) / len(times) if times else 0,
        "per_subject": per_subject,
        "n": len(valid),
    }


def print_model_summary(model_label, metrics):
    print(f"\n{'='*60}")
    print(f"  SUMMARY - {model_label}")
    print(f"{'='*60}")
    if not metrics:
        print("  No valid results")
        return
    print(f"  Questions evaluated: {metrics['n']}")
    print(f"  Accuracy:            {metrics['accuracy']*100:.1f}%")
    print(f"  Exact match rate:    {metrics['exact_match_rate']*100:.1f}%")
    print(f"  Avg TTFT:            {metrics['avg_ttft_ms']:.0f} ms "
          f"(min {metrics['min_ttft_ms']:.0f} / max {metrics['max_ttft_ms']:.0f})")
    print(f"  Avg time/query:      {metrics['avg_total_time_s']:.2f} s")


def print_comparison_table(model_metrics):
    labels = list(model_metrics.keys())
    if not labels:
        return

    all_subjects = set()
    for metrics in model_metrics.values():
        if metrics:
            all_subjects.update(metrics["per_subject"].keys())
    subjects = sorted(all_subjects)

    print(f"\n{'='*60}")
    print(f"  PER-SUBJECT COMPARISON")
    print(f"{'='*60}")

    col_width = 14
    header = f"{'Subject':<32}" + "".join(f"{label[:col_width-1]:<{col_width}}" for label in labels)
    print(header)
    print("-" * len(header))

    for subject in subjects:
        row = f"{subject[:31]:<32}"
        for label in labels:
            m = model_metrics[label]
            if m and subject in m["per_subject"]:
                s = m["per_subject"][subject]
                pct = (s["correct"] / s["total"] * 100) if s["total"] else 0
                row += f"{pct:>6.0f}%       "[:col_width]
            else:
                row += f"{'-':<{col_width}}"
        print(row)

    print("-" * len(header))
    overall_row = f"{'OVERALL':<32}"
    for label in labels:
        m = model_metrics[label]
        if m:
            overall_row += f"{m['accuracy']*100:>6.0f}%       "[:col_width]
        else:
            overall_row += f"{'-':<{col_width}}"
    print(overall_row)


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("  MMLU Quality Benchmark")
    print(f"  {QUESTIONS_PER_SUBJECT} questions/subject | temp: {TEMPERATURE} | max_tokens: {MAX_TOKENS}")
    print("=" * 60 + "\n")

    questions = load_mmlu_subset()
    export_browser_prompts(questions, "mmlu_browser_prompts.json")

    runs = [
        ("Gemini 2.5 Flash", "gemini-2.5-flash", run_gemini),
        ("Gemini 3 Flash", "gemini-3-flash-preview", run_gemini),
        ("Claude Haiku 4.5", "claude-haiku-4-5-20251001", run_anthropic),
    ]

    model_metrics = {}
    for label, model_name, runner in runs:
        results = runner(model_name, questions)
        if not results:
            model_metrics[label] = None
            continue

        safe_name = model_name.replace("-", "_").replace(".", "_")
        save_csv(results, f"mmlu_{safe_name}_{timestamp}.csv")

        metrics = compute_metrics(results)
        model_metrics[label] = metrics
        print_model_summary(label, metrics)

    print_comparison_table(model_metrics)
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
