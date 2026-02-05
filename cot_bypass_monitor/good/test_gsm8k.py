#!/usr/bin/env python3
"""
CMI (Chain-of-thought Manipulation Index) test suite.
Automatically loads prompts from data/gsm8k.jsonl and runs intervention analysis.
"""

import json
import re
from pathlib import Path
import torch

from cot_bypass_monitor import (
    tokenizer,
    DEVICE,
    find_cot_span_positions,
    run_intervention_analysis,
)


def build_ids(full_prompt: str):
    enc = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(DEVICE) for k, v in enc.items()}


def extract_final_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K answer format (#### <number>)."""
    match = re.search(r'####\s*(\d+)', answer_text)
    if match:
        return match.group(1)
    return answer_text.strip().split()[-1]


def parse_cot_steps(answer_text: str) -> str:
    """Parse the chain-of-thought steps from GSM8K answer format."""
    # Remove the calculator annotations like <<48/2=24>>
    cleaned = re.sub(r'<<[^>]+>>', '', answer_text)
    # Remove the final answer line
    cleaned = re.sub(r'####\s*\d+\s*$', '', cleaned)
    return cleaned.strip()


def load_gsm8k_prompts(jsonl_path: str, num_prompts: int = 20) -> list:
    """
    Load prompts from a GSM8K JSONL file and convert them to the required format.
    
    Args:
        jsonl_path: Path to the JSONL file
        num_prompts: Number of prompts to load (default: 20)
    
    Returns:
        List of prompt dictionaries with with_cot, no_cot, answer, etc.
    """
    prompts = []
    jsonl_file = Path(jsonl_path)
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"GSM8K file not found: {jsonl_path}")
    
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_prompts:
                break
            
            data = json.loads(line.strip())
            question = data['question']
            answer_text = data['answer']
            
            # Extract final answer and CoT steps
            final_answer = extract_final_answer(answer_text)
            cot_steps = parse_cot_steps(answer_text)
            
            # Create prompt with CoT
            prompt_with_cot = (
                f"Question: {question}\n\n"
                f"Let's think step by step.\n"
                f"{cot_steps}\n\n"
                f"Final answer: {final_answer}"
            )
            
            # Create prompt without CoT
            prompt_no_cot = (
                f"Question: {question}\n\n"
                f"Final answer: {final_answer}"
            )
            
            prompts.append({
                "id": f"gsm_{i+1:03d}",
                "task_type": "arithmetic",
                "with_cot": prompt_with_cot,
                "no_cot": prompt_no_cot,
                "answer": final_answer,
                "correct": 1,  # GSM8K answers are correct by definition
            })
    
    print(f"Loaded {len(prompts)} prompts from {jsonl_path}")
    return prompts


def run_single(example):
    prompt_with_cot = example["with_cot"]
    prompt_no_cot = example["no_cot"]
    answer_str = example["answer"]

    ids_with_cot = build_ids(prompt_with_cot)
    ids_without_cot = build_ids(prompt_no_cot)

    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    answer_ids = torch.tensor(answer_ids, device=DEVICE)

    cot_span = "Let's think"
    cot_positions = find_cot_span_positions(
        full_prompt=prompt_with_cot,
        cot_span=cot_span,
        tokenizer=tokenizer,
    )
    if not cot_positions:
        full_ids = ids_with_cot["input_ids"][0]
        cot_positions = list(range(max(0, len(full_ids) - 7), len(full_ids)))

    print(f"\n=== Example {example['id']} ({example['task_type']}) ===")
    print("CoT token positions:", cot_positions)

    result = run_intervention_analysis(
        ids_with_cot=ids_with_cot,
        ids_without_cot=ids_without_cot,
        answer_ids=answer_ids,
        cot_positions=cot_positions,
        num_layers=None,
        control_samples=8,
        span_samples=8,
        max_span_size=3,
        layer_spans=None,
    )

    print("\nCMI result:")
    print(f"  Baseline logP: {result['baseline_logp']:.4f}")
    print(f"  CMI mean:      {result['cmi_mean']:.4f}")
    print(f"  Bypass mean:   {result['bypass_mean']:.4f}")
    print(f"  No-CoT-effect: {result['no_cot_effect']}")

    print("\nPer-span records (first 4):")
    for rec in result["intervention_records"][:4]:
        print(
            f"  Layers {rec['layers']}: "
            f"cot_drop={rec['cot_drop']:.4f}, "
            f"ctrl_drop={rec['control_drop']:.4f}, "
            f"cmi={rec['cmi_score']:.4f}, "
            f"bypass={rec['bypass_score']:.4f}"
        )

    return {
        "id": example["id"],
        "task_type": example["task_type"],
        "correct": example["correct"],
        "prompt_with_cot": prompt_with_cot,
        "prompt_no_cot": prompt_no_cot,
        "answer": answer_str,
        "cot_positions": cot_positions,
        "baseline_logp": result["baseline_logp"],
        "cmi_mean": result["cmi_mean"],
        "bypass_mean": result["bypass_mean"],
        "no_cot_effect": result["no_cot_effect"],
        "intervention_records": result["intervention_records"],
    }


def main():
    Path("data").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    # Load prompts from GSM8K JSONL file
    gsm8k_path = Path("data/gsm8k.jsonl")
    PROMPTS = load_gsm8k_prompts(gsm8k_path, num_prompts=20)

    all_results = []
    for ex in PROMPTS:
        res = run_single(ex)
        all_results.append(res)

    Path("data/cmi_suite.json").write_text(json.dumps(all_results, indent=2))
    print("\nSaved all results to data/cmi_suite.json")

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "ID": r["id"],
            "Task": r["task_type"],
            "Correct": "✓" if r["correct"] else "✗",
            "CMI_mean": round(r["cmi_mean"], 3),
            "Bypass": round(r["bypass_mean"], 3),
            "Baseline_logP": round(r["baseline_logp"], 3),
        })
    Path("data/cmi_summary_table.json").write_text(json.dumps(summary_rows, indent=2))
    print("Saved summary table data to data/cmi_summary_table.json")

    for r in all_results:
        layer_rows = []
        for rec in r["intervention_records"]:
            layer_rows.append({
                "layers": rec["layers"],
                "cot_drop": round(rec["cot_drop"], 4),
                "control_drop": round(rec["control_drop"], 4),
                "cmi_score": round(rec["cmi_score"], 4),
                "bypass_score": round(rec["bypass_score"], 4),
            })
        out_path = Path(f"data/cmi_layers_{r['id']}.json")
        out_path.write_text(json.dumps(layer_rows, indent=2))
        print(f"Saved layer-span table data for {r['id']} to {out_path}")


if __name__ == "__main__":
    main()