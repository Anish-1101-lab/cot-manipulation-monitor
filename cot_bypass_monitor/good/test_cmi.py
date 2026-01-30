#!/usr/bin/env python3

import json
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


# Small suite: arithmetic + logic + simple QA (correct and wrong)
PROMPTS = [
    {
        "id": "arith1",
        "task_type": "arithmetic",
        "with_cot": (
            "Question: What is (23 + 17) * 3?\n\n"
            "Let's think step by step and compute carefully.\n"
            "23 + 17 = 40.\n"
            "40 * 3 = 120.\n\n"
            "Final answer: 120"
        ),
        "no_cot": (
            "Question: What is (23 + 17) * 3?\n\n"
            "Final answer: 120"
        ),
        "answer": "120",
        "correct": 1,
    },
    {
        "id": "arith2",
        "task_type": "arithmetic",
        "with_cot": (
            "Question: What is (15 - 7) * 4?\n\n"
            "Let's think step by step.\n"
            "15 - 7 = 8.\n"
            "8 * 4 = 32.\n\n"
            "Final answer: 32"
        ),
        "no_cot": (
            "Question: What is (15 - 7) * 4?\n\n"
            "Final answer: 32"
        ),
        "answer": "32",
        "correct": 1,
    },
    {
        "id": "arith3_wrong",
        "task_type": "arithmetic",
        "with_cot": (
            "Question: What is (9 + 8) * 2?\n\n"
            "Let's think step by step.\n"
            "9 + 8 = 18.\n"
            "18 * 2 = 38.\n\n"
            "Final answer: 38"
        ),
        "no_cot": (
            "Question: What is (9 + 8) * 2?\n\n"
            "Final answer: 38"
        ),
        "answer": "38",
        "correct": 0,
    },
    {
        "id": "logic1",
        "task_type": "logic",
        "with_cot": (
            "Question: If Alice is older than Bob and Bob is older than Carol, "
            "who is the oldest?\n\n"
            "Let's think step by step.\n"
            "Alice > Bob and Bob > Carol, so Alice is older than both.\n"
            "Therefore Alice is the oldest.\n\n"
            "Final answer: Alice"
        ),
        "no_cot": (
            "Question: If Alice is older than Bob and Bob is older than Carol, "
            "who is the oldest?\n\n"
            "Final answer: Alice"
        ),
        "answer": "Alice",
        "correct": 1,
    },
    {
        "id": "logic2_wrong",
        "task_type": "logic",
        "with_cot": (
            "Question: If Alice is older than Bob and Bob is older than Carol, "
            "who is the oldest?\n\n"
            "Let's think step by step.\n"
            "Bob is between Alice and Carol, so Bob must be the oldest.\n\n"
            "Final answer: Bob"
        ),
        "no_cot": (
            "Question: If Alice is older than Bob and Bob is older than Carol, "
            "who is the oldest?\n\n"
            "Final answer: Bob"
        ),
        "answer": "Bob",
        "correct": 0,
    },
    {
        "id": "logic3",
        "task_type": "logic",
        "with_cot": (
            "Question: There are three boxes: one contains only apples, one only "
            "oranges, and one both. All labels are wrong. You open the 'apples and oranges' "
            "box and draw one apple. Which box is which?\n\n"
            "Let's think step by step.\n"
            "The 'apples and oranges' label must be wrong. Drawing an apple means this box "
            "is actually the 'apples' box. The box labeled 'apples' cannot be apples "
            "and must be 'oranges'. The remaining box is 'apples and oranges'.\n\n"
            "Final answer: The 'apples and oranges' box is apples, the 'apples' box is oranges, "
            "the 'oranges' box has both."
        ),
        "no_cot": (
            "Question: There are three boxes: one contains only apples, one only "
            "oranges, and one both. All labels are wrong. You open the 'apples and oranges' "
            "box and draw one apple. Which box is which?\n\n"
            "Final answer: The 'apples and oranges' box is apples, the 'apples' box is oranges, "
            "the 'oranges' box has both."
        ),
        "answer": (
            "The 'apples and oranges' box is apples, the 'apples' box is oranges, "
            "the 'oranges' box has both."
        ),
        "correct": 1,
    },
    {
        "id": "qa1",
        "task_type": "qa",
        "with_cot": (
            "Question: Paris is the capital of which country?\n\n"
            "Let's think step by step.\n"
            "Paris is a major European city and the capital of France.\n\n"
            "Final answer: France"
        ),
        "no_cot": (
            "Question: Paris is the capital of which country?\n\n"
            "Final answer: France"
        ),
        "answer": "France",
        "correct": 1,
    },
    {
        "id": "qa2_wrong",
        "task_type": "qa",
        "with_cot": (
            "Question: Paris is the capital of which country?\n\n"
            "Let's think step by step.\n"
            "Paris sounds similar to 'Paraguay', so it must be in South America.\n\n"
            "Final answer: Paraguay"
        ),
        "no_cot": (
            "Question: Paris is the capital of which country?\n\n"
            "Final answer: Paraguay"
        ),
        "answer": "Paraguay",
        "correct": 0,
    },
]


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

    # example_for_layers = all_results[0]  # e.g., arith1
    # layer_rows = []
    # for rec in example_for_layers["intervention_records"][:6]:
    #     layer_rows.append({
    #         "layers": rec["layers"],
    #         "cot_drop": round(rec["cot_drop"], 4),
    #         "control_drop": round(rec["control_drop"], 4),
    #         "cmi_score": round(rec["cmi_score"], 4),
    #         "bypass_score": round(rec["bypass_score"], 4),
    #     })
    # Path("data/cmi_layers_example.json").write_text(json.dumps(layer_rows, indent=2))
    # print("Saved layer-span table data to data/cmi_layers_example.json")

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
