#!/usr/bin/env python3

import json
from pathlib import Path
import torch

from cot_bypass_monitor import (
    tokenizer,
    DEVICE,
    find_cot_span_positions,
    concat_answer,
    compute_answer_logprob,
    run_intervention_analysis,
)


def build_ids(full_prompt: str):
    enc = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(DEVICE) for k, v in enc.items()}


def load_strategyqa_top_n(n: int = 100):
    data_path = Path(__file__).parent / "data" / "strategy_qa.json"
    items = json.loads(data_path.read_text())
    examples = []
    for row in items[:n]:
        question = row.get("question", "").strip()
        answer = row.get("answer", "")
        if isinstance(answer, bool):
            answer_str = "Yes" if answer else "No"
        else:
            answer_str = str(answer)
        facts = row.get("facts") or []
        if facts:
            facts_block = "Facts:\n- " + "\n- ".join(facts)
        else:
            facts_block = ""
        decomposition = row.get("decomposition") or []
        if decomposition:
            decomposition_block = "Decomposition:\n- " + "\n- ".join(decomposition)
        else:
            decomposition_block = ""
        evidence = row.get("evidence")
        if evidence:
            evidence_block = "Evidence:\n" + json.dumps(evidence, ensure_ascii=True)
        else:
            evidence_block = ""

        with_cot = (
            f"Question: {question}\n"
            f"{facts_block}\n"
            f"{decomposition_block}\n"
            f"{evidence_block}\n"
            "Let's think step by step.\n"
            f"Final answer: {answer_str}"
        ).strip()
        no_cot = (
            f"Question: {question}\n"
            f"Final answer: {answer_str}"
        ).strip()

        examples.append(
            {
                "id": row.get("qid", f"strategyqa_{len(examples)+1}"),
                "task_type": "strategyqa",
                "with_cot": with_cot,
                "no_cot": no_cot,
                "answer": answer_str,
                "correct": None,
            }
        )
    return examples


def run_single(example):
    prompt_with_cot = example["with_cot"]
    prompt_no_cot = example["no_cot"]
    answer_str = example["answer"]

    ids_with_cot = build_ids(prompt_with_cot)
    ids_without_cot = build_ids(prompt_no_cot)

    answer_ids_list = tokenizer.encode(answer_str, add_special_tokens=False)
    answer_ids = torch.tensor(answer_ids_list, device=DEVICE)

    full_with_cot = concat_answer(ids_with_cot["input_ids"], answer_ids_list)
    full_without_cot = concat_answer(ids_without_cot["input_ids"], answer_ids_list)
    logp_with = compute_answer_logprob(full_with_cot, answer_ids_list)
    logp_no_cot = compute_answer_logprob(full_without_cot, answer_ids_list)

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

    from cot_bypass_monitor import get_model_layers, model
    num_layers = len(get_model_layers(model))
    layer_spans = [[i] for i in range(num_layers)]
    result = run_intervention_analysis(
        ids_with_cot=ids_with_cot,
        ids_without_cot=ids_without_cot,
        answer_ids=answer_ids,
        cot_positions=cot_positions,
        num_layers=num_layers,
        control_samples=8,
        span_samples=8,
        max_span_size=3,
        layer_spans=layer_spans,
    )

    print("\nCMI result:")
    print(f"  Baseline logP: {result['baseline_logp']:.4f}")
    print(f"  LogP (with CoT): {logp_with:.4f}")
    print(f"  LogP (no CoT):   {logp_no_cot:.4f}")
    print(f"  LogP Δ (with - no): {logp_with - logp_no_cot:.4f}")
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
        "logp_with_cot": logp_with,
        "logp_no_cot": logp_no_cot,
        "logp_with_minus_no": logp_with - logp_no_cot,
        "cmi_mean": result["cmi_mean"],
        "bypass_mean": result["bypass_mean"],
        "no_cot_effect": result["no_cot_effect"],
        "intervention_records": result["intervention_records"],
        "layerwise_records": result["intervention_records"],
        "layerwise_cmi_mean": result["cmi_mean"],
        "layerwise_bypass_mean": result["bypass_mean"],
    }


def main():
    Path("data").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    prompts = load_strategyqa_top_n(20)
    all_results = []
    for ex in prompts:
        res = run_single(ex)
        all_results.append(res)

    Path("data/cmi_suite.json").write_text(json.dumps(all_results, indent=2))
    print("\nSaved all results to data/cmi_suite.json")

    summary_rows = []
    for r in all_results:
        if r["correct"] is None:
            correct_marker = "?"
        else:
            correct_marker = "✓" if r["correct"] else "✗"
        summary_rows.append(
            {
                "ID": r["id"],
                "Task": r["task_type"],
                "Correct": correct_marker,
                "CMI_mean": round(r["cmi_mean"], 3),
                "Bypass": round(r["bypass_mean"], 3),
                "Baseline_logP": round(r["baseline_logp"], 3),
            }
        )
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

    span_tables = {}
    per_layer_tables = {}
    for r in all_results:
        layer_rows = []
        for rec in r["intervention_records"]:
            layer_rows.append(
                {
                    "layers": rec["layers"],
                    "cot_drop": round(rec["cot_drop"], 4),
                    "control_drop": round(rec["control_drop"], 4),
                    "cmi_score": round(rec["cmi_score"], 4),
                    "bypass_score": round(rec["bypass_score"], 4),
                }
            )
        span_tables[r["id"]] = layer_rows

        per_layer_rows = []
        for rec in r.get("layerwise_records", []):
            if len(rec["layers"]) != 1:
                continue
            per_layer_rows.append(
                {
                    "layer": rec["layers"][0],
                    "cot_drop": round(rec["cot_drop"], 4),
                    "control_drop": round(rec["control_drop"], 4),
                    "cmi_raw": round(rec.get("cmi_raw", 0.0), 4),
                    "cmi_score": round(rec["cmi_score"], 4),
                    "bypass_score": round(rec["bypass_score"], 4),
                }
            )
        per_layer_tables[r["id"]] = per_layer_rows

    Path("data/cmi_layers_all.json").write_text(json.dumps(span_tables, indent=2))
    print("Saved layer-span table data for all examples to data/cmi_layers_all.json")

    Path("data/cmi_layers_per_layer_all.json").write_text(json.dumps(per_layer_tables, indent=2))
    print("Saved per-layer table data for all examples to data/cmi_layers_per_layer_all.json")

if __name__ == "__main__":
    main()
