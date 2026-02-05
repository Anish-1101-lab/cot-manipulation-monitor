#!/usr/bin/env python3
import json
from pathlib import Path
import argparse


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    return sum(values) / max(1, len(values))


def max_layer_index(rows):
    max_idx = -1
    for rec in rows:
        layer = rec.get("layer")
        if isinstance(layer, int) and layer > max_idx:
            max_idx = layer
    return max_idx


def main():
    parser = argparse.ArgumentParser(
        description="Compute summary stats for DialoGPT StrategyQA causal module."
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base directory containing cmi_summary_table.json and cmi_layers_per_layer_all.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="CMI threshold for active layers (strictly greater than).",
    )
    parser.add_argument(
        "--model",
        default="DialoGPT",
        help="Model name to print in the LaTeX row.",
    )
    args = parser.parse_args()

    base = Path(args.base) if args.base is not None else Path(__file__).resolve().parent
    summary_path = base / "cmi_summary_table.json"
    per_layer_path = base / "cmi_layers_per_layer_all.json"

    summary = load_json(summary_path)
    per_layer = load_json(per_layer_path)

    mean_cmi = mean([float(r.get("CMI_mean", 0.0)) for r in summary])

    per_datapoint_counts = []
    per_datapoint_totals = []
    for _, rows in per_layer.items():
        active = 0
        for rec in rows:
            if float(rec.get("cmi_score", 0.0)) > args.threshold:
                active += 1
        per_datapoint_counts.append(active)
        per_datapoint_totals.append(max_layer_index(rows) + 1)

    avg_active_layers = mean(per_datapoint_counts)
    avg_total_layers = mean(per_datapoint_totals)
    active_pct = (avg_active_layers / avg_total_layers * 100.0) if avg_total_layers > 0 else 0.0

    # Print LaTeX row
    print(
        f"{args.model} & {mean_cmi:.4f} & {avg_total_layers:.1f} & "
        f"{avg_active_layers:.2f} & {active_pct:.2f}\\% \\\\ "
    )


if __name__ == "__main__":
    main()
