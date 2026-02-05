#!/usr/bin/env python3
import json
from pathlib import Path
import argparse


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_per_layer_files(modelsrun_dir: Path):
    prefix = "cmi_layers_per_layer_all_"
    for path in modelsrun_dir.iterdir():
        if path.is_file() and path.name.startswith(prefix):
            yield path


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
        description="Average number of active (CMI > threshold) layers per datapoint, per model."
    )
    parser.add_argument(
        "--modelsrun",
        default="modelsrun",
        help="Directory containing cmi_layers_per_layer_all_* files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="CMI threshold (strictly greater than).",
    )
    args = parser.parse_args()

    modelsrun_dir = Path(args.modelsrun)
    if not modelsrun_dir.exists():
        raise FileNotFoundError(f"modelsrun directory not found: {modelsrun_dir}")

    files = sorted(iter_per_layer_files(modelsrun_dir))
    if not files:
        raise FileNotFoundError("No cmi_layers_per_layer_all_* files found.")

    for path in files:
        data = load_json(path)
        model_name = path.name.replace("cmi_layers_per_layer_all_", "")
        if model_name.endswith(".json"):
            model_name = model_name[:-5]

        per_datapoint_counts = []
        per_datapoint_totals = []
        for _, rows in data.items():
            count = 0
            for rec in rows:
                if float(rec.get("cmi_score", 0.0)) > args.threshold:
                    count += 1
            per_datapoint_counts.append(count)
            per_datapoint_totals.append(max_layer_index(rows) + 1)

        avg_active = mean(per_datapoint_counts)
        avg_total = mean(per_datapoint_totals)
        pct = (avg_active / avg_total * 100.0) if avg_total > 0 else 0.0
        print(
            f"{model_name}: avg_active_layers={avg_active:.3f} "
            f"avg_total_layers={avg_total:.3f} active_pct={pct:.2f}% "
            f"(n={len(per_datapoint_counts)})"
        )


if __name__ == "__main__":
    main()
