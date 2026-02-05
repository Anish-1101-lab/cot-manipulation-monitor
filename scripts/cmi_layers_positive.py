#!/usr/bin/env python3
import json
from pathlib import Path
import argparse


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_layers(records_by_id, threshold, mode):
    # records_by_id: {sample_id: [ {layer, cmi_score, ...}, ...], ...}
    layer_to_scores = {}
    for _, rows in records_by_id.items():
        for rec in rows:
            layer = rec.get("layer")
            if layer is None:
                continue
            layer_to_scores.setdefault(layer, []).append(float(rec.get("cmi_score", 0.0)))

    positive_layers = []
    for layer, scores in sorted(layer_to_scores.items()):
        if mode == "any":
            stat = max(scores) if scores else 0.0
        else:
            stat = sum(scores) / max(1, len(scores))
        if stat > threshold:
            positive_layers.append(layer)

    return positive_layers, layer_to_scores


def iter_per_layer_files(modelsrun_dir: Path):
    prefix = "cmi_layers_per_layer_all_"
    for path in modelsrun_dir.iterdir():
        if path.is_file() and path.name.startswith(prefix):
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Report per-model layers with CMI > threshold from modelsrun."
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
    parser.add_argument(
        "--mode",
        choices=["mean", "any"],
        default="mean",
        help="Use mean CMI per layer or any positive sample.",
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

        positive_layers, layer_to_scores = collect_layers(
            data, threshold=args.threshold, mode=args.mode
        )
        print(f"{model_name}: {positive_layers}")


if __name__ == "__main__":
    main()
