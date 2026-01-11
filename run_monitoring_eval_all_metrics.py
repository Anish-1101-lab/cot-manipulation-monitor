import json
from pathlib import Path
from statistics import mean

from cot_monitor.compare_modes import compare_modes

MODEL = "llama-3.1-8b-instant"  # or your Groq model
INPUT_PATH = Path("data/monitor_prompts.jsonl")
OUT_PATH = Path("data/monitor_results_all_metrics.jsonl")


def load_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping non-JSON line:", repr(line))
                continue

            human_msg = next(
                (m["content"] for m in obj.get("prompt", []) if m.get("type") == "human"),
                None,
            )
            if human_msg is None:
                continue
            yield human_msg, obj


def _print_metrics(label: str, metrics: dict):
    print(f"{label} metrics:")
    for key in sorted(metrics.keys()):
        val = metrics[key]
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


def main():
    all_deltas = []
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for i, (prompt, raw) in enumerate(load_prompts(INPUT_PATH), start=1):
            print(f"\n=== Example {i} ===")
            print("User prompt:", prompt)

            cmp_res = compare_modes(prompt, model=MODEL)
            normal_metrics = cmp_res["normal"]["metrics"]
            audit_metrics = cmp_res["audit_aware"]["metrics"]
            deltas = cmp_res["deltas"]

            _print_metrics("Normal", normal_metrics)
            _print_metrics("Audit-aware", audit_metrics)
            _print_metrics("Delta", deltas)

            record = {
                "raw": raw,
                "prompt": prompt,
                "normal": cmp_res["normal"],
                "audit_aware": cmp_res["audit_aware"],
                "deltas": deltas,
            }
            fout.write(json.dumps(record) + "\n")

            all_deltas.append(deltas)

    if all_deltas:
        print("\n=== Aggregate over all prompts (avg deltas) ===")
        keys = sorted({k for d in all_deltas for k in d.keys()})
        for key in keys:
            vals = [d[key] for d in all_deltas if key in d]
            avg_val = mean(vals) if vals else 0.0
            print(f"{key}: {avg_val:.4f}")


if __name__ == "__main__":
    main()
