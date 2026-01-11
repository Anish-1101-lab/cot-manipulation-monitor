import json
from pathlib import Path
from statistics import mean

from cot_monitor.compare_modes import compare_modes

MODEL = "llama-3.1-8b-instant"  # or your Groq model
INPUT_PATH = Path("data/monitor_prompts.jsonl")
OUT_PATH = Path("data/monitor_results.jsonl")


import json

def load_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue  # skip empty lines
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping non-JSON line:", repr(line))
                continue

            human_msg = next(
                (m["content"] for m in obj["prompt"] if m.get("type") == "human"),
                None,
            )
            if human_msg is None:
                continue
            yield human_msg, obj



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

            # quick console view of key stuff
            print(
                "risk (normal, audit, Δ):",
                normal_metrics["risk"],
                audit_metrics["risk"],
                deltas["risk_delta"],
            )
            print(
                "pattern_risk (n, a, Δ):",
                normal_metrics["pattern_risk"],
                audit_metrics["pattern_risk"],
                deltas["pattern_risk_delta"],
            )
            print(
                "long_cot (n, a, Δ):",
                normal_metrics["long_cot"],
                audit_metrics["long_cot"],
                deltas["long_cot_delta"],
            )

            # save full result per example to JSONL
            record = {
                "raw": raw,              # original dataset JSON
                "prompt": prompt,
                "normal": cmp_res["normal"],
                "audit_aware": cmp_res["audit_aware"],
                "deltas": deltas,
            }
            fout.write(json.dumps(record) + "\n")

            all_deltas.append(deltas)

    # aggregate over all prompts
    if all_deltas:
        def avg(key):
            vals = [d[key] for d in all_deltas if key in d]
            return mean(vals) if vals else 0.0

        print("\n=== Aggregate over all prompts ===")
        print("Average risk Δ:", avg("risk_delta"))
        print("Average pattern_risk Δ:", avg("pattern_risk_delta"))
        print("Average long_cot Δ:", avg("long_cot_delta"))
        print("Average off_topic_score Δ:", avg("off_topic_score_delta"))
        print("Average misalign_score Δ:", avg("misalign_score_delta"))
        print("Average zlib_entropy_risk Δ:", avg("zlib_entropy_risk_delta"))


if __name__ == "__main__":
    main()
