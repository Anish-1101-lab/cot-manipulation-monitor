import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/monitor_results_all_metrics.jsonl")


def load_df(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = {
                "prompt": obj["prompt"],
                "risk_normal": obj["normal"]["metrics"]["risk"],
                "risk_audit": obj["audit_aware"]["metrics"]["risk"],
            }
            # add all deltas
            for k, v in obj["deltas"].items():
                rec[k] = v
            records.append(rec)
    return pd.DataFrame(records)


def describe_metric(df: pd.DataFrame, col: str) -> dict:
    vals = df[col].dropna().to_numpy()
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def main():
    df = load_df(INPUT_PATH)
    print(f"Loaded {len(df)} examples\n")

    metrics = [
        "risk_delta",
        "long_cot_delta",
        "off_topic_score_delta",
        "misalign_score_delta",
        "zlib_entropy_risk_delta",
    ]

    print("=== Summary statistics for key deltas (audit_aware − normal) ===")
    rows = []
    for m in metrics:
        stats_m = describe_metric(df, m)
        rows.append(
            {
                "metric": m,
                "mean": stats_m["mean"],
                "std": stats_m["std"],
                "min": stats_m["min"],
                "max": stats_m["max"],
            }
        )
        print(
            f"{m}: mean={stats_m['mean']:.3f}, sd={stats_m['std']:.3f}, "
            f"min={stats_m['min']:.3f}, max={stats_m['max']:.3f}"
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("data/metric_deltas_summary.csv", index=False)

    # proportion of prompts with large risk increase (> 10 points)
    big_increase = (df["risk_delta"] > 10).mean()
    print(f"\nProportion with risk_delta > 10: {big_increase*100:.1f}%")

    # mean risk normal vs audit (for the paper sentence)
    mean_risk_normal = df["risk_normal"].mean()
    mean_risk_audit = df["risk_audit"].mean()
    print(
        f"\nMean risk (normal): {mean_risk_normal:.2f}, "
        f"Mean risk (audit_aware): {mean_risk_audit:.2f}, "
        f"Delta: {mean_risk_audit - mean_risk_normal:.2f}"
    )

    # Simple text interpretations you can paste in the paper
    print("\n=== Text interpretations ===")

    print(
        f"- Monitoring-aware prompting increases mean risk scores from "
        f"{mean_risk_normal:.1f} to {mean_risk_audit:.1f} "
        f"(Δ={mean_risk_audit - mean_risk_normal:.1f})."
    )
    print(
        f"- CoT length factor shows a mean increase of "
        f"{summary_df.loc[summary_df.metric == 'long_cot_delta','mean'].iloc[0]:.3f}, "
        "indicating systematically longer CoTs under audit-aware prompting."
    )
    print(
        f"- Off-topic score increases by "
        f"{summary_df.loc[summary_df.metric == 'off_topic_score_delta','mean'].iloc[0]:.3f} "
        "on average, suggesting more meta- or evaluator-focused reasoning."
    )
    print(
        f"- Misalignment score increases by "
        f"{summary_df.loc[summary_df.metric == 'misalign_score_delta','mean'].iloc[0]:.3f}, "
        "indicating slightly weaker coupling between CoT and final answer in audit-aware mode."
    )
    print(
        f"- Zlib entropy risk changes by "
        f"{summary_df.loc[summary_df.metric == 'zlib_entropy_risk_delta','mean'].iloc[0]:.3f} "
        "on average, reflecting modest shifts in how templated or compressible CoTs appear."
    )
    print(
        f"- {big_increase*100:.1f}% of prompts exhibit risk increases exceeding 10 points, "
        "highlighting a non-trivial subset where audit-aware prompting surfaces substantially "
        "more concerning reasoning traces."
    )


if __name__ == "__main__":
    main()