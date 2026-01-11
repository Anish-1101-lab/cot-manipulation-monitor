import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_PATH = Path("data/monitor_results.jsonl")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = {
                "prompt": obj["prompt"],
                # core metrics
                "risk_normal": obj["normal"]["metrics"]["risk"],
                "risk_audit": obj["audit_aware"]["metrics"]["risk"],
            }
            # add all deltas
            for k, v in obj["deltas"].items():
                rec[k] = v
            records.append(rec)
    return pd.DataFrame(records)


def main():
    df = load_records(INPUT_PATH)
    print("Loaded", len(df), "examples")

    # ---- 1. Overall mean deltas table ----
    key_deltas = [
        "risk_delta",
        "long_cot_delta",
        "off_topic_score_delta",
        "misalign_score_delta",
        "zlib_entropy_risk_delta",
    ]
    summary = (
        df[key_deltas]
        .agg(["mean", "std"])
        .T
        .rename(columns={"mean": "Mean", "std": "Std"})
    )
    print("\nSummary of key deltas:")
    print(summary)

    summary.to_csv(OUT_DIR / "summary_deltas.csv", index_label="metric")

    # ---- 2. Bar chart: mean delta per metric ----
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x=summary.index,
        y="Mean",
        data=summary.reset_index(),
        yerr=summary["Std"].values,
        capsize=0.2,
        color="#4C72B0",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Mean Δ (audit_aware − normal)")
    ax.set_xlabel("Metric")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_deltas_bar.png", dpi=300)
    plt.close()

    # ---- 3. Per-example risk (normal vs audit_aware) ----
    plt.figure(figsize=(6, 4))
    ax = sns.scatterplot(
        x="risk_normal",
        y="risk_audit",
        data=df,
    )
    ax.plot([0, 100], [0, 100], "--", color="gray")  # y = x line
    ax.set_xlabel("Risk (normal)")
    ax.set_ylabel("Risk (audit_aware)")
    ax.set_title("Per-example risk: normal vs audit-aware")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "risk_scatter.png", dpi=300)
    plt.close()

    # ---- 4. Optional: histogram of risk_deltas ----
    plt.figure(figsize=(6, 4))
    ax = sns.histplot(df["risk_delta"], bins=10, kde=False)
    ax.set_xlabel("Risk Δ (audit_aware − normal)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "risk_delta_hist.png", dpi=300)
    plt.close()

    # ---- 5. Example-level bar for a single prompt ----
    # Pick the example with largest absolute risk_delta
    idx = df["risk_delta"].abs().idxmax()
    row = df.loc[idx]
    print("\nExample with largest |risk_delta|:")
    print("Prompt:", row["prompt"])
    print("risk_normal:", row["risk_normal"], "risk_audit:", row["risk_audit"])

    ex_metrics = {
        "risk_delta": row["risk_delta"],
        "long_cot_delta": row["long_cot_delta"],
        "off_topic_score_delta": row["off_topic_score_delta"],
        "misalign_score_delta": row["misalign_score_delta"],
        "zlib_entropy_risk_delta": row["zlib_entropy_risk_delta"],
    }
    ex_df = pd.DataFrame(
        {"metric": list(ex_metrics.keys()), "delta": list(ex_metrics.values())}
    )

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="metric", y="delta", data=ex_df, color="#55A868")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Δ (audit_aware − normal)")
    ax.set_xlabel("Metric")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "single_example_deltas.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
