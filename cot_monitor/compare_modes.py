# cot_monitor/compare_modes.py

from typing import Dict, Any

from cot_monitor.cot_extractor import extract_cot
from cot_monitor.provenance import compute_risk_score


def compare_modes(prompt: str, model: str) -> Dict[str, Any]:
    """
    Run normal and audit_aware on the same prompt and compute per-metric deltas.

    Returns a dict of the form:

    {
      "normal": {
        "cot": str,
        "answer": str,
        "metrics": {
          "risk": int,
          "pattern_risk": float,
          "long_cot": float,
          "cos_sim_prompt_cot": float,
          "cos_sim_answer_cot": float,
          "off_topic_score": float,
          "misalign_score": float,
          "reasoning_quality": float,
          "unstructured_risk": float,
          "compression_ratio": float,
          "zlib_entropy_risk": float,
          "pat_<name>": float,
          ...
        },
      },
      "audit_aware": {
        "cot": str,
        "answer": str,
        "metrics": { ... same keys ... },
      },
      "deltas": {
        "<metric>_delta": audit_value - normal_value,
        ...
        "risk_delta": float,
        "cot_length_delta": float,
        "pattern_risk_delta": float,
      },
    }
    """
    # ----- Run normal mode -----
    res_normal = extract_cot(prompt, mode="normal", model=model)
    cot_normal = res_normal.get("cot", "")
    ans_normal = res_normal.get("answer", "")
    metrics_normal = compute_risk_score(prompt, cot_normal, ans_normal)

    # ----- Run audit-aware mode -----
    res_audit = extract_cot(prompt, mode="audit_aware", model=model)
    cot_audit = res_audit.get("cot", "")
    ans_audit = res_audit.get("answer", "")
    metrics_audit = compute_risk_score(prompt, cot_audit, ans_audit)

    # ----- Compute per-metric deltas for all numeric metrics -----
    deltas: Dict[str, float] = {}
    common_keys = set(metrics_normal.keys()) & set(metrics_audit.keys())

    for key in common_keys:
        v_n = metrics_normal[key]
        v_a = metrics_audit[key]
        # only compute deltas for numeric metrics
        if isinstance(v_n, (int, float)) and isinstance(v_a, (int, float)):
            deltas[f"{key}_delta"] = float(v_a) - float(v_n)

    # Convenience aliases for the three main deltas used in the write-up
    # (these will just mirror the generic *_delta entries).
    if "risk" in metrics_normal and "risk" in metrics_audit:
        deltas["risk_delta"] = deltas.get(
            "risk_delta",
            float(metrics_audit["risk"]) - float(metrics_normal["risk"]),
        )

    if "long_cot" in metrics_normal and "long_cot" in metrics_audit:
        deltas["cot_length_delta"] = deltas.get(
            "cot_length_delta",
            float(metrics_audit["long_cot"]) - float(metrics_normal["long_cot"]),
        )

    if "pattern_risk" in metrics_normal and "pattern_risk" in metrics_audit:
        deltas["pattern_risk_delta"] = deltas.get(
            "pattern_risk_delta",
            float(metrics_audit["pattern_risk"]) - float(metrics_normal["pattern_risk"]),
        )

    return {
        "normal": {
            "cot": cot_normal,
            "answer": ans_normal,
            "metrics": metrics_normal,
        },
        "audit_aware": {
            "cot": cot_audit,
            "answer": ans_audit,
            "metrics": metrics_audit,
        },
        "deltas": deltas,
    }
