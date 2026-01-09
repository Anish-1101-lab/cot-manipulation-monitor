# cot_monitor/provenance.py

from typing import Dict
from .patterns import detect_patterns, aggregate_pattern_risk

def compute_text_features(cot: str) -> Dict[str, float]:
    """
    Extract simple text-level features from CoT:
    - pattern_risk: weighted manipulation template overlap (0–1)
    - long_cot: length factor (0–1)
    - per-pattern scores: pat_<name>
    """
    pattern_scores = detect_patterns(cot)
    pattern_risk = aggregate_pattern_risk(pattern_scores)

    num_tokens = len(cot.split())
    # Simple length feature: saturates after ~300 tokens
    long_cot = min(num_tokens / 300.0, 1.0)

    feats: Dict[str, float] = {
        "pattern_risk": pattern_risk,
        "long_cot": long_cot,
    }
    for name, score in pattern_scores.items():
        feats[f"pat_{name}"] = score

    return feats

def compute_risk_score(cot: str) -> Dict[str, float]:
    """
    Aggregate features into a 0–100 manipulation risk score.
    Returns:
      {
        "risk": int 0–100,
        "pattern_risk": ...,
        "long_cot": ...,
        "pat_<pattern>": ...,
      }
    """
    feats = compute_text_features(cot)

    pattern_risk = feats["pattern_risk"]
    long_cot = feats["long_cot"]

    # Linear combo like CRaft RI = w1*TOS + w2*IPR + ...
    # Here: focus on manipulation patterns; length is weak prior.
    risk_0_1 = 0.9 * pattern_risk + 0.1 * long_cot
    risk_0_100 = int(risk_0_1 * 100)

    return {
        "risk": risk_0_100,
        **feats,
    }
