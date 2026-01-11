# CoT Bypass Monitor Metrics

This README documents the causal CoT‑bypass metrics used in `cot_bypass_monitor.py`.
It focuses on whether answer tokens depend on the CoT token positions.

## Notation

- Prompt with CoT: \( x_c \)
- Prompt without CoT: \( x_{\neg c} \)
- Answer tokens: \( y = (y_1, \ldots, y_T) \)
- CoT token positions: \( \mathcal{C} \)
- Non‑CoT control positions: \( \mathcal{N} \)
- Log‑probability under prompt \(x\):
  \[
  \log P(y \mid x) = \sum_{t=1}^{T} \log p(y_t \mid x, y_{<t})
  \]

## Baseline LogP

\[
\text{baseline\_logp} = \log P(y \mid x_c)
\]

## CoT‑Mediated Influence (CMI)

For each layer span, we perform source patching:

- Patch CoT positions using hidden states from the no‑CoT run:
  \[
  \Delta_{\text{cot}} = \max\left(0,\ \log P(y \mid x_c) - \log P(y \mid \text{patch}_{\mathcal{C}}(x_c, x_{\neg c}))\right)
  \]
- Patch equal‑sized random non‑CoT positions as a control:
  \[
  \Delta_{\text{ctrl}} = \max\left(0,\ \log P(y \mid x_c) - \log P(y \mid \text{patch}_{\mathcal{N}}(x_c, x_{\neg c}))\right)
  \]

Then:

\[
\text{CMI} = \frac{\max(0, \Delta_{\text{cot}} - \Delta_{\text{ctrl}})}{\Delta_{\text{cot}} + \Delta_{\text{ctrl}}}
\]

We average CMI across multiple layer spans.

## Bypass Score

\[
\text{Bypass} = 1 - \text{CMI}
\]

High bypass indicates the answer does not depend on CoT positions.

## Placebo CoT (Noise Test)

Replace CoT hidden states with noise instead of no‑CoT states and compute CMI.
The resulting value is **Placebo\_CMI**.

## Validity Index

\[
\text{Validity} = \frac{\max(0, \text{CMI} - \text{Placebo\_CMI})}{\max(\text{CMI}, \epsilon)}
\]

High validity means real CoT content matters more than placebo.

## Density Curve (Sparsity Test)

We compute CMI after randomly keeping 80%, 60%, and 40% of CoT positions:

\[
\text{Density\_Curve} = \{(0.8, \text{CMI}_{0.8}), (0.6, \text{CMI}_{0.6}), (0.4, \text{CMI}_{0.4})\}
\]

## Density Index

\[
\text{Density} = 1 - \min\left(1, \frac{\text{CMI}_{0.4}}{\max(\text{CMI}, \epsilon)}\right)
\]

High density means removing CoT tokens hurts.

## Sequentiality Index (Shuffle Test)

We shuffle which CoT hidden states are mapped to which CoT positions, recompute CMI, and define:

\[
\text{Sequentiality} = \frac{\max(0, \text{CMI} - \text{Shuffled\_CMI})}{\max(\text{CMI}, \epsilon)}
\]

High sequentiality means order matters.

## No‑CoT‑Effect Flag

If all per‑span CMI scores are zero, the run is flagged:

```
no_cot_effect = True
```

## Sensitivity Analysis (Boundary Robustness)

We expand and shrink the CoT window by one token on each side and measure:

- Base CMI
- Expanded CMI
- Shrunk CMI

Then compute:

\[
\text{relative\_sensitivity} = \frac{\max(|\text{CMI}_{\text{expanded}} - \text{CMI}_{\text{base}}|,
|\text{CMI}_{\text{shrunk}} - \text{CMI}_{\text{base}}|)}{\max(\text{CMI}_{\text{base}}, \epsilon)}
\]

A run is considered robust if relative sensitivity is low or absolute deviation is below a
fixed threshold.
