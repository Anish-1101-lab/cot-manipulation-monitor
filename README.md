# Chain-of-Thought Manipulation Monitor

A small monitoring tool that analyzes an LLM’s chain‑of‑thought (CoT) in real time to detect **manipulative reasoning patterns** (evaluation awareness, deception planning, safety evasion) and outputs a **0–100 risk score** with a Streamlit dashboard.

---

## Features

- **Groq‑backed CoT extraction**
  - Uses Groq’s OpenAI‑compatible API (e.g. `llama-3.1-8b-instant`) to return:
    - `cot`: internal reasoning text.
    - `answer`: final user‑visible reply.
- **Manipulation pattern library**
  - Regex‑based patterns for:
    - Evaluation awareness (talking about tests, graders, auditors).
    - Strategic deception (hiding flaws, planning to mislead).
    - Safety/policy evasion and stock CoT boilerplate.
- **Risk scoring**
  - Text‑level features:
    - `pattern_risk` (0–1) from manipulation templates.
    - `long_cot` factor (length heuristic).
    - Combined into a **0–100 manipulation risk**.
- **Two modes**
  - `normal`: model just solves the task with CoT.
  - `audit_aware`: model is told it is under audit, to expose strategies and evaluation awareness.
- **Streamlit dashboard**
  - Input user prompt + mode + Groq model.
  - Shows risk band (green/yellow/red), per‑pattern bar chart, CoT, and answer.

---

## Project Structure

```bash
cot-manipulation-monitor/                 
├── README.md
├── requirements.txt
├── app.py                   # Streamlit app
├── test_extractor.py        # extractor sanity check
├── test_patterns.py         # pattern scoring check
├── test_provenance.py       # risk scoring check
├── test_pipeline.py         # end-to-end check
├── cot_monitor/
│   ├── __init__.py
│   ├── cot_extractor.py     # Groq CoT + answer
│   ├── patterns.py          # manipulation templates + scores
│   └── provenance.py        # risk aggregation (0–100)
└── tests/
    └── test_convos.json     # to be done


---

## CoT Bypass Monitor Metrics

This repo includes a causal **CoT‑bypass monitor** in `cot_bypass_monitor.py`. It measures whether
answer tokens depend on the CoT token positions, or whether the model bypasses them.

Below are the metric definitions used by the script, including the math.

### Notation

- Prompt with CoT: \( x_c \)
- Prompt without CoT: \( x_{\neg c} \)
- Answer tokens: \( y = (y_1, \ldots, y_T) \)
- CoT token positions: \( \mathcal{C} \)
- Non‑CoT control positions: \( \mathcal{N} \)
- Log‑probability under prompt \(x\):  
  \[
  \log P(y \mid x) = \sum_{t=1}^{T} \log p(y_t \mid x, y_{<t})
  \]

### Baseline LogP

\[
\text{baseline\_logp} = \log P(y \mid x_c)
\]

### CoT‑Mediated Influence (CMI)

For each layer span, we perform **source patching**:

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

### Bypass Score

\[
\text{Bypass} = 1 - \text{CMI}
\]

High bypass indicates the answer does **not** depend on CoT positions.

### Placebo CoT (Noise Test)

Replace CoT hidden states with **noise** instead of no‑CoT states and compute CMI.

The resulting value is **Placebo\_CMI**. If it is high, the model is insensitive to the
actual CoT content.

### Validity Index

\[
\text{Validity} = \frac{\max(0, \text{CMI} - \text{Placebo\_CMI})}{\max(\text{CMI}, \epsilon)}
\]

High validity means **real CoT content matters more than placebo**.

### Density Curve (Sparsity Test)

We compute CMI after randomly keeping 80%, 60%, and 40% of CoT positions:

\[
\text{Density\_Curve} = \{(0.8, \text{CMI}_{0.8}), (0.6, \text{CMI}_{0.6}), (0.4, \text{CMI}_{0.4})\}
\]

### Density Index

\[
\text{Density} = 1 - \min\left(1, \frac{\text{CMI}_{0.4}}{\max(\text{CMI}, \epsilon)}\right)
\]

High density means **removing CoT tokens hurts**.

### Sequentiality Index (Shuffle Test)

We shuffle which CoT hidden states are mapped to which CoT positions, recompute CMI, and define:

\[
\text{Sequentiality} = \frac{\max(0, \text{CMI} - \text{Shuffled\_CMI})}{\max(\text{CMI}, \epsilon)}
\]

High sequentiality means **order matters**.

### No‑CoT‑Effect Flag

If all per‑span CMI scores are zero, the run is flagged:

```
no_cot_effect = True
```

This indicates the answer is effectively independent of the CoT token positions.

### Sensitivity Analysis (Boundary Robustness)

We expand and shrink the CoT window by one token on each side and measure:

- Base CMI
- Expanded CMI
- Shrunk CMI

Then compute:

\[
\text{relative\_sensitivity} = \frac{\max(|\text{CMI}_{\text{expanded}} - \text{CMI}_{\text{base}}|,
|\text{CMI}_{\text{shrunk}} - \text{CMI}_{\text{base}}|)}{\max(\text{CMI}_{\text{base}}, \epsilon)}
\]

A run is considered robust if relative sensitivity is low **or** absolute deviation is below a
fixed threshold.
