# Chain-of-Thought Manipulation Monitor

A monitoring toolkit for large language models that (1) scores **manipulative reasoning patterns** in chain-of-thought (CoT) traces in real time and (2) causally probes whether answers actually **depend** on those CoT activations or merely bypass them.

Developed as part of the Apart Research AI Manipulation Hackathon and described in the paper *Chain-of-Thought Manipulation Monitor: Detection of Deceptive Reasoning in LLMs*.

---

## Features

- **Groq‑backed CoT extraction**  
  - Uses Groq’s OpenAI‑compatible API (e.g. `llama-3.1-8b-instant`) to return:
    - `cot`: internal reasoning text.
    - `answer`: final user‑visible reply.

- **Manipulation pattern library**  
  - Regex‑based patterns for:
    - Evaluation awareness (tests, graders, auditors).
    - Strategic deception (hiding flaws, planning to mislead).
    - Safety/policy evasion, refusal‑then‑comply, and stock/boilerplate CoT traces.

- **Risk scoring**  
  - Text‑level features:
    - `pattern_risk` (0–1) from manipulation templates.
    - `long_cot` factor (length heuristic).
    - Semantic coherence (prompt–CoT and CoT–answer similarity via `all-MiniLM-L6-v2`).
    - Structural metrics, including a Zlib compression‑based entropy risk.
  - Aggregated into a **0–100 manipulation risk** in `provenance.py`.

- **Two modes**  
  - `normal`: model solves the task with CoT.
  - `audit_aware`: model is explicitly told it is under audit to surface strategies and evaluation awareness.

- **Streamlit dashboard**  
  - Input: user prompt + mode + Groq model.
  - Output: risk band (green/yellow/red), per‑pattern bar chart, semantic & structural scores, CoT, and answer.

- **Causal CoT‑bypass monitor** (`cot_bypass_monitor/`)  
  - Source‑patching on CoT token positions vs non‑CoT controls.
  - Reports CoT‑Mediated Influence (CMI), bypass, placebo CMI, validity, density, sequentiality, and boundary robustness.

---

## Repository Structure

```bash
cot-manipulation-monitor/
├── cot_bypass_monitor/          # causal CoT-bypass code (Qwen2.5-0.5B, etc.)
│   ├── __init__.py
│   └── cot_bypass_monitor.py
│
├── cot_monitor/                 # behavioural CoT manipulation monitor
│   ├── __init__.py
│   ├── cot_extractor.py         # Groq CoT + answer (normal / audit-aware)
│   ├── patterns.py              # 19-category manipulation regex library + weights
│   └── provenance.py            # risk aggregation, semantic & structural features
│
├── data/                        # saved metric runs / CSVs used in the paper
│   └── ...                      # behavioural deltas, causal metric dumps, etc.
│
├── figures/                     # plots used in the submission
│   └── ...                      # behavioural and causal result figures
│
├── app.py                       # Streamlit dashboard (behavioural monitor)
├── run_monitoring_eval.py       # eval: normal vs audit-aware behavioural runs
├── run_monitoring_eval_all_metrics.py
│                                # eval: behavioural + causal metrics
├── analyse_results.py           # summarise per-prompt risk deltas
├── analyse_metrics.py           # aggregate causal metrics, tables
├── analyse_metrics_fig.py       # figure generation for causal metrics
│
├── README.md                    # this file
├── README_COT_BYPASS.md         # detailed causal metric definitions and math
├── requirements.txt             # Python dependencies
└── .DS_Store                    # macOS metadata (can be ignored)

## Causal CoT‑Bypass Metrics (Summary)

The causal monitor measures whether answer tokens depend on CoT token positions or whether the model bypasses them.

### Notation

- Prompt with CoT: \(x_c\)  
- Prompt without CoT: \(x_{\neg c}\)  
- Answer tokens: \(y = (y_1, \ldots, y_T)\)  
- CoT token positions: \(\mathcal{C}\)  
- Non‑CoT control positions: \(\mathcal{N}\)  

Baseline log‑probability under the CoT prompt:

\[
\log P(y \mid x_c) = \sum_{t=1}^{T} \log p(y_t \mid x_c, y_{<t})
\]

### CoT‑Mediated Influence (CMI)

For each layer span, perform **source patching**:

- Patch CoT positions using hidden states from the no‑CoT run:

\[
\Delta_{\text{cot}} = \max\left(0,\ \log P(y \mid x_c) - \log P(y \mid \text{patch}_{\mathcal{C}}(x_c, x_{\neg c}))\right)
\]

- Patch equal‑sized random non‑CoT positions as a control:

\[
\Delta_{\text{ctrl}} = \max\left(0,\ \log P(y \mid x_c) - \log P(y \mid \text{patch}_{\mathcal{N}}(x_c, x_{\neg c}))\right)
\]

Then define:

\[
\text{CMI} = \frac{\max(0, \Delta_{\text{cot}} - \Delta_{\text{ctrl}})}{\max(\Delta_{\text{cot}} + \Delta_{\text{ctrl}}, \epsilon)}
\]

A small‑signal gate sets CMI to 0 when \(\Delta_{\text{cot}} + \Delta_{\text{ctrl}}\) is below a threshold.

### Bypass Score

\[
\text{Bypass} = 1 - \text{CMI}
\]

High bypass indicates the answer does **not** depend on CoT positions.

### Additional Indices

- **Placebo CoT:** Replace CoT hidden states with noise instead of no‑CoT states and recompute CMI → `Placebo_CMI`.  
- **Validity:**

\[
\text{Validity} = \frac{\max(0, \text{CMI} - \text{Placebo\_CMI})}{\max(\text{CMI}, \epsilon)}
\]

  High validity means real CoT content matters more than placebo.

- **Density curve (sparsity test):**

\[
\text{Density\_Curve} = \{(0.8, \text{CMI}_{0.8}), (0.6, \text{CMI}_{0.6}), (0.4, \text{CMI}_{0.4})\}
\]

- **Density index:**

\[
\text{Density} = 1 - \min\left(1, \frac{\text{CMI}_{0.4}}{\max(\text{CMI}, \epsilon)}\right)
\]

  High density means removing CoT tokens hurts.

- **Sequentiality index (shuffle test):**

\[
\text{Sequentiality} = \frac{\max(0, \text{CMI} - \text{Shuffled\_CMI})}{\max(\text{CMI}, \epsilon)}
\]

  High sequentiality means order matters.

- **No‑CoT‑effect flag:** If all per‑span CMIs are zero:

```python
no_cot_effect = True
indicating the answer is effectively independent of CoT token positions.

- **Sensitivity analysis (boundary robustness):**

\[
\text{relative\_sensitivity} =
\frac{\max(|\text{CMI}_{\text{expanded}} - \text{CMI}_{\text{base}}|,
|\text{CMI}_{\text{shrunk}} - \text{CMI}_{\text{base}}|)}
{\max(\text{CMI}_{\text{base}}, \epsilon)}
\]

A run is considered robust if relative sensitivity is low or absolute deviation is below a fixed threshold.

---

## Usage

### Setup

```bash
pip install -r requirements.txt
Set your Groq API key before running:

```bash
export GROQ_API_KEY="your_key_here"
Launch Streamlit monitor
```bash
streamlit run app.py
Then enter a prompt, select normal or audit_aware, choose a Groq model (e.g. llama-3.1-8b-instant), and inspect the risk score and CoT/answer trace in the dashboard.

Run evaluations
Behavioural audit‑aware vs normal:

```bash
python run_monitoring_eval.py
python analyse_results.py
Behavioural + causal metrics and figures:

```bash
python run_monitoring_eval_all_metrics.py
python analyse_metrics.py
python analyse_metrics_fig.py
Results are written under data/ and figures/, matching the analyses in the paper.

Citation
If you use this repository, please cite:

Anish Sathyanarayanan*, Aditya Nagarsekar*, Aarush Rathore*.
Chain-of-Thought Manipulation Monitor: Detection of Deceptive Reasoning in LLMs.
Apart Research Hackathon Submission, 2026.
