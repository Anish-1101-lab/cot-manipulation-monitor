# Chain-of-Thought Manipulation Monitor

A monitoring toolkit for large language models that (1) scores **manipulative reasoning patterns** in chain-of-thought (CoT) traces in real time and (2) causally probes whether answers actually **depend** on those CoT activations or merely bypass them.

Developed as part of the Apart Research AI Manipulation Hackathon and described in the paper *Chain-of-Thought Manipulation Monitor: Detection of Deceptive Reasoning in LLMs* ([PDF](./Apart_Submission_Final.pdf)).

---

## Features

- **Groq-backed CoT extraction**
  - Uses Groq's OpenAI-compatible API (e.g. `llama-3.1-8b-instant`) to return:
    - `cot`: internal reasoning text.
    - `answer`: final user-visible reply.

- **Manipulation pattern library**
  - Regex-based patterns for:
    - Evaluation awareness (tests, graders, auditors).
    - Strategic deception (hiding flaws, planning to mislead).
    - Safety/policy evasion, refusal-then-comply, and stock/boilerplate CoT traces.

- **Risk scoring**
  - Text-level features:
    - `pattern_risk` (0–1) from manipulation templates.
    - `long_cot` factor (length heuristic).
    - Semantic coherence (prompt–CoT and CoT–answer similarity via `all-MiniLM-L6-v2`).
    - Structural metrics, including a Zlib compression-based entropy risk.
  - Aggregated into a **0–100 manipulation risk** in `provenance.py`.

- **Two modes**
  - `normal`: model solves the task with CoT.
  - `audit_aware`: model is explicitly told it is under audit to surface strategies and evaluation awareness.

- **Streamlit dashboard**
  - Input: user prompt + mode + Groq model.
  - Output: risk band (green/yellow/red), per-pattern bar chart, semantic & structural scores, CoT, and answer.

- **Causal CoT-bypass monitor** (`48
- `)
  - Source-patching on CoT token positions vs non-CoT controls.
  - Reports CoT-Mediated Influence (CMI), bypass, placebo CMI, validity, density, sequentiality, and boundary robustness.

---

## Repository Structure

```bash
cot-manipulation-monitor/
├── cot_bypass_monitor/          # causal CoT-bypass code 
│  ├── good/  
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
├── run_monitoring_eval_all_metrics.py # eval: behavioural + causal metrics
│                                
├── analyse_metrics.py           # aggregate causal metrics, tables
├── analyse_metrics_fig.py       # figure generation for causal metrics
│
├── README.md                    # this file
├── README_COT_BYPASS.md         # detailed causal metric definitions and math
├── requirements.txt             # Python dependencies
└── .DS_Store                    # macOS metadata (can be ignored)
├── Apart_Submission_Final.pdf  # research paper submission
```


---

## Usage

### Setup

```bash
pip install -r requirements.txt
```

Set your Groq API key before running:

```bash
export GROQ_API_KEY="your_key_here"
```

### Launch Streamlit monitor

```bash
streamlit run app.py
```

Then enter a prompt, select `normal` or `audit_aware`, choose a Groq model (e.g. `llama-3.1-8b-instant`), and inspect the risk score and CoT/answer trace in the dashboard.

### Run evaluations

**Behavioural audit-aware vs normal:**

```bash
python run_monitoring_eval.py
```

**Behavioural + causal metrics and figures:**

```bash
python run_monitoring_eval_all_metrics.py
python analyse_metrics.py
python analyse_metrics_fig.py
```

Results are written under `data/` and `figures/`, matching the analyses in the paper.

---

## Citation

If you use this repository, please cite:

```
Anish Sathyanarayanan, Aditya Nagarsekar, Aarush Rathore.
Chain-of-Thought Manipulation Monitor: Detection of Deceptive Reasoning in LLMs.
Apart Research Hackathon Submission, 2026.
```

---
## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or feedback, please contact the authors or open an issue on GitHub.









