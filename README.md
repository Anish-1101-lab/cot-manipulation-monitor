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


