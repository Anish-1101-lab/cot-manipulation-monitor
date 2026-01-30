# COTMAN: Chain-of-Thought Manipulation ANalysis of Deceptive & Bypassed Reasoning

A monitoring toolkit for large language models that (i) scores **manipulative reasoning patterns** in chain-of-thought (CoT) traces in real time and (ii) **causally probes** whether model answers actually depend on those CoT activations or merely bypass them.

The experiments and figures in the paper can be reproduced using the behavioural monitor under `cot_monitor/` and the causal CoT‑bypass monitor under `cot_bypass_monitor/`.

---

## Repository Structure

```text
cot-manipulation-monitor/
├── cot_bypass_monitor/                # causal CoT-bypass (CMI / bypass) code
│   └── good/
│       ├── data_cmi/                  # CMI / bypass numeric dumps used in the paper
│       │   ├── cmi_suite.json              # per-instance causal records
│       │   ├── cmi_summary_table.json      # table-ready CMI summary (Table 1)
│       │   ├── cmi_layers_*.json           # layer-span interventions per instance
│       │   └── cmi_test_single.json        # simple single-prompt example
│       ├── cot_bypass_monitor.py           # core CMI / bypass implementation
│       ├── cot_bypass_results.json(l)      # optional raw runs
│       └── test_cmi.py                     # small CMI suite (arith / logic / QA)
│
├── cot_monitor/                     # behavioural CoT manipulation monitor
│   ├── __init__.py
│   ├── compare_modes.py             # helper for normal vs audit-aware runs
│   ├── cot_extractor.py             # Groq-backed CoT + answer extraction
│   ├── patterns.py                  # 19-category manipulation regex library + weights
│   ├── provenance.py                # risk aggregation, semantic & structural features
│   └── data/
│       ├── metric_deltas_summary.csv        # behavioural deltas (normal vs audit-aware)
│       ├── monitor_prompts.jsonl            # prompts used in behavioural evals
│       ├── monitor_results.jsonl            # quick eval results (subset of metrics)
│       └── monitor_results_all_metrics.jsonl# full metric dumps
│
├── figures/                         # plots used in the paper
│   ├── mean_deltas_bar.png              # behavioural: mean risk deltas (Fig. 1a)
│   ├── risk_delta_hist.png              # behavioural: risk-delta distribution (Fig. 1b)
│   ├── risk_scatter.png                 # behavioural: per-metric scatter (optional)
│   ├── single_example_deltas.png        # behavioural: example delta breakdown
│   └── new_plot_layer_heatmap.png       # causal: layer-wise CMI heatmap (Appendix)
│
├── venv/                            # local virtual environment (not required by users)
│
├── app.py                           # Streamlit dashboard (behavioural monitor)
├── analyse_metrics_figures.py       # behavioural figure generation from eval outputs
├── plots.py                         # causal layer-wise heatmap + diagnostic plots
├── README_COT_BYPASS.md             # detailed causal metric definitions and math
├── README.md                        # this file
├── requirements.txt                 # Python dependencies
├── run_monitoring_eval.py           # quick behavioural eval (key metrics only)
└── run_monitoring_eval_all_metrics.py # full behavioural eval (all metrics)
```

## Installation

We recommend using a virtual environment.

```bash
git clone https://github.com/Anish-1101-lab/cot-manipulation-monitor.git
cd cot-manipulation-monitor

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

For the behavioural monitor and the Groq-backed CoT extractor, set your Groq API key:

```bash
export GROQ_API_KEY="your_key_here"   # Windows PowerShell: $env:GROQ_API_KEY="..."
```

## Behavioural CoT Manipulation Monitor

The behavioural monitor scores manipulation-relevant patterns directly from CoT text, using regex patterns, semantic similarity, and structural heuristics.

### Streamlit dashboard

To launch the interactive dashboard:

```bash
source venv/bin/activate
streamlit run app.py
```

The app lets you:

- enter a user prompt,
- choose normal or audit_aware mode,
- select a Groq model (e.g. llama-3.1-8b-instant),
- inspect the CoT + answer, pattern hits, and the 0–100 manipulation risk score.

### Quick behavioural evaluation (run_monitoring_eval.py)

`run_monitoring_eval.py` runs a normal vs audit-aware comparison on a prompt set and reports a focused set of metrics.

```bash
source venv/bin/activate
python run_monitoring_eval.py
```

This script:

- reads prompts from `cot_monitor/data/monitor_prompts.jsonl`,
- calls `compare_modes` for each prompt,
- prints, per prompt, normal vs audit vs delta for:
  - risk
  - pattern_risk
  - long_cot
- writes full per-example results to `cot_monitor/data/monitor_results.jsonl`,
- prints aggregate averages over all prompts for a small set of deltas (risk, pattern_risk, long_cot, off_topic_score, misalign_score, zlib_entropy_risk).

Use this when you want a lightweight run and console summary.

### Full behavioural evaluation (run_monitoring_eval_all_metrics.py)

`run_monitoring_eval_all_metrics.py` is the full behavioural eval: it keeps every metric key and averages all deltas.

```bash
source venv/bin/activate
python run_monitoring_eval_all_metrics.py
```

This script:

- reads the same `monitor_prompts.jsonl`,
- for each prompt, prints a full "Normal / Audit-aware / Delta metrics" block (all keys),
- writes the complete results to `cot_monitor/data/monitor_results_all_metrics.jsonl`,
- at the end, computes the mean for every delta key it sees across prompts and prints them.

Use this when you need the full metric dump for analysis or ablations.

### Behavioural figures

After running one of the eval scripts, regenerate the main behavioural figures used in the paper:

```bash
source venv/bin/activate
python analyse_metrics_figures.py
```

This reads from `cot_monitor/data/metric_deltas_summary.csv` and related files and produces:

- `figures/mean_deltas_bar.png`
- `figures/risk_delta_hist.png`
- `figures/risk_scatter.png`
- `figures/single_example_deltas.png`

These correspond to the plots described in the "Behavioural Monitor Results" section.

## Causal CoT-Bypass Monitor (CMI / Bypass)

The causal monitor estimates whether answers causally depend on CoT token activations via hidden-state patching, as defined in the paper (CoT-Mediated Influence, CMI, and bypass = 1−CMI).

All causal scripts and data live under `cot_bypass_monitor/good/`.

### Instance-level CMI suite

The `test_cmi.py` script runs a small suite of arithmetic, logic, and QA prompts with and without CoT, then computes span-averaged CMI and bypass per instance.

```bash
cd cot_bypass_monitor/good
source ../../venv/bin/activate        # adjust path if needed

python test_cmi.py
```

This produces:

- `data_cmi/cmi_suite.json` – full per-instance records, including per-span interventions,
- `data_cmi/cmi_summary_table.json` – compact summary (ID, task, correctness, CMI mean, bypass, baseline log P),
- `data_cmi/cmi_layers_<ID>.json` – layer-span interventions (cot_drop, control_drop, CMI, bypass) per instance.

The rows in `cmi_summary_table.json` directly correspond to the instance-level CMI table in the paper.

### Causal plots and layer-wise heatmap

The `plots.py` script generates causal figures for the paper, including the layer-wise CMI heatmap.

```bash
cd cot_bypass_monitor/good
source ../../venv/bin/activate

python plots.py
```

This uses the data structure in `plots.py` (mirroring `data_cmi/cmi_suite.json`) and writes causal plots into `figures/`, including:

- CMI / bypass bar plots and diagnostics (optional, for analysis),
- `figures/new_plot_layer_heatmap.png` – a layer-wise CMI heatmap ("Mechanistic Localization: Heatmap") summarizing, for each instance, the maximum span-level CMI at each layer index.

Include `new_plot_layer_heatmap.png` in the LaTeX paper as the appendix figure illustrating where in depth CoT-mediated influence is concentrated.

## Reproducing Paper Results

### Behavioural (Section 3.1):

```bash
python run_monitoring_eval.py            # or run_monitoring_eval_all_metrics.py
python analyse_metrics_figures.py
```

Figures are written under `figures/` and match the behavioural plots in the paper.

### Causal (Section 3.2 and Appendix):

```bash
cd cot_bypass_monitor/good
python test_cmi.py        # writes CMI tables to data_cmi/
python plots.py           # writes causal figures to figures/
```

Use `data_cmi/cmi_summary_table.json` for the instance-level CMI table and `data_cmi/cmi_layers_*.json` plus `figures/new_plot_layer_heatmap.png` for the layer-span summary.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

