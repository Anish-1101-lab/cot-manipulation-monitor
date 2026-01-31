# COTMAN
# Chain-of-Thought Manipulation ANalysis of Deceptive & Bypassed Reasoning

A monitoring toolkit for large language models that (i) scores **manipulative reasoning patterns** in chain-of-thought (CoT) traces in real time and (ii) **causally probes** whether model answers actually depend on those CoT activations or merely bypass them.

The experiments and figures can be reproduced using the behavioural monitor under `cot_monitor/` and the causal CoT‑bypass monitor under `cot_bypass_monitor/`.

---

## Repository Structure

```text
repo/
├── cot_bypass_monitor/                # causal CoT-bypass (CMI / bypass) code
│   └── good/
│       ├── data/                      # datasets + causal outputs (local runs)
│       │   ├── cmi_suite.json               # per-instance causal records
│       │   ├── cmi_summary_table.json       # table-ready CMI summary
│       │   ├── cmi_layers_example.json      # example layerwise record
│       │   ├── cmi_layers_all.json          # span-level records (all IDs)
│       │   ├── cmi_layers_per_layer_all.json# per-layer records (all IDs)
│       │   ├── cmi_test_single.json         # single-prompt example
│       │   ├── strategy_qa.json             # StrategyQA dataset (JSON)
│       │   └── TruthfulQA.csv               # TruthfulQA dataset (CSV)
│       ├── data/qwen/                 # model-specific outputs (optional)
│       ├── cot_bypass_monitor.py            # core CMI / bypass implementation
│       ├── cot_bypass_results.json(l)       # optional raw runs
│       ├── test_cmi.py                      # causal suite (arith / logic / QA)
│       ├── test_cmi_layerwise_strategyqa.py # layerwise CMI on StrategyQA
│       └── test_truthqa.py                  # TruthfulQA evaluation
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
├── figures_and_data/               # generated figures + export bundles
│   ├── behavioural_module/
│   └── causal_module/
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
- select a Groq model,
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

After running one of the eval scripts, regenerate the main behavioural figures:

```bash
source venv/bin/activate
python analyse_metrics_figures.py
```

This reads from `cot_monitor/data/metric_deltas_summary.csv` and related files and produces figures under `figures_and_data/behavioural_module/`.

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

This produces per-run JSONs under `cot_bypass_monitor/good/data/` (plus any custom output directories your scripts target). Some runs may optionally write to `cot_bypass_monitor/good/data/qwen/`.

The rows in `cmi_summary_table.json` directly correspond to the instance-level CMI table in the paper.

### Causal plots and layer-wise heatmap

The `plots.py` script generates causal figures, including the layer-wise CMI heatmap.

```bash
cd cot_bypass_monitor/good
source ../../venv/bin/activate

python plots.py
```

This uses the data structure in `plots.py` and writes causal plots into the repo root by default (e.g., `new_plot_layer_heatmap.png`). You can relocate outputs into `figures_and_data/causal_module/` as needed.

## Reproducing Results

### Behavioural

```bash
python run_monitoring_eval.py            # or run_monitoring_eval_all_metrics.py
python analyse_metrics_figures.py
```

Figures are written under `figures_and_data/behavioural_module/`.

### Causal

```bash
cd cot_bypass_monitor/good
python test_cmi.py        # writes CMI tables to data/
python plots.py           # writes causal figures (e.g., new_plot_layer_heatmap.png)
```

Use `data/cmi_summary_table.json` for the instance-level CMI table and `new_plot_layer_heatmap.png` for the layerwise summary.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

