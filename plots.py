import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

sns.set_theme(style="whitegrid", context="paper")

def plot_layer_heatmap(data, sample_indices=None, y_labels=None):
    heatmap_data = []
    label_list = []
    num_layers = 0
    for item in data:
        for rec in item:
            num_layers = max(num_layers, rec.get("layer", 0) + 1)

    if sample_indices is None:
        sample_indices = list(range(len(data)))

    for idx in sample_indices:
        layer_rows = data[idx]
        row = np.zeros(num_layers)
        for rec in layer_rows:
            layer = rec.get("layer", 0)
            cmi = rec.get("cmi_score", 0.0)
            if 0 <= layer < num_layers:
                row[layer] = cmi
        heatmap_data.append(row)
        if y_labels is not None and len(y_labels) > idx and y_labels[idx]:
            label = y_labels[idx]
        else:
            label = f"ex_{idx+1:03d}"
        label_list.append(label)

    plt.figure(figsize=(12, max(6, len(heatmap_data) * 0.15)))
    sns.heatmap(
        np.array(heatmap_data),
        yticklabels=label_list,
        xticklabels=2,
        cmap="magma",
        linewidths=0.3,
        cbar_kws={'label': 'CMI Score'},
    )
    plt.title("Layerwise CMI Heatmap (Per Datapoint)")
    plt.xlabel("Layer Index")
    plt.ylabel("Datapoint")
    plt.tight_layout()
    plt.savefig("new_plot_layer_heatmap.png", dpi=300)
    print("Generated: new_plot_layer_heatmap.png")
    plt.show()

def plot_layer_gate_signal(data):
    num_layers = 0
    for item in data:
        for rec in item:
            num_layers = max(num_layers, rec.get("layer", 0) + 1)

    sums = np.zeros(num_layers)
    counts = np.zeros(num_layers)
    for layer_rows in data:
        for rec in layer_rows:
            layer = rec.get("layer", 0)
            if 0 <= layer < num_layers:
                val = rec.get("cot_drop", 0.0) + rec.get("control_drop", 0.0)
                sums[layer] += val
                counts[layer] += 1

    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    plt.figure(figsize=(10, 4))
    plt.plot(range(num_layers), means, marker="o", linewidth=1.2)
    plt.title("Mean (cot_drop + control_drop) by Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Drop Sum")
    plt.tight_layout()
    plt.savefig("new_plot_layer_gate_signal.png", dpi=300)
    print("Generated: new_plot_layer_gate_signal.png")
    plt.show()

def plot_layer_drop_means(data):
    num_layers = 0
    for item in data:
        for rec in item:
            num_layers = max(num_layers, rec.get("layer", 0) + 1)

    cot_sums = np.zeros(num_layers)
    ctrl_sums = np.zeros(num_layers)
    counts = np.zeros(num_layers)
    for layer_rows in data:
        for rec in layer_rows:
            layer = rec.get("layer", 0)
            if 0 <= layer < num_layers:
                cot_sums[layer] += rec.get("cot_drop", 0.0)
                ctrl_sums[layer] += rec.get("control_drop", 0.0)
                counts[layer] += 1

    cot_means = np.divide(cot_sums, counts, out=np.zeros_like(cot_sums), where=counts > 0)
    ctrl_means = np.divide(ctrl_sums, counts, out=np.zeros_like(ctrl_sums), where=counts > 0)
    plt.figure(figsize=(10, 4))
    plt.plot(range(num_layers), cot_means, marker="o", linewidth=1.2, label="Mean cot_drop")
    plt.plot(range(num_layers), ctrl_means, marker="o", linewidth=1.2, label="Mean control_drop")
    plt.title("Mean CoT vs Control Drop by Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Drop")
    plt.legend()
    plt.tight_layout()
    plt.savefig("new_plot_layer_drop_means.png", dpi=300)
    print("Generated: new_plot_layer_drop_means.png")
    plt.show()

if __name__ == "__main__":
    suite_path = Path("data/cmi_suite.json")
    layer_path = Path("data/cmi_layers_per_layer_all.json")
    if not suite_path.exists() or not layer_path.exists():
        suite_path = Path("data/qwen/cmi_suite.json")
        layer_path = Path("data/qwen/cmi_layers_per_layer_all.json")

    if not suite_path.exists() or not layer_path.exists():
        raise FileNotFoundError(
            "Expected data/cmi_suite.json and data/cmi_layers_per_layer_all.json "
            "or data/qwen/cmi_suite.json and data/qwen/cmi_layers_per_layer_all.json"
        )

    suite = json.loads(suite_path.read_text())
    per_layer_all = json.loads(layer_path.read_text())

    ordered = []
    labels = []
    for item in suite:
        rows = per_layer_all.get(item["id"], [])
        ordered.append(rows)
        labels.append(item.get("id", ""))

    sample_indices = list(range(min(len(ordered), 20)))
    plot_layer_heatmap(ordered, sample_indices=sample_indices, y_labels=labels)
    plot_layer_gate_signal(ordered)
    plot_layer_drop_means(ordered)
