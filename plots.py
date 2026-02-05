import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

sns.set_theme(style="whitegrid", context="paper")

def plot_layer_drop_means(data, model_name, output_dir):
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
    plt.plot(
        range(num_layers),
        cot_means,
        marker="o",
        linewidth=1.2,
        label="Mean cot_drop",
    )
    plt.plot(
        range(num_layers),
        ctrl_means,
        marker="o",
        linewidth=1.2,
        label="Mean control_drop",
    )
    plt.title(f"Mean CoT vs Control Drop ({model_name})")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Drop")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"new_plot_layer_drop_means_{model_name}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Generated: {output_path}")
    plt.close()

def iter_per_layer_files(modelsrun_dir):
    prefix = "cmi_layers_per_layer_all_"
    for path in modelsrun_dir.iterdir():
        if path.is_file() and path.name.startswith(prefix):
            yield path

if __name__ == "__main__":
    modelsrun_dir = Path("modelsrun")
    if not modelsrun_dir.exists():
        raise FileNotFoundError("Expected modelsrun directory with per-layer results.")

    per_layer_files = sorted(iter_per_layer_files(modelsrun_dir))
    if not per_layer_files:
        raise FileNotFoundError("No cmi_layers_per_layer_all_* files found in modelsrun.")

    for path in per_layer_files:
        data = json.loads(path.read_text())
        model_name = path.name.replace("cmi_layers_per_layer_all_", "")
        if model_name.endswith(".json"):
            model_name = model_name[:-5]
        ordered = list(data.values())
        plot_layer_drop_means(ordered, model_name=model_name, output_dir=modelsrun_dir)
