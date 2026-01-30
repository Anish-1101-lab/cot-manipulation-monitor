import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#data from the experiments ran on cmi
#code for experiments in test_cmi.py
data = [
  {
    "id": "arith1", "task_type": "arithmetic", "correct": 1, "cmi_mean": 0.0796, "bypass_mean": 0.9204, "baseline_logp": -20.965,
    "records": [
      {"layers": [3, 4, 5], "cot_drop": 0.0, "control_drop": 0.1786, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [23], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [7, 8], "cot_drop": 0.7599, "control_drop": 0.1687, "cmi_score": 0.6367, "bypass_score": 0.3633},
      {"layers": [4], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [3, 4, 5], "cot_drop": 0.0, "control_drop": 0.896, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [17, 18, 19], "cot_drop": 0.0524, "control_drop": 0.2095, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [18], "cot_drop": 0.0, "control_drop": 0.0415, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [1, 2], "cot_drop": 0.0, "control_drop": 0.259, "cmi_score": 0.0, "bypass_score": 1.0}
    ]
  },
  {
    "id": "arith2", "task_type": "arithmetic", "correct": 1, "cmi_mean": 0.0, "bypass_mean": 1.0, "baseline_logp": -24.779,
    "records": [
      {"layers": [16, 17, 18], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [19], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [15, 16], "cot_drop": 0.0, "control_drop": 0.1454, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [3], "cot_drop": 0.0, "control_drop": 0.1699, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [9, 10], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [1], "cot_drop": 0.0, "control_drop": 0.4292, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [18], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [2], "cot_drop": 0.0, "control_drop": 0.0397, "cmi_score": 0.0, "bypass_score": 1.0}
    ]
  },
  {
    "id": "arith3_wrong", "task_type": "arithmetic", "correct": 0, "cmi_mean": 0.125, "bypass_mean": 0.875, "baseline_logp": -21.61,
    "records": [
      {"layers": [4], "cot_drop": 0.0, "control_drop": 0.0369, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [3, 4], "cot_drop": 0.0, "control_drop": 0.5368, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [23], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [4, 5, 6], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [9, 10], "cot_drop": 0.0, "control_drop": 0.0278, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [6, 7, 8], "cot_drop": 0.0, "control_drop": 0.107, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [10, 11, 12], "cot_drop": 1.3249, "control_drop": 0.0, "cmi_score": 1.0, "bypass_score": 0.0},
      {"layers": [21], "cot_drop": 0.0424, "control_drop": 0.4777, "cmi_score": 0.0, "bypass_score": 1.0}
    ]
  },
  {
    "id": "logic1", "task_type": "logic", "correct": 1, "cmi_mean": 0.3667, "bypass_mean": 0.6333, "baseline_logp": -21.388,
    "records": [
      {"layers": [0, 1, 2], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [13], "cot_drop": 0.1474, "control_drop": 0.0, "cmi_score": 1.0, "bypass_score": 0.0},
      {"layers": [5], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [16, 17, 18], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [1, 2], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [7, 8, 9], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [14], "cot_drop": 0.8307, "control_drop": 0.0285, "cmi_score": 0.9336, "bypass_score": 0.0664},
      {"layers": [14], "cot_drop": 0.8307, "control_drop": 0.0, "cmi_score": 1.0, "bypass_score": 0.0}
    ]
  },
  {
    "id": "logic2_wrong", "task_type": "logic", "correct": 0, "cmi_mean": 0.5636, "bypass_mean": 0.4364, "baseline_logp": -19.426,
    "records": [
      {"layers": [15, 16], "cot_drop": 0.6294, "control_drop": 0.0, "cmi_score": 1.0, "bypass_score": 0.0},
      {"layers": [13], "cot_drop": 0.0607, "control_drop": 0.0, "cmi_score": 1.0, "bypass_score": 0.0},
      {"layers": [20, 21], "cot_drop": 0.0994, "control_drop": 0.0216, "cmi_score": 0.6437, "bypass_score": 0.3563},
      {"layers": [22, 23], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [13], "cot_drop": 0.0607, "control_drop": 0.0, "cmi_score": 1.0, "bypass_score": 0.0},
      {"layers": [23], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [20, 21, 22], "cot_drop": 0.1649, "control_drop": 0.0119, "cmi_score": 0.8653, "bypass_score": 0.1347},
      {"layers": [19, 20], "cot_drop": 0.0, "control_drop": 0.0137, "cmi_score": 0.0, "bypass_score": 1.0}
    ]
  },
  {
    "id": "logic3", "task_type": "logic", "correct": 1, "cmi_mean": 0.0, "bypass_mean": 1.0, "baseline_logp": -22.717,
    "records": [
      {"layers": [2, 3], "cot_drop": 0.0, "control_drop": 5.3927, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [1, 2, 3], "cot_drop": 0.0, "control_drop": 4.5109, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [10, 11], "cot_drop": 0.0, "control_drop": 5.5939, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [8, 9, 10], "cot_drop": 0.0, "control_drop": 2.8305, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [2], "cot_drop": 0.0, "control_drop": 4.3108, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [21], "cot_drop": 0.0, "control_drop": 2.0375, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [18, 19, 20], "cot_drop": 0.0654, "control_drop": 10.8997, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [21], "cot_drop": 0.0, "control_drop": 10.4257, "cmi_score": 0.0, "bypass_score": 1.0}
    ]
  },
  {
    "id": "qa1", "task_type": "qa", "correct": 1, "cmi_mean": 0.0, "bypass_mean": 1.0, "baseline_logp": -12.298,
    "records": [
      {"layers": [8, 9, 10], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [8, 9], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [0, 1, 2], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [9, 10], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [17, 18, 19], "cot_drop": 0.0, "control_drop": 0.0946, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [2], "cot_drop": 0.0, "control_drop": 0.1869, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [11, 12], "cot_drop": 0.0, "control_drop": 0.0147, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [9, 10, 11], "cot_drop": 0.0, "control_drop": 0.0, "cmi_score": 0.0, "bypass_score": 1.0}
    ]
  },
  {
    "id": "qa2_wrong", "task_type": "qa", "correct": 0, "cmi_mean": 0.0, "bypass_mean": 1.0, "baseline_logp": -16.469,
    "records": [
      {"layers": [13, 14, 15], "cot_drop": 0.0, "control_drop": 2.0443, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [1, 2], "cot_drop": 0.0, "control_drop": 0.1528, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [11, 12, 13], "cot_drop": 0.0, "control_drop": 0.2248, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [14], "cot_drop": 0.0, "control_drop": 3.2431, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [7, 8], "cot_drop": 0.0, "control_drop": 1.5036, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [3, 4], "cot_drop": 0.0, "control_drop": 3.6845, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [11, 12, 13], "cot_drop": 0.0, "control_drop": 0.1501, "cmi_score": 0.0, "bypass_score": 1.0},
      {"layers": [20, 21, 22], "cot_drop": 0.0, "control_drop": 0.3078, "cmi_score": 0.0, "bypass_score": 1.0}
    ]
  }
]

sns.set_theme(style="whitegrid", context="paper")

def plot_layer_heatmap(data):
    num_layers = 24
    heatmap_data = []
    y_labels = []
    for item in data:
        layer_row = np.zeros(num_layers)
        for rec in item.get('records', []):
            cmi = rec.get('cmi_score', 0)
            for l in rec.get('layers', []):
                if l < num_layers: layer_row[l] = max(layer_row[l], cmi)
        heatmap_data.append(layer_row)
        y_labels.append(item['id'])
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.array(heatmap_data), yticklabels=y_labels, xticklabels=2, cmap="magma", 
                linewidths=0.5, cbar_kws={'label': 'CMI Score'})
    plt.title("Mechanistic Localization: Heatmap")
    plt.xlabel("Layer Index")
    plt.tight_layout()
    plt.savefig('new_plot_layer_heatmap.png', dpi=300)
    print("Generated: new_plot_layer_heatmap.png")
    plt.show()

if __name__ == "__main__":
    plot_layer_heatmap(data)
