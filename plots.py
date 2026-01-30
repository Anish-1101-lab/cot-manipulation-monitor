import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. FULL DATASET
# ==========================================
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

# ==========================================
# PART A: ORIGINAL PAPER EQUIVALENTS
# ==========================================

def plot_original_figure2_bars(data):
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df, x='id', y='cmi_mean', ax=axes[0], palette="mako", hue='id', legend=False)
    axes[0].set_title("CoT Mediated Influence (CMI) by Sample")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].set_ylabel("Mean CMI Score")
    axes[0].set_ylim(0, 1.0)

    sns.barplot(data=df, x='id', y='bypass_mean', ax=axes[1], palette="rocket_r", hue='id', legend=False)
    axes[1].set_title("Bypass Score (1 - CMI)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].set_ylabel("Bypass Score")
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig('fig2_original_cmi_bypass.png', dpi=300)
    print("Generated: fig2_original_cmi_bypass.png")
    plt.show()

def plot_original_3d_scatter(data):
    df = pd.DataFrame(data)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df['cmi_mean']
    y = df['bypass_mean']
    z = df['baseline_logp']
    
    task_types = df['task_type'].unique()
    colors = ['r', 'g', 'b', 'orange']
    
    for i, task in enumerate(task_types):
        subset = df[df['task_type'] == task]
        ax.scatter(subset['cmi_mean'], subset['bypass_mean'], subset['baseline_logp'], 
                   c=colors[i % len(colors)], label=task, s=100, edgecolors='w')

    ax.set_xlabel('CMI Score')
    ax.set_ylabel('Bypass Score')
    ax.set_zlabel('Baseline LogP')
    ax.set_title('3D Relationship: CMI vs Bypass vs Confidence')
    ax.legend()
    plt.savefig('fig2_original_3d_scatter.png', dpi=300)
    print("Generated: fig2_original_3d_scatter.png")
    plt.show()

# ==========================================
# PART B: PREVIOUSLY APPROVED PLOTS
# ==========================================

def plot_reasoning_vs_hallucination(data):
    df = pd.DataFrame(data)
    df['Status'] = df['correct'].map({1: 'Correct', 0: 'Incorrect'})
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='cmi_mean', y='baseline_logp', hue='Status', style='task_type',
                    palette={'Correct': 'green', 'Incorrect': 'red'}, s=250, alpha=0.85)
    
    # Annotate with offset
    for i, row in df.iterrows():
        plt.text(row['cmi_mean']+0.015, row['baseline_logp']+0.2, row['id'], fontsize=9, weight='bold')
        
    plt.axvline(x=0.05, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.01, -25, "Bypassed / Hallucination", rotation=90, verticalalignment='bottom', color='gray')
    plt.text(0.06, -25, "Mediated Reasoning", rotation=90, verticalalignment='bottom', color='gray')
    plt.title("Reasoning vs. Hallucination: CMI vs Confidence")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('new_plot_reasoning_vs_hallucination.png', dpi=300)
    print("Generated: new_plot_reasoning_vs_hallucination.png")
    plt.show()

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

def plot_control_validity(data):
    plot_points = []
    for item in data:
        for rec in item.get('records', []):
            plot_points.append({'id': item['id'], 'cot_drop': rec['cot_drop'], 
                                'control_drop': rec['control_drop'], 'type': item['task_type']})
    df = pd.DataFrame(plot_points)
    plt.figure(figsize=(8, 8))
    limit = max(df['cot_drop'].max(), df['control_drop'].max()) * 1.05
    plt.plot([0, limit], [0, limit], ls="--", c=".3", label="y=x (Noise Floor)")
    sns.scatterplot(data=df, x='control_drop', y='cot_drop', hue='type', style='type', s=120, alpha=0.7)
    
    # Annotate interesting points
    for i, row in df.iterrows():
        if row['control_drop'] > 2.0 or (row['cot_drop'] > 0.5 and row['control_drop'] < 0.2):
             plt.text(row['control_drop'], row['cot_drop'], row['id'], fontsize=7)

    plt.title("Validity Check: CoT Drop vs. Control Drop")
    plt.tight_layout()
    plt.savefig('new_plot_control_validity.png', dpi=300)
    print("Generated: new_plot_control_validity.png")
    plt.show()

# ==========================================
# PART C: NEW INSIGHT PLOTS
# ==========================================

def plot_reasoning_sparsity(data):
    """
    Shows Mean CMI vs Max CMI to illustrate reasoning sparsity.
    Samples near diagonal use CoT consistently.
    Samples high/left use CoT sparsely (localized 'flash of insight').
    """
    points = []
    for item in data:
        # Calculate Max CMI for this sample
        cmi_scores = [r.get('cmi_score', 0) for r in item.get('records', [])]
        max_cmi = max(cmi_scores) if cmi_scores else 0
        points.append({
            'id': item['id'],
            'mean_cmi': item['cmi_mean'],
            'max_cmi': max_cmi,
            'task_type': item['task_type']
        })
    
    df = pd.DataFrame(points)
    plt.figure(figsize=(8, 8))
    
    # y=x line
    plt.plot([0, 1], [0, 1], ls="--", c=".3", label="Consistent Reasoning (Mean ≈ Max)")
    
    sns.scatterplot(data=df, x='mean_cmi', y='max_cmi', hue='task_type', style='task_type', s=150)
    
    for i, row in df.iterrows():
        plt.text(row['mean_cmi']+0.02, row['max_cmi'], row['id'], fontsize=9)
        
    plt.title("Reasoning Sparsity: Mean vs. Peak Influence")
    plt.xlabel("Mean CMI (Average Influence)")
    plt.ylabel("Max CMI (Peak Influence)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('new_plot_reasoning_sparsity.png', dpi=300)
    print("Generated: new_plot_reasoning_sparsity.png")
    plt.show()

def plot_cmi_distribution(data):
    """
    Box/Swarm plot showing the distribution of CMI scores *within* each sample.
    Reveals consistency variance (e.g., arith1 has high variance, qa1 has zero variance).
    """
    rows = []
    for item in data:
        for rec in item.get('records', []):
            rows.append({
                'id': item['id'],
                'cmi_score': rec.get('cmi_score', 0),
                'task_type': item['task_type']
            })
    df = pd.DataFrame(rows)
    
    plt.figure(figsize=(10, 6))
    
    # Box plot for distribution range
    sns.boxplot(data=df, x='id', y='cmi_score', palette="pastel", showfliers=False)
    # Strip plot to show actual data points
    sns.stripplot(data=df, x='id', y='cmi_score', color='black', alpha=0.6, jitter=True)
    
    plt.title("Intra-Sample Reasoning Consistency (CMI Distribution)")
    plt.ylabel("CMI Score per Layer Span")
    plt.xlabel("Sample ID")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('new_plot_cmi_distribution.png', dpi=300)
    print("Generated: new_plot_cmi_distribution.png")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(">>> Generating Original Paper Equivalents...")
    plot_original_figure2_bars(data)
    plot_original_3d_scatter(data)
    
    print("\n>>> Generating Validated Diagnostic Plots...")
    plot_reasoning_vs_hallucination(data)
    plot_layer_heatmap(data)
    plot_control_validity(data)
    
    print("\n>>> Generating New Insight Plots (Sparsity & Consistency)...")
    plot_reasoning_sparsity(data)
    plot_cmi_distribution(data)
    
    print("\n>>> All plots generated successfully.")