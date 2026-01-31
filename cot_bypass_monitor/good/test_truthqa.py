import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import json
import numpy as np
import contextlib
from pathlib import Path
import csv
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CMI_DENOM_FLOOR = 1e-3
CMI_BASE_FLOOR = 1e-3  # Lowered for higher sensitivity
CMI_DROP_FLOOR = 0.0    # Set to 0 to see even tiny signals
MAX_COT_TOKENS = 80

# --- MODEL LOADING ---
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.transformer.h

# --- CORE UTILITIES ---
@torch.no_grad()
def compute_answer_logprob(input_ids, answer_ids):
    out = model(input_ids=input_ids)
    logits = out.logits[0]
    start = input_ids.shape[1] - len(answer_ids)
    logp = 0.0
    for i, aid in enumerate(answer_ids):
        idx = start + i - 1
        if idx < 0 or idx >= logits.shape[0]: continue
        logp += float(torch.log_softmax(logits[idx], dim=-1)[aid].item())
    return logp

@torch.no_grad()
def cache_model_states(input_ids):
    out = model(input_ids=input_ids, output_hidden_states=True)
    return {'hidden_states': [h.detach().cpu() for h in out.hidden_states]}

@contextlib.contextmanager
def patch_model_layers(model, layer_specs):
    original_forwards = {}
    layers = get_model_layers(model)
    try:
        for idx, spec in layer_specs.items():
            layer = layers[idx]
            original_forwards[idx] = layer.forward
            
            def make_patched(old_f, s):
                def new_f(*args, **kwargs):
                    h = args[0]
                    patch = s['patch_tensor'].to(h.device)
                    mask = s['mask_positions']
                    h_mod = h.clone()
                    h_mod[:, mask, :] = patch[:, mask, :]
                    return old_f(h_mod, *args[1:], **kwargs)
                return new_f
            
            layer.forward = make_patched(layer.forward, spec)
        yield
    finally:
        for idx, old_f in original_forwards.items():
            layers[idx].forward = old_f

# --- INTERVENTION LOGIC ---
def build_patch_tensor(source_states, target_states, positions, mode="noise"):
    patch = target_states.clone()
    if mode == "noise":
        mean = target_states.mean(dim=(0, 1), keepdim=True)
        std = target_states.std(dim=(0, 1), keepdim=True) + 1e-6
        for pos in positions:
            noise = torch.randn_like(patch[:, pos, :]) * std[:, 0, :] + mean[:, 0, :]
            patch[:, pos, :] = noise
    else: # source patching
        src_len = source_states.shape[1]
        for pos in positions:
            src_pos = min(pos, src_len - 1)
            patch[:, pos, :] = source_states[:, src_pos, :]
    return patch

def run_cmi_analysis(full_ids, answer_ids, h_with, h_without, cot_pos, layer_spans, control_samples, mode="noise"):
    baseline_logp = compute_answer_logprob(full_ids, answer_ids)
    records = []
    seq_len = full_ids.shape[1]
    non_cot = [i for i in range(seq_len) if i not in cot_pos]

    for span in layer_spans:
        # CoT Intervention
        specs = {l: {'patch_tensor': build_patch_tensor(h_without[l+1], h_with[l+1], cot_pos, mode), 'mask_positions': cot_pos} for l in span}
        with patch_model_layers(model, specs):
            cot_logp = compute_answer_logprob(full_ids, answer_ids)
        cot_drop = max(0.0, baseline_logp - cot_logp)

        # Control Intervention
        ctrl_drops = []
        for _ in range(control_samples):
            ctrl_pos = random.sample(non_cot, k=min(len(cot_pos), len(non_cot)))
            c_specs = {l: {'patch_tensor': build_patch_tensor(h_without[l+1], h_with[l+1], ctrl_pos, mode), 'mask_positions': ctrl_pos} for l in span}
            with patch_model_layers(model, c_specs):
                ctrl_drops.append(max(0.0, baseline_logp - compute_answer_logprob(full_ids, answer_ids)))
        
        ctrl_drop = float(np.mean(ctrl_drops))
        raw = (cot_drop - ctrl_drop)
        denom = max(cot_drop + ctrl_drop, CMI_DENOM_FLOOR)
        cmi = max(0.0, raw) / denom if (cot_drop + ctrl_drop) > CMI_DROP_FLOOR else 0.0
        
        records.append({'layers': span, 'cmi_score': cmi, 'bypass_score': 1.0 - cmi})

    return {
        'baseline_logp': baseline_logp,
        'cmi_mean': float(np.mean([r['cmi_score'] for r in records])),
        'bypass_mean': float(np.mean([r['bypass_score'] for r in records])),
        'records': records
    }

def load_truthfulqa_comparison(n=100):
    data_path = Path("TruthfulQA.csv")
    if not data_path.exists(): 
        raise FileNotFoundError("TruthfulQA.csv not found!")
    examples = []
    with open(data_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n: break
            examples.append({
                "id": f"tqa_{i}",
                "question": row["Question"],
                "best_answer": row["Best Answer"],
                "incorrect_answer": row["Best Incorrect Answer"]
            })
    return examples


def main():
    examples = load_truthfulqa_comparison(10)
    results = []
    num_layers = len(get_model_layers(model))
    layer_spans = [[i] for i in range(num_layers)] 

    # Header for the console output
    print(f"{'ID':<6} | {'Type':<10} | {'Baseline LogP':<15} | {'CMI':<10} | {'Bypass':<10}")
    print("-" * 65)

    for ex in examples:
        # We test both the Correct (Truth) and Incorrect (Myth) answers
        answer_types = [
            ("Truth", ex["best_answer"]),
            ("Myth", ex["incorrect_answer"])
        ]

        for label, ans_text in answer_types:
            with_cot_text = f"Question: {ex['question']}\nLet's think step by step.\nFinal answer: {ans_text}"
            no_cot_text = f"Question: {ex['question']}\n {ans_text}"
            
            ids_with = tokenizer(with_cot_text, return_tensors="pt").input_ids.to(DEVICE)
            ids_no = tokenizer(no_cot_text, return_tensors="pt").input_ids.to(DEVICE)
            ans_ids = tokenizer.encode(ans_text, add_special_tokens=False)

            # Find "Let's think step by step." positions
            trigger = "Let's think step by step."
            start_char = with_cot_text.find(trigger)
            tokens_before = len(tokenizer.encode(with_cot_text[:start_char], add_special_tokens=False))
            tokens_trigger = len(tokenizer.encode(with_cot_text[:start_char+len(trigger)], add_special_tokens=False))
            cot_pos = list(range(tokens_before, tokens_trigger))

            # Cache states
            h_with = cache_model_states(ids_with)['hidden_states']
            h_without = cache_model_states(ids_no)['hidden_states']

            # Run CMI Analysis
            res = run_cmi_analysis(ids_with, ans_ids, h_with, h_without, cot_pos, layer_spans, control_samples=4, mode="noise")
            
            # Calculate No-CoT LogP for Delta calculation
            logp_no_cot = compute_answer_logprob(ids_no, ans_ids)
            logp_delta = res['baseline_logp'] - logp_no_cot

            print(f"{ex['id']:<6} | {label:<10} | {res['baseline_logp']:<15.4f} | {res['cmi_mean']:<10.4f} | {res['bypass_mean']:<10.4f}")
            
            results.append({
                "id": ex["id"],
                "type": label,
                "answer_text": ans_text,
                "baseline_logp": res['baseline_logp'],
                "logp_delta": logp_delta,
                "cmi": res['cmi_mean'],
                "bypass": res['bypass_mean']
            })
        print("-" * 65) # Visual separator between questions

    # Save detailed results
    output_file = "truthfulqa_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()

# --- DATA LOADING (TRUTHFUL QA) ---
# def load_truthfulqa(n=100):
#     data_path = Path("TruthfulQA.csv")
#     if not data_path.exists(): raise FileNotFoundError("TruthfulQA.csv not found!")
#     examples = []
#     with open(data_path, mode='r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for i, row in enumerate(reader):
#             if i >= n: break
#             q, a = row["Question"], row["Best Answer"]
#             examples.append({
#                 "id": f"tqa_{i}",
#                 "with_cot": f"Question: {q}\nLet's think step by step.\nFinal answer: {a}",
#                 "no_cot": f"Question: {q}\n {a}",
#                 "answer": a
#             })
#     return examples

# # --- MAIN EXECUTION ---

# def main():
#     examples = load_truthfulqa(10)  # Testing with 10 first
#     results = []
#     num_layers = len(get_model_layers(model))
#     # Analyzing individual layers to see exactly where reasoning happens
#     layer_spans = [[i] for i in range(num_layers)] 

#     print(f"{'ID':<10} | {'Baseline LogP':<15} | {'CMI':<10} | {'Bypass':<10}")
#     print("-" * 55)

#     for ex in examples:
#         ids_with = tokenizer(ex['with_cot'], return_tensors="pt").input_ids.to(DEVICE)
#         ids_no = tokenizer(ex['no_cot'], return_tensors="pt").input_ids.to(DEVICE)
#         ans_ids = tokenizer.encode(ex['answer'], add_special_tokens=False)

#         # Token alignment for "Let's think step by step."
#         trigger = "Let's think step by step."
#         start_char = ex['with_cot'].find(trigger)
#         tokens_before = len(tokenizer.encode(ex['with_cot'][:start_char], add_special_tokens=False))
#         tokens_trigger = len(tokenizer.encode(ex['with_cot'][:start_char+len(trigger)], add_special_tokens=False))
#         cot_pos = list(range(tokens_before, tokens_trigger))

#         # Cache states for patching
#         h_with = cache_model_states(ids_with)['hidden_states']
#         h_without = cache_model_states(ids_no)['hidden_states']

#         # Run analysis (Noise mode is more disruptive and better for short answers)
#         res = run_cmi_analysis(ids_with, ans_ids, h_with, h_without, cot_pos, layer_spans, control_samples=4, mode="noise")
        
#         # Report baseline along with CMI metrics
#         print(f"{ex['id']:<10} | {res['baseline_logp']:<15.4f} | {res['cmi_mean']:<10.4f} | {res['bypass_mean']:<10.4f}")
        
#         results.append({
#             "id": ex["id"], 
#             "baseline_logp": res['baseline_logp'],
#             "cmi": res['cmi_mean'], 
#             "bypass": res['bypass_mean']
#         })

#     # Save to JSON
#     output_file = "truthfulqa_detailed_results.json"
#     with open(output_file, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nResults saved to {output_file}")

# if __name__ == "__main__":
#     main()