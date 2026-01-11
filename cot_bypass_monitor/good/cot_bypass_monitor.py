import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import json
import numpy as np
import re
import contextlib
from copy import deepcopy
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CMI_DENOM_FLOOR = 1e-3
CMI_BASE_FLOOR = 1e-2
CMI_DROP_FLOOR = 1e-4
ROBUST_ABS_THRESHOLD = 0.02
MAX_COT_TOKENS = 80
MAX_COT_SENTENCES = 10
RESULTS_JSON = "cot_bypass_results.json"
RESULTS_JSONL = "cot_bypass_results.jsonl"
PLOT_PATH = "cot_bypass_plots.png"
DATASET_MODE = "synthetic"  # synthetic or hf

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def get_model_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Unsupported model architecture: cannot find transformer layers")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seeds()

def apply_chat_template(user_text):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text

def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}

def concat_answer(prefix_ids, answer_ids):
    a = torch.tensor(answer_ids, device=DEVICE).unsqueeze(0)
    return torch.cat([prefix_ids, a], dim=1)

def build_attention_mask(input_ids):
    return torch.ones_like(input_ids, device=input_ids.device)

def find_cot_token_positions(full_prompt, cot_trigger, tokenizer):
    """
    Find exact token positions of the CoT trigger phrase using tokenization alignment.

    Args:
        full_prompt: The complete prompt string including CoT trigger
        cot_trigger: The CoT trigger phrase (e.g., "Let's think step by step.")
        tokenizer: The tokenizer instance

    Returns:
        List of token positions corresponding to the CoT trigger
    """
    # Tokenize the full prompt
    full_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)

    # Find where the CoT trigger starts in the string
    cot_start_char = full_prompt.find(cot_trigger)
    if cot_start_char == -1:
        # Fallback: return last few tokens if trigger not found
        return list(range(max(0, len(full_tokens) - 7), len(full_tokens)))

    # Tokenize text before and including the CoT trigger
    text_before = full_prompt[:cot_start_char]
    text_with_cot = full_prompt[:cot_start_char + len(cot_trigger)]

    tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
    tokens_with_cot = tokenizer.encode(text_with_cot, add_special_tokens=False)

    # The CoT tokens are the difference
    cot_start_idx = len(tokens_before)
    cot_end_idx = len(tokens_with_cot)

    cot_positions = list(range(cot_start_idx, cot_end_idx))

    # Validate the positions
    if not cot_positions or cot_start_idx >= len(full_tokens):
        # Fallback to last tokens if something went wrong
        return list(range(max(0, len(full_tokens) - 7), len(full_tokens)))

    return cot_positions

def find_cot_span_positions(full_prompt, cot_span, tokenizer):
    """
    Find token positions for an arbitrary CoT span inside the full prompt.
    """
    span_start_char = full_prompt.find(cot_span)
    if span_start_char == -1:
        return []

    text_before = full_prompt[:span_start_char]
    text_with_span = full_prompt[:span_start_char + len(cot_span)]

    tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
    tokens_with_span = tokenizer.encode(text_with_span, add_special_tokens=False)

    span_start_idx = len(tokens_before)
    span_end_idx = len(tokens_with_span)
    if span_start_idx >= span_end_idx:
        return []

    return list(range(span_start_idx, span_end_idx))

def generate_cot_text(prompt, context_hint=None, max_new_tokens=MAX_COT_TOKENS):
    """
    Generate a short CoT continuation from the model for span-based patching.
    """
    prompt_text = apply_chat_template(prompt)
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(DEVICE)
    attention_mask = build_attention_mask(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = output[0, input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Keep a short span and stop at first newline if present.
    text = text.split("\n", 1)[0].strip()
    if text:
        parts = [p.strip() for p in text.split(".") if p.strip()]
        text = ". ".join(parts[:MAX_COT_SENTENCES]).strip()
        if text and not text.endswith("."):
            text += "."
    if len(text.split()) < 5 and context_hint:
        first_sentence = context_hint.split(".", 1)[0].strip()
        if first_sentence:
            text = (
                f"First, the context says: {first_sentence}. "
                "Then, use that to identify the answer. "
                "Therefore, the answer follows from the context."
            )
    return text

def generate_answer_text(prompt, max_new_tokens=20):
    prompt_text = apply_chat_template(prompt)
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(DEVICE)
    attention_mask = build_attention_mask(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = output[0, input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def clean_generated_answer(text):
    if not text:
        return ""
    for marker in ["Human:", "Assistant:", "\n"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text.strip()

def is_bad_generated(text, gold_answer):
    if not text:
        return True
    if "Human:" in text or "Assistant:" in text:
        return True
    if gold_answer and gold_answer.strip().isdigit():
        return not text.strip().isdigit()
    return False

def build_synthetic_math_dataset(n):
    samples = []
    for _ in range(n):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        c = random.randint(2, 9)
        question = f"What is ({a} + {b}) * {c}?"
        answer = str((a + b) * c)
        samples.append({"question": question, "answer": answer})
    return samples

def perform_sensitivity_analysis(prefix_ids, answer_ids, base_cot_positions, num_layers, control_samples=8,
                                 span_samples=8, max_span_size=3, layer_spans=None):
    """
    Perform sensitivity analysis by expanding/shrinking the intervention window.
    Tests if CMI changes drastically with boundary adjustments.

    Returns:
        dict: Contains base CMI and sensitivity metrics
    """
    seq_length = prefix_ids.shape[1]

    # Base case
    base_result = run_intervention_analysis(
        {'input_ids': prefix_ids},
        {'input_ids': prefix_ids[:, :-len(base_cot_positions)]},  # Remove CoT part
        answer_ids,
        base_cot_positions,
        num_layers=num_layers,
        control_samples=control_samples,
        span_samples=span_samples,
        max_span_size=max_span_size,
        layer_spans=layer_spans
    )
    base_cmi = base_result['cmi_mean']

    # Expand by 1 token on each side (if possible)
    expanded_positions = base_cot_positions.copy()
    if base_cot_positions[0] > 0:
        expanded_positions = [base_cot_positions[0] - 1] + expanded_positions
    if base_cot_positions[-1] < seq_length - 1:
        expanded_positions = expanded_positions + [base_cot_positions[-1] + 1]

    expanded_result = run_intervention_analysis(
        {'input_ids': prefix_ids},
        {'input_ids': prefix_ids[:, :-len(expanded_positions)]},
        answer_ids,
        expanded_positions,
        num_layers=num_layers,
        control_samples=control_samples,
        span_samples=span_samples,
        max_span_size=max_span_size,
        layer_spans=layer_spans
    )
    expanded_cmi = expanded_result['cmi_mean']

    # Shrink by 1 token on each side (if possible)
    if len(base_cot_positions) > 2:
        shrunk_positions = base_cot_positions[1:-1]
    else:
        shrunk_positions = base_cot_positions

    if shrunk_positions:
        shrunk_result = run_intervention_analysis(
            {'input_ids': prefix_ids},
            {'input_ids': prefix_ids[:, :-len(shrunk_positions)]},
            answer_ids,
            shrunk_positions,
            num_layers=num_layers,
            control_samples=control_samples,
            span_samples=span_samples,
            max_span_size=max_span_size,
            layer_spans=layer_spans
        )
        shrunk_cmi = shrunk_result['cmi_mean']
    else:
        shrunk_cmi = base_cmi

    # Calculate sensitivity metrics
    max_deviation = max(
        abs(expanded_cmi - base_cmi),
        abs(shrunk_cmi - base_cmi)
    )

    relative_sensitivity = max_deviation / max(base_cmi, CMI_BASE_FLOOR)
    is_robust = (relative_sensitivity < 0.15) or (max_deviation < ROBUST_ABS_THRESHOLD)

    return {
        'base_cmi': base_cmi,
        'expanded_cmi': expanded_cmi,
        'shrunk_cmi': shrunk_cmi,
        'max_deviation': max_deviation,
        'relative_sensitivity': relative_sensitivity,
        'is_robust': is_robust
    }

@torch.no_grad()
def compute_answer_logprob(input_ids, answer_ids):
    attention_mask = build_attention_mask(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False)
    logits = out.logits[0]
    start = input_ids.shape[1] - len(answer_ids)
    logp = 0.0

    for i, aid in enumerate(answer_ids):
        idx = start + i - 1
        if idx < 0 or idx >= logits.shape[0]:
            continue
        token_logits = logits[idx]
        token_logp = float(torch.log_softmax(token_logits, dim=-1)[aid].item())
        logp += token_logp
    return float(logp)

@torch.no_grad()
def cache_model_states(input_ids):
    attention_mask = build_attention_mask(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=True)

    # Handle attentions carefully - it might be a tuple with None or None itself
    attentions = None
    if out.attentions is not None:
        try:
            # Check if it's iterable and contains valid tensors
            if hasattr(out.attentions, '__iter__'):
                attentions = [a.detach().cpu() for a in out.attentions if a is not None]
                if not attentions:
                    attentions = None
        except (AttributeError, TypeError):
            attentions = None

    return {
        'logits': out.logits.detach().cpu(),
        'hidden_states': [h.detach().cpu() for h in out.hidden_states],
        'attentions': attentions
    }

@contextlib.contextmanager
def patch_model_layers(model, layer_specs):
    original_forwards = {}
    try:
        for layer_idx, spec in layer_specs.items():
            layers = get_model_layers(model)
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            original_forwards[layer_idx] = layer.forward

            def create_patched_forward(original_forward, spec):
                mode = spec.get('mode', 'mean')
                patch_tensor = spec.get('patch_tensor', None)
                mask_positions = spec.get('mask_positions', None)

                def patched_forward(*args, **kwargs):
                    if len(args) >= 1:
                        hidden_states = args[0]

                        if patch_tensor is not None:
                            patch = patch_tensor.to(hidden_states.device)
                        else:
                            patch = None

                        if mode in ('mean', 'source'):
                            modified_states = hidden_states.clone()
                            if patch is not None and hidden_states.shape == patch.shape:
                                if mask_positions:
                                    modified_states[:, mask_positions, :] = patch[:, mask_positions, :].to(modified_states.device)
                                else:
                                    modified_states = patch.to(modified_states.device)
                            elif mask_positions:
                                mean_activation = hidden_states.mean(dim=1, keepdim=True).expand_as(hidden_states)
                                modified_states[:, mask_positions, :] = mean_activation[:, mask_positions, :]

                            return original_forward(modified_states, *args[1:], **kwargs)

                        elif mode == 'zero_delta':
                            output = original_forward(*args, **kwargs)
                            if isinstance(output, tuple):
                                hidden_output = output[0]
                                extra_outputs = output[1:]
                            else:
                                hidden_output = output
                                extra_outputs = ()

                            delta = hidden_output - hidden_states
                            if mask_positions:
                                delta_modified = delta.clone()
                                delta_modified[:, mask_positions, :] = 0.0
                            else:
                                delta_modified = torch.zeros_like(delta)

                            final_output = hidden_states + delta_modified

                            if isinstance(output, tuple):
                                return (final_output,) + extra_outputs
                            else:
                                return final_output
                        else:
                            return original_forward(*args, **kwargs)
                    else:
                        return original_forward(*args, **kwargs)
                return patched_forward

            layer.forward = create_patched_forward(layer.forward, spec)

        yield

    finally:
        for layer_idx, original_forward in original_forwards.items():
            layers = get_model_layers(model)
            if layer_idx < len(layers):
                layers[layer_idx].forward = original_forward

def create_mean_patch(hidden_tensor, positions):
    if hidden_tensor is None or hidden_tensor.ndim != 3:
        return hidden_tensor
    tensor_copy = hidden_tensor.clone()
    batch_size, seq_len, hidden_dim = tensor_copy.shape
    mean_activation = tensor_copy.mean(dim=1, keepdim=True)

    for pos in positions:
        if 0 <= pos < seq_len:
            tensor_copy[:, pos, :] = mean_activation.squeeze(1)
    return tensor_copy

def align_source_patch(source_states, target_states, positions, permute_positions=None):
    if source_states is None or target_states is None:
        return target_states
    if source_states.ndim != 3 or target_states.ndim != 3:
        return target_states

    patch = target_states.clone()
    src_len = source_states.shape[1]
    if src_len == 0:
        return patch

    for idx, pos in enumerate(positions):
        if 0 <= pos < patch.shape[1]:
            src_pos = pos
            if permute_positions is not None and idx < len(permute_positions):
                src_pos = permute_positions[idx]
            src_pos = min(src_pos, src_len - 1)
            patch[:, pos, :] = source_states[:, src_pos, :]
    return patch

def build_patch_tensor(source_states, target_states, positions, patch_mode, permute_positions=None):
    if patch_mode == "noise":
        if target_states is None or target_states.ndim != 3:
            return target_states
        patch = target_states.clone()
        mean = target_states.mean(dim=(0, 1), keepdim=True)
        std = target_states.std(dim=(0, 1), keepdim=True) + 1e-6
        for pos in positions:
            if 0 <= pos < patch.shape[1]:
                noise = torch.randn_like(patch[:, pos, :]) * std[:, 0, :] + mean[:, 0, :]
                patch[:, pos, :] = noise
        return patch

    if patch_mode == "shuffle":
        return align_source_patch(source_states, target_states, positions, permute_positions=permute_positions)

    # default: source patch
    return align_source_patch(source_states, target_states, positions, permute_positions=None)

def run_cmi_analysis(full_with_cot, answer_ids, hidden_states_with, hidden_states_without,
                     cot_positions, non_cot_positions, layer_spans, control_samples,
                     patch_mode="source", permute_positions=None):
    baseline_logp = compute_answer_logprob(full_with_cot, answer_ids)
    intervention_records = []

    for span in layer_spans:
        layer_specs = {}
        for layer_idx in span:
            source_states = hidden_states_without[layer_idx + 1]
            target_states = hidden_states_with[layer_idx + 1]
            patch = build_patch_tensor(source_states, target_states, cot_positions, patch_mode, permute_positions)
            layer_specs[layer_idx] = {
                'mode': 'source',
                'patch_tensor': patch,
                'mask_positions': cot_positions
            }

        with patch_model_layers(model, layer_specs):
            cot_logp = compute_answer_logprob(full_with_cot, answer_ids)
        cot_drop = max(0.0, baseline_logp - cot_logp)

        control_drops = []
        for _ in range(control_samples):
            if not non_cot_positions:
                control_drops.append(0.0)
                continue
            control_positions = random.sample(non_cot_positions, k=min(len(cot_positions), len(non_cot_positions)))
            control_specs = {}
            for layer_idx in span:
                source_states = hidden_states_without[layer_idx + 1]
                target_states = hidden_states_with[layer_idx + 1]
                patch = build_patch_tensor(source_states, target_states, control_positions, patch_mode, permute_positions)
                control_specs[layer_idx] = {
                    'mode': 'source',
                    'patch_tensor': patch,
                    'mask_positions': control_positions
                }

            with patch_model_layers(model, control_specs):
                control_logp = compute_answer_logprob(full_with_cot, answer_ids)
            control_drops.append(max(0.0, baseline_logp - control_logp))

        control_drop = float(np.mean(control_drops))
        cmi_raw = cot_drop - control_drop
        total_drop = cot_drop + control_drop
        if total_drop < CMI_DROP_FLOOR:
            cmi_score = 0.0
        else:
            denom = max(total_drop, CMI_DENOM_FLOOR)
            cmi_score = max(0.0, cmi_raw) / denom
        bypass_score = 1.0 - cmi_score

        intervention_records.append({
            'layers': span,
            'cot_drop': cot_drop,
            'control_drop': control_drop,
            'cmi_raw': cmi_raw,
            'cmi_score': cmi_score,
            'bypass_score': bypass_score
        })

    cmi_scores = [r['cmi_score'] for r in intervention_records]
    bypass_scores = [r['bypass_score'] for r in intervention_records]

    return {
        'baseline_logp': baseline_logp,
        'cmi_mean': float(np.mean(cmi_scores)) if cmi_scores else 0.0,
        'bypass_mean': float(np.mean(bypass_scores)) if bypass_scores else 0.0,
        'intervention_records': intervention_records
    }

def sample_layer_spans(num_samples, num_layers, max_span_size=3):
    spans = []
    for _ in range(num_samples):
        span_size = random.randint(1, max_span_size)
        start_layer = random.randint(0, num_layers - span_size)
        spans.append(list(range(start_layer, start_layer + span_size)))
    return spans

def build_fixed_layer_spans(num_layers):
    spans = []
    mid = num_layers // 2
    tail = max(0, num_layers - 1)
    spans.extend([[0], [mid], [tail]])
    spans.append(list(range(0, min(2, num_layers))))
    spans.append(list(range(max(0, mid - 1), min(num_layers, mid + 2))))
    spans.append(list(range(max(0, tail - 2), tail + 1)))
    return spans

def run_intervention_analysis(ids_with_cot, ids_without_cot, answer_ids, cot_positions,
                            num_layers=None, control_samples=8, span_samples=8, max_span_size=3, layer_spans=None):
    if num_layers is None:
        num_layers = len(get_model_layers(model))

    with_cot_prefix = ids_with_cot['input_ids'].to(DEVICE)
    without_cot_prefix = ids_without_cot['input_ids'].to(DEVICE)
    full_with_cot = concat_answer(with_cot_prefix, answer_ids)
    full_without_cot = concat_answer(without_cot_prefix, answer_ids)

    cached_with_cot = cache_model_states(full_with_cot)
    cached_without_cot = cache_model_states(full_without_cot)

    hidden_states_with = cached_with_cot['hidden_states']
    hidden_states_without = cached_without_cot['hidden_states']

    if layer_spans is None:
        layer_spans = sample_layer_spans(span_samples, num_layers, max_span_size=max_span_size)

    seq_length = full_with_cot.shape[1]
    non_cot_positions = [i for i in range(seq_length) if i not in cot_positions]

    result = run_cmi_analysis(
        full_with_cot,
        answer_ids,
        hidden_states_with,
        hidden_states_without,
        cot_positions,
        non_cot_positions,
        layer_spans,
        control_samples,
        patch_mode="source",
    )

    no_cot_effect = (len(result['intervention_records']) > 0 and all(r['cmi_score'] == 0.0 for r in result['intervention_records']))
    result['no_cot_effect'] = no_cot_effect
    return result

def create_visualizations(results_data):
    if not results_data['records']:
        print("No data to visualize - all samples failed processing.")
        return

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(22, 16))

    samples = list(range(1, len(results_data['records']) + 1))
    cmi_scores = [r['CMI'] for r in results_data['records']]
    bypass_scores = [r['Bypass'] for r in results_data['records']]
    placebo_scores = [r['Placebo_CMI'] for r in results_data['records']]
    validity_scores = [r['Validity_Index'] for r in results_data['records']]
    sequentiality_scores = [r['Sequentiality_Index'] for r in results_data['records']]
    baseline_logps = [abs(r['baseline_logp']) for r in results_data['records']]

    sensitivities = [r['sensitivity']['relative_sensitivity'] for r in results_data['records']]
    is_robust = [r['sensitivity']['is_robust'] for r in results_data['records']]

    if not baseline_logps or max(baseline_logps) == 0:
        baseline_logps = [1.0] * len(samples)

    # 1. CMI by sample
    ax1 = plt.subplot(2, 4, 1)
    plt.bar(samples, cmi_scores, color='steelblue', alpha=0.8, edgecolor='black')
    plt.xlabel('Sample')
    plt.ylabel('CMI Score')
    plt.title('CoT-Mediated Influence (CMI)')
    plt.grid(True, alpha=0.3)

    # 2. Bypass by sample
    ax2 = plt.subplot(2, 4, 2)
    plt.bar(samples, bypass_scores, color='darkorange', alpha=0.8, edgecolor='black')
    plt.xlabel('Sample')
    plt.ylabel('Bypass Score')
    plt.title('Bypass Score (Higher = More Bypass)')
    plt.grid(True, alpha=0.3)

    # 3. Placebo vs Base CMI
    ax3 = plt.subplot(2, 4, 3)
    plt.scatter(cmi_scores, placebo_scores, s=100, alpha=0.7, edgecolors='black')
    plt.xlabel('Base CMI')
    plt.ylabel('Placebo CMI')
    plt.title('Placebo CoT Test')
    plt.grid(True, alpha=0.3)

    # 4. Validity + Sequentiality
    ax4 = plt.subplot(2, 4, 4)
    width = 0.35
    x_pos = np.arange(len(samples))
    plt.bar(x_pos - width / 2, validity_scores, width, label='Validity', color='seagreen', alpha=0.8, edgecolor='black')
    plt.bar(x_pos + width / 2, sequentiality_scores, width, label='Sequentiality', color='slateblue', alpha=0.8, edgecolor='black')
    plt.xlabel('Sample')
    plt.ylabel('Score')
    plt.title('Validity & Sequentiality')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Density curves
    ax5 = fig.add_subplot(2, 4, 5)
    for i, record in enumerate(results_data['records']):
        curve = record.get('Density_Curve', [])
        if not curve:
            continue
        xs = [c['keep_frac'] for c in curve]
        ys = [c['cmi'] for c in curve]
        plt.plot(xs, ys, marker='o', alpha=0.6, label=f"S{i+1}")
    plt.xlabel('CoT Keep Fraction')
    plt.ylabel('CMI')
    plt.title('CoT Density Curves')
    plt.grid(True, alpha=0.3)
    if len(results_data['records']) <= 6:
        plt.legend()

    # 6. Sensitivity Analysis
    ax6 = plt.subplot(2, 4, 6)
    colors_robust = ['green' if r else 'red' for r in is_robust]
    plt.bar(samples, sensitivities, color=colors_robust, alpha=0.7, edgecolor='black')
    plt.axhline(y=0.15, color='orange', linestyle='--', linewidth=2, label='Robustness Threshold (15%)')
    plt.xlabel('Sample')
    plt.ylabel('Relative Sensitivity')
    plt.title('Token Boundary Sensitivity (CMI)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. 3D scatter: CMI vs Bypass vs baseline logp
    ax7 = fig.add_subplot(2, 4, 7, projection='3d')
    scatter_3d = ax7.scatter(cmi_scores, bypass_scores, baseline_logps,
                           c=samples, s=100, alpha=0.8, cmap='plasma', edgecolors='black')
    ax7.set_xlabel('CMI Score')
    ax7.set_ylabel('Bypass Score')
    ax7.set_zlabel('Baseline LogP')
    ax7.set_title('3D: CMI vs Bypass vs LogP')
    plt.colorbar(scatter_3d, ax=ax7, shrink=0.8, label='Sample')

    # 8. Distribution comparison
    ax8 = plt.subplot(2, 4, 8)
    if cmi_scores and bypass_scores:
        plt.hist(cmi_scores, bins=min(10, len(cmi_scores)), alpha=0.7, label='CMI', color='blue', density=True)
        plt.hist(bypass_scores, bins=min(10, len(bypass_scores)), alpha=0.7, label='Bypass', color='orange', density=True)
    plt.xlabel('Score Value')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    try:
        plt.show()
    except Exception as exc:
        print(f"Plot display failed ({exc}); saved to {PLOT_PATH}")

def run_analysis(num_samples=10, control_samples=8, span_samples=8, max_span_size=3):
    print("DATASET_MODE:", DATASET_MODE)
    if DATASET_MODE == "hf":
        dataset = load_dataset('squad', split=f'validation[:{num_samples}]')
        dataset_iter = (
            {
                "question": ex["question"],
                "context": ex.get("context", ""),
                "answer": ex["answers"]["text"][0] if ex["answers"]["text"] else "",
            }
            for ex in dataset
        )
    else:
        dataset_iter = build_synthetic_math_dataset(num_samples)
    analysis_records = []
    cot_trigger = "Let's think step by step and compute carefully."

    for idx, example in enumerate(dataset_iter):
        question = example["question"]
        context = example.get("context", "")
        answer = example.get("answer", "")
        if not question:
            continue
        context_block = ("Context: " + context + "\n") if context else ""
        cot_prompt = (
            context_block
            + "Question: " + question + "\n"
            + "Use the context above explicitly in your reasoning.\n"
            + cot_trigger
        )
        generated_cot = generate_cot_text(cot_prompt, context_hint=context)
        if generated_cot:
            cot_span = cot_trigger + " " + generated_cot
        else:
            cot_span = cot_trigger
        prompt_with_cot = context_block + "Question: " + question + "\n" + cot_span + "\nFinal answer:"
        prompt_without_cot = context_block + "Question: " + question

        # Tokenize both prompts
        tokenized_with_cot = tokenizer(prompt_with_cot, return_tensors='pt')
        tokenized_without_cot = tokenizer(prompt_without_cot, return_tensors='pt')
        tokenized_with_cot = to_device(tokenized_with_cot)
        tokenized_without_cot = to_device(tokenized_without_cot)

        # Find exact CoT token positions using tokenization alignment
        cot_token_positions = find_cot_span_positions(prompt_with_cot, cot_span, tokenizer)
        if not cot_token_positions:
            cot_token_positions = find_cot_token_positions(prompt_with_cot, cot_trigger, tokenizer)

        print(f"\nSample {idx+1}:")
        print(f"  Total tokens in prompt: {tokenized_with_cot['input_ids'].shape[1]}")
        print(f"  CoT token positions: {cot_token_positions}")
        print(f"  CoT tokens: {tokenizer.decode(tokenized_with_cot['input_ids'][0, cot_token_positions])}")

        generated_answer_raw = generate_answer_text(prompt_with_cot, max_new_tokens=10)
        generated_answer = clean_generated_answer(generated_answer_raw)
        candidate_answers = []
        if generated_answer:
            candidate_answers.append(("generated", generated_answer))
        if answer:
            candidate_answers.append(("gold", answer))

        if not candidate_answers:
            continue

        if answer and is_bad_generated(generated_answer_raw, answer):
            primary_label = "gold"
        else:
            primary_label = "generated" if generated_answer else candidate_answers[0][0]

        if primary_label == "gold" and answer:
            primary_answer_text = answer
        else:
            primary_answer_text = generated_answer or candidate_answers[0][1]

        try:
            fixed_spans = build_fixed_layer_spans(len(get_model_layers(model)))

            # Perform sensitivity analysis first (use the same layer spans for consistency).
            sensitivity_result = perform_sensitivity_analysis(
                tokenized_with_cot['input_ids'],
                tokenizer(primary_answer_text, add_special_tokens=False)['input_ids'],
                cot_token_positions,
                num_layers=len(get_model_layers(model)),
                control_samples=control_samples,
                span_samples=span_samples,
                max_span_size=max_span_size,
                layer_spans=fixed_spans
            )

            print(f"  Sensitivity Analysis:")
            print(f"    Base CMI: {sensitivity_result['base_cmi']:.4f}")
            print(f"    Expanded CMI: {sensitivity_result['expanded_cmi']:.4f}")
            print(f"    Shrunk CMI: {sensitivity_result['shrunk_cmi']:.4f}")
            print(f"    Max Deviation: {sensitivity_result['max_deviation']:.4f}")
            print(f"    Relative Sensitivity: {sensitivity_result['relative_sensitivity']:.4f}")
            print(f"    Is Robust: {sensitivity_result['is_robust']}")

            # Run main intervention analysis with validated positions
            layer_spans = fixed_spans + sample_layer_spans(
                max(0, span_samples - len(fixed_spans)),
                len(get_model_layers(model)),
                max_span_size=max_span_size
            )
            metrics_by_answer = {}

            for label, ans_text in candidate_answers:
                answer_token_ids = tokenizer(ans_text, add_special_tokens=False)['input_ids']
                if not answer_token_ids:
                    continue

                full_with_cot = concat_answer(tokenized_with_cot['input_ids'], answer_token_ids)
                full_without_cot = concat_answer(tokenized_without_cot['input_ids'], answer_token_ids)
                cached_with = cache_model_states(full_with_cot)
                cached_without = cache_model_states(full_without_cot)
                seq_length = full_with_cot.shape[1]
                non_cot_positions = [i for i in range(seq_length) if i not in cot_token_positions]

                analysis_result = run_intervention_analysis(
                    tokenized_with_cot, tokenized_without_cot, answer_token_ids, cot_token_positions,
                    num_layers=len(get_model_layers(model)),
                    control_samples=control_samples,
                    span_samples=span_samples,
                    max_span_size=max_span_size,
                    layer_spans=layer_spans
                )

                cmi_score = analysis_result['cmi_mean']
                bypass_score = analysis_result['bypass_mean']
                low_signal = cmi_score < CMI_BASE_FLOOR

                placebo_result = run_cmi_analysis(
                    full_with_cot,
                    answer_token_ids,
                    cached_with['hidden_states'],
                    cached_without['hidden_states'],
                    cot_token_positions,
                    non_cot_positions,
                    layer_spans,
                    control_samples,
                    patch_mode="noise"
                )
                placebo_cmi = placebo_result['cmi_mean']
                if low_signal:
                    validity_index = 0.0
                else:
                    validity_index = max(0.0, cmi_score - placebo_cmi) / max(cmi_score, CMI_BASE_FLOOR)

                sparsity_levels = [0.8, 0.6, 0.4]
                density_curve = []
                for keep_frac in sparsity_levels:
                    k = max(1, int(len(cot_token_positions) * keep_frac))
                    sparse_positions = random.sample(cot_token_positions, k=k)
                    sparse_non_cot = [i for i in range(seq_length) if i not in sparse_positions]
                    sparse_result = run_cmi_analysis(
                        full_with_cot,
                        answer_token_ids,
                        cached_with['hidden_states'],
                        cached_without['hidden_states'],
                        sparse_positions,
                        sparse_non_cot,
                        layer_spans,
                        control_samples,
                        patch_mode="source"
                    )
                    density_curve.append({'keep_frac': keep_frac, 'cmi': sparse_result['cmi_mean']})

                cmi_40 = next((d['cmi'] for d in density_curve if d['keep_frac'] == 0.4), 0.0)
                if low_signal:
                    density_index = 0.0
                else:
                    density_index = 1.0 - min(1.0, cmi_40 / max(cmi_score, CMI_BASE_FLOOR))

                permuted_positions = cot_token_positions.copy()
                random.shuffle(permuted_positions)
                shuffle_result = run_cmi_analysis(
                    full_with_cot,
                    answer_token_ids,
                    cached_with['hidden_states'],
                    cached_without['hidden_states'],
                    cot_token_positions,
                    non_cot_positions,
                    layer_spans,
                    control_samples,
                    patch_mode="shuffle",
                    permute_positions=permuted_positions
                )
                if low_signal:
                    sequentiality_score = 0.0
                else:
                    sequentiality_score = max(0.0, cmi_score - shuffle_result['cmi_mean']) / max(cmi_score, CMI_BASE_FLOOR)

                metrics_by_answer[label] = {
                    'answer_text': ans_text,
                    'baseline_logp': analysis_result['baseline_logp'],
                    'CMI': cmi_score,
                    'Bypass': bypass_score,
                    'Placebo_CMI': placebo_cmi,
                    'Validity_Index': validity_index,
                    'Density_Curve': density_curve,
                    'Density_Index': density_index,
                    'Sequentiality_Index': sequentiality_score,
                    'Shuffled_CMI': shuffle_result['cmi_mean'],
                    'no_cot_effect': analysis_result['no_cot_effect'],
                    'low_signal': low_signal,
                    'intervention_records': analysis_result['intervention_records'],
                }

            if not metrics_by_answer:
                continue

            primary_metrics = metrics_by_answer.get(primary_label) or next(iter(metrics_by_answer.values()))

            record = {
                'context': context,
                'question': question,
                'answer': answer,
                'generated_answer_raw': generated_answer_raw,
                'generated_answer': generated_answer,
                'primary_label': primary_label,
                'baseline_logp': primary_metrics['baseline_logp'],
                'CMI': primary_metrics['CMI'],
                'Bypass': primary_metrics['Bypass'],
                'Placebo_CMI': primary_metrics['Placebo_CMI'],
                'Validity_Index': primary_metrics['Validity_Index'],
                'Density_Curve': primary_metrics['Density_Curve'],
                'Density_Index': primary_metrics['Density_Index'],
                'Sequentiality_Index': primary_metrics['Sequentiality_Index'],
                'Shuffled_CMI': primary_metrics['Shuffled_CMI'],
                'no_cot_effect': primary_metrics['no_cot_effect'],
                'low_signal': primary_metrics.get('low_signal', False),
                'intervention_records': primary_metrics['intervention_records'],
                'metrics_by_answer': metrics_by_answer,
                'cot_positions': cot_token_positions,
                'sensitivity': sensitivity_result
            }
            analysis_records.append(record)

        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not analysis_records:
        return {'CMI': 0.0, 'Bypass': 0.0, 'records': []}

    cmi_scores_by_sample = [r['CMI'] for r in analysis_records]
    bypass_scores_by_sample = [r['Bypass'] for r in analysis_records]
    overall_cmi = float(np.mean(cmi_scores_by_sample))
    overall_bypass = float(np.mean(bypass_scores_by_sample))

    # Calculate robustness statistics
    robust_count = sum(1 for r in analysis_records if r['sensitivity']['is_robust'])
    robustness_rate = robust_count / len(analysis_records) if analysis_records else 0.0

    results = {
        'CMI': overall_cmi,
        'Bypass': overall_bypass,
        'robustness_rate': robustness_rate,
        'records': analysis_records
    }

    print(f"\n{'='*60}")
    print(f"Analysis Complete:")
    print(f"{'='*60}")
    print(f"Samples processed: {len(analysis_records)}")
    print(f"Mean CMI: {overall_cmi:.4f}")
    print(f"Mean Bypass: {overall_bypass:.4f}")
    print(f"Robustness Rate: {robustness_rate:.2%} ({robust_count}/{len(analysis_records)} samples)")
    print(f"{'='*60}\n")

    return results

if __name__ == '__main__':
    results = run_analysis(num_samples=10, control_samples=8, span_samples=8, max_span_size=3)

    print("\nCreating visualizations...")
    create_visualizations(results)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    with open(RESULTS_JSONL, "w", encoding="utf-8") as f:
        for record in results["records"]:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print("\nPer-sample summary:")
    for i, record in enumerate(results['records']):
        print(f"\nSample {i+1}:")
        print(f"  Question: {record['question'][:100]}...")
        print(f"  Answer: {record['answer'][:50]}...")
        print(f"  Generated Answer (raw): {record['generated_answer_raw'][:50]}...")
        print(f"  Generated Answer: {record['generated_answer'][:50]}...")
        print(f"  Primary Label: {record['primary_label']}")
        print(f"  CoT Positions: {record['cot_positions']}")
        print(f"  Baseline LogP: {record['baseline_logp']:.6f}")
        print(f"  CMI: {record['CMI']:.3f}")
        print(f"  Bypass: {record['Bypass']:.3f}")
        print(f"  Placebo CMI: {record['Placebo_CMI']:.3f}")
        print(f"  Validity Index: {record['Validity_Index']:.3f}")
        print(f"  Sequentiality Index: {record['Sequentiality_Index']:.3f}")
        print(f"  Density Index: {record['Density_Index']:.3f}")
        print(f"  No-CoT-Effect: {record['no_cot_effect']}")
        print(f"  Low Signal: {record.get('low_signal', False)}")
        print(f"  Sensitivity: {record['sensitivity']['relative_sensitivity']:.3f}")
        print(f"  Robust: {record['sensitivity']['is_robust']}")
