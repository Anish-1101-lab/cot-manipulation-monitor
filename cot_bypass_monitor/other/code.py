import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import numpy as np
import contextlib
from copy import deepcopy
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "microsoft/DialoGPT-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seeds()

def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}

def concat_answer(prefix_ids, answer_ids):
    a = torch.tensor(answer_ids, device=DEVICE).unsqueeze(0)
    return torch.cat([prefix_ids, a], dim=1)

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

def perform_sensitivity_analysis(prefix_ids, answer_ids, base_cot_positions, num_layers, control_samples=8, span_samples=12, max_span_size=3):
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
        max_span_size=max_span_size
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
        max_span_size=max_span_size
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
            max_span_size=max_span_size
        )
        shrunk_cmi = shrunk_result['cmi_mean']
    else:
        shrunk_cmi = base_cmi

    # Calculate sensitivity metrics
    max_deviation = max(
        abs(expanded_cmi - base_cmi),
        abs(shrunk_cmi - base_cmi)
    )

    relative_sensitivity = max_deviation / (base_cmi + 1e-12)

    return {
        'base_cmi': base_cmi,
        'expanded_cmi': expanded_cmi,
        'shrunk_cmi': shrunk_cmi,
        'max_deviation': max_deviation,
        'relative_sensitivity': relative_sensitivity,
        'is_robust': relative_sensitivity < 0.15  # Less than 15% change is considered robust
    }

@torch.no_grad()
def compute_answer_logprob(input_ids, answer_ids):
    out = model(input_ids=input_ids, output_attentions=False, output_hidden_states=False)
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
    out = model(input_ids=input_ids, output_attentions=True, output_hidden_states=True)

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
            if layer_idx >= len(model.transformer.h):
                continue
            layer = model.transformer.h[layer_idx]
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
            if layer_idx < len(model.transformer.h):
                model.transformer.h[layer_idx].forward = original_forward

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

def align_source_patch(source_states, target_states, positions):
    if source_states is None or target_states is None:
        return target_states
    if source_states.ndim != 3 or target_states.ndim != 3:
        return target_states

    patch = target_states.clone()
    src_len = source_states.shape[1]
    if src_len == 0:
        return patch

    for pos in positions:
        if 0 <= pos < patch.shape[1]:
            src_pos = min(pos, src_len - 1)
            patch[:, pos, :] = source_states[:, src_pos, :]
    return patch

def sample_layer_spans(num_samples, num_layers, max_span_size=3):
    spans = []
    for _ in range(num_samples):
        span_size = random.randint(1, max_span_size)
        start_layer = random.randint(0, num_layers - span_size)
        spans.append(list(range(start_layer, start_layer + span_size)))
    return spans

def run_intervention_analysis(ids_with_cot, ids_without_cot, answer_ids, cot_positions,
                            num_layers=None, control_samples=10, span_samples=12, max_span_size=3):
    if num_layers is None:
        num_layers = len(model.transformer.h)

    with_cot_prefix = ids_with_cot['input_ids'].to(DEVICE)
    without_cot_prefix = ids_without_cot['input_ids'].to(DEVICE)
    full_with_cot = concat_answer(with_cot_prefix, answer_ids)
    full_without_cot = concat_answer(without_cot_prefix, answer_ids)

    cached_with_cot = cache_model_states(full_with_cot)
    cached_without_cot = cache_model_states(full_without_cot)

    baseline_logp = compute_answer_logprob(full_with_cot, answer_ids)

    hidden_states_with = cached_with_cot['hidden_states']
    hidden_states_without = cached_without_cot['hidden_states']

    layer_spans = sample_layer_spans(span_samples, num_layers, max_span_size=max_span_size)

    intervention_records = []

    seq_length = full_with_cot.shape[1]
    non_cot_positions = [i for i in range(seq_length) if i not in cot_positions]

    for span in layer_spans:
        # CoT replacement with no-CoT states (causal test)
        layer_specs = {}
        for layer_idx in span:
            source_states = hidden_states_without[layer_idx + 1]
            target_states = hidden_states_with[layer_idx + 1]
            aligned_patch = align_source_patch(source_states, target_states, cot_positions)
            layer_specs[layer_idx] = {
                'mode': 'source',
                'patch_tensor': aligned_patch,
                'mask_positions': cot_positions
            }

        with patch_model_layers(model, layer_specs):
            cot_logp = compute_answer_logprob(full_with_cot, answer_ids)
        cot_drop = max(0.0, baseline_logp - cot_logp)

        # Control replacement on non-CoT positions
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
                aligned_patch = align_source_patch(source_states, target_states, control_positions)
                control_specs[layer_idx] = {
                    'mode': 'source',
                    'patch_tensor': aligned_patch,
                    'mask_positions': control_positions
                }

            with patch_model_layers(model, control_specs):
                control_logp = compute_answer_logprob(full_with_cot, answer_ids)
            control_drops.append(max(0.0, baseline_logp - control_logp))

        control_drop = float(np.mean(control_drops))
        cmi_raw = cot_drop - control_drop
        denom = cot_drop + control_drop + 1e-12
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

def create_visualizations(results_data):
    if not results_data['records']:
        print("No data to visualize - all samples failed processing.")
        return

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(22, 16))

    samples = list(range(1, len(results_data['records']) + 1))
    cmi_scores = [r['CMI'] for r in results_data['records']]
    bypass_scores = [r['Bypass'] for r in results_data['records']]
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

    # 3. CMI vs Bypass scatter
    ax3 = plt.subplot(2, 4, 3)
    colors = plt.cm.viridis(np.array(baseline_logps) / max(baseline_logps))
    scatter = plt.scatter(cmi_scores, bypass_scores, c=colors, s=100, alpha=0.7, edgecolors='black')
    plt.xlabel('CMI Score')
    plt.ylabel('Bypass Score')
    plt.title('CMI vs Bypass')
    plt.colorbar(scatter, label='Baseline LogP (normalized)')
    plt.grid(True, alpha=0.3)

    # 4. Sensitivity Analysis
    ax4 = plt.subplot(2, 4, 4)
    colors_robust = ['green' if r else 'red' for r in is_robust]
    plt.bar(samples, sensitivities, color=colors_robust, alpha=0.7, edgecolor='black')
    plt.axhline(y=0.15, color='orange', linestyle='--', linewidth=2, label='Robustness Threshold (15%)')
    plt.xlabel('Sample')
    plt.ylabel('Relative Sensitivity')
    plt.title('Token Boundary Sensitivity (CMI)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 3D scatter: CMI vs Bypass vs baseline logp
    ax5 = fig.add_subplot(2, 4, 5, projection='3d')
    scatter_3d = ax5.scatter(cmi_scores, bypass_scores, baseline_logps,
                           c=samples, s=100, alpha=0.8, cmap='plasma', edgecolors='black')
    ax5.set_xlabel('CMI Score')
    ax5.set_ylabel('Bypass Score')
    ax5.set_zlabel('Baseline LogP')
    ax5.set_title('3D: CMI vs Bypass vs LogP')
    plt.colorbar(scatter_3d, ax=ax5, shrink=0.8, label='Sample')

    # 6. Distribution comparison
    ax6 = plt.subplot(2, 4, 6)
    if cmi_scores and bypass_scores:
        plt.hist(cmi_scores, bins=min(10, len(cmi_scores)), alpha=0.7, label='CMI', color='blue', density=True)
        plt.hist(bypass_scores, bins=min(10, len(bypass_scores)), alpha=0.7, label='Bypass', color='orange', density=True)
    plt.xlabel('Score Value')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. CMI vs baseline logp
    ax7 = plt.subplot(2, 4, 7)
    plt.scatter(baseline_logps, cmi_scores, s=100, alpha=0.7, edgecolors='black')
    plt.xlabel('Baseline LogP')
    plt.ylabel('CMI Score')
    plt.title('CMI vs Baseline LogP')
    plt.grid(True, alpha=0.3)

    # 8. Bypass vs baseline logp
    ax8 = plt.subplot(2, 4, 8)
    plt.scatter(baseline_logps, bypass_scores, s=100, alpha=0.7, edgecolors='black')
    plt.xlabel('Baseline LogP')
    plt.ylabel('Bypass Score')
    plt.title('Bypass vs Baseline LogP')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_analysis(num_samples=4, control_samples=8, span_samples=12, max_span_size=3):
    dataset = load_dataset('squad', split=f'validation[:{num_samples}]')
    analysis_records = []
    cot_trigger = "Let's think step by step."

    for idx, example in enumerate(dataset):
        question = example['question']
        answers = example['answers']['text']
        if not answers:
            continue

        answer = answers[0]
        prompt_with_cot = "Question: " + question + "\n" + cot_trigger
        prompt_without_cot = "Question: " + question

        # Tokenize both prompts
        tokenized_with_cot = tokenizer(prompt_with_cot, return_tensors='pt')
        tokenized_without_cot = tokenizer(prompt_without_cot, return_tensors='pt')
        tokenized_with_cot = to_device(tokenized_with_cot)
        tokenized_without_cot = to_device(tokenized_without_cot)

        # Find exact CoT token positions using tokenization alignment
        cot_token_positions = find_cot_token_positions(prompt_with_cot, cot_trigger, tokenizer)

        print(f"\nSample {idx+1}:")
        print(f"  Total tokens in prompt: {tokenized_with_cot['input_ids'].shape[1]}")
        print(f"  CoT token positions: {cot_token_positions}")
        print(f"  CoT tokens: {tokenizer.decode(tokenized_with_cot['input_ids'][0, cot_token_positions])}")

        answer_token_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        if not answer_token_ids:
            continue

        try:
            # Perform sensitivity analysis first
            sensitivity_result = perform_sensitivity_analysis(
                tokenized_with_cot['input_ids'],
                answer_token_ids,
                cot_token_positions,
                num_layers=len(model.transformer.h),
                control_samples=control_samples,
                span_samples=span_samples,
                max_span_size=max_span_size
            )

            print(f"  Sensitivity Analysis:")
            print(f"    Base CMI: {sensitivity_result['base_cmi']:.4f}")
            print(f"    Expanded CMI: {sensitivity_result['expanded_cmi']:.4f}")
            print(f"    Shrunk CMI: {sensitivity_result['shrunk_cmi']:.4f}")
            print(f"    Max Deviation: {sensitivity_result['max_deviation']:.4f}")
            print(f"    Relative Sensitivity: {sensitivity_result['relative_sensitivity']:.4f}")
            print(f"    Is Robust: {sensitivity_result['is_robust']}")

            # Run main intervention analysis with validated positions
            analysis_result = run_intervention_analysis(
                tokenized_with_cot, tokenized_without_cot, answer_token_ids, cot_token_positions,
                num_layers=len(model.transformer.h),
                control_samples=control_samples,
                span_samples=span_samples,
                max_span_size=max_span_size
            )

            cmi_score = analysis_result['cmi_mean']
            bypass_score = analysis_result['bypass_mean']

            record = {
                'question': question,
                'answer': answer,
                'baseline_logp': analysis_result['baseline_logp'],
                'CMI': cmi_score,
                'Bypass': bypass_score,
                'intervention_records': analysis_result['intervention_records'],
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
    results = run_analysis(num_samples=6, control_samples=8, span_samples=12, max_span_size=3)

    print("\nCreating visualizations...")
    create_visualizations(results)

    print("\nPer-sample summary:")
    for i, record in enumerate(results['records']):
        print(f"\nSample {i+1}:")
        print(f"  Question: {record['question'][:100]}...")
        print(f"  Answer: {record['answer'][:50]}...")
        print(f"  CoT Positions: {record['cot_positions']}")
        print(f"  Baseline LogP: {record['baseline_logp']:.6f}")
        print(f"  CMI: {record['CMI']:.3f}")
        print(f"  Bypass: {record['Bypass']:.3f}")
        print(f"  Sensitivity: {record['sensitivity']['relative_sensitivity']:.3f}")
        print(f"  Robust: {record['sensitivity']['is_robust']}")