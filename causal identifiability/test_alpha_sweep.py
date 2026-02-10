"""
Alpha sweep test for formality trait non-identifiability.

Tests steering strength α ∈ [0.0, 0.5, 1.0, 2.0] to verify that
orthogonal components scale identically with v (evidence for non-identifiability).

Usage:
    python test_alpha_sweep.py --alphas 0.0 0.5 1.0 2.0 --n_seeds 10
"""

import json
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from persona_vector_experiment import PersonaVectorExperiment

# ACL paper formatting: set before creating figures
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
})


def test_alpha_sweep(model_name: str, trait: str = "formality", 
                    alphas: list = None, n_seeds: int = 10):
    """
    Test steering strength α sweep for formality trait.
    
    If non-identifiable, v and v+v_perp should scale similarly across alphas.
    
    Args:
        model_name: HuggingFace model name
        trait: Semantic trait (default: 'formality')
        alphas: List of steering strength multipliers
        n_seeds: Number of random orthogonal seeds per alpha
    
    Returns:
        Dictionary with alpha sweep results
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 2.0]
    
    print(f"\nALPHA SWEEP TEST FOR NON-IDENTIFIABILITY")
    print(f"Model: {model_name}")
    print(f"Trait: {trait}")
    print(f"Alpha values: {alphas}")
    print(f"Seeds per alpha: {n_seeds}\n")
    
    # Initialize
    experiment = PersonaVectorExperiment(model_name)
    
    # Test prompts
    test_prompts = [
        "Write about your thoughts on",
        "Describe your experience with",
        "Share your opinion about",
        "Explain your view on",
        "Discuss your perspective about",
        "Tell me what you think of",
        "Express your feelings about",
        "Give me your take on",
        "Talk about your experience with",
        "What are your thoughts on",
        "How do you feel about",
        "What's your opinion on",
        "Can you describe",
        "Please explain",
        "Share your views about",
        "Discuss the topic of",
        "What do you make of",
        "Tell me about",
        "Your perspective on",
        "Comment on the subject of"
    ]
    
    # Step 1: Extract steering vector
    print(f"Step 1: Extracting {trait} steering vector...")
    v = experiment.extract_steering_vector(trait, n_pairs=50)
    v_norm = torch.norm(v).item()
    print(f"  [OK] Steering vector extracted")
    print(f"  [OK] Shape: {v.shape}")
    print(f"  [OK] Norm: {v_norm:.4f}\n")
    
    # Step 2: Alpha sweep
    print(f"Step 2: Testing alpha sweep...")
    alpha_results = {}
    
    for alpha in alphas:
        print(f"\n  Testing alpha = {alpha}...")
        alpha_data = {
            'alpha': alpha,
            'seeds': []
        }
        
        for seed_idx in range(n_seeds):
            # Create random orthogonal vector
            random_vec = torch.randn_like(v)
            v_perp = random_vec - (random_vec @ v) / (v @ v) * v
            v_perp = v_perp / torch.norm(v_perp) * v_norm
            
            # Test vectors: v, v+v_perp
            v_steered = v * alpha
            v_plus_perp_steered = (v + v_perp) * alpha
            
            # Generate with v
            scores_v = []
            for prompt in test_prompts[:10]:
                try:
                    texts = experiment.generate_with_steering(
                        prompt, v_steered, alpha=1.0, max_new_tokens=40, num_return_sequences=3
                    )
                    for text in texts:
                        score = experiment.compute_semantic_score(text, trait)
                        scores_v.append(score)
                except:
                    continue
            
            # Generate with v + v_perp
            scores_v_perp = []
            for prompt in test_prompts[:10]:
                try:
                    texts = experiment.generate_with_steering(
                        prompt, v_plus_perp_steered, alpha=1.0, max_new_tokens=40, num_return_sequences=3
                    )
                    for text in texts:
                        score = experiment.compute_semantic_score(text, trait)
                        scores_v_perp.append(score)
                except:
                    continue
            
            mean_v = np.mean(scores_v) if scores_v else 0.0
            mean_perp = np.mean(scores_v_perp) if scores_v_perp else 0.0
            
            # Compute effect
            effect_diff = abs(mean_perp - mean_v)
            
            alpha_data['seeds'].append({
                'seed': seed_idx,
                'mean_v': float(mean_v),
                'mean_v_perp': float(mean_perp),
                'effect_diff': float(effect_diff),
                'samples_v': len(scores_v),
                'samples_perp': len(scores_v_perp)
            })
            
            print(f"    Seed {seed_idx+1}/{n_seeds}: v={mean_v:.4f}, v+perp={mean_perp:.4f}, diff={effect_diff:.4f}")
        
        # Compute alpha statistics
        effect_diffs = [s['effect_diff'] for s in alpha_data['seeds']]
        alpha_data['summary'] = {
            'mean_effect_diff': float(np.mean(effect_diffs)),
            'std_effect_diff': float(np.std(effect_diffs)),
            'min_effect_diff': float(np.min(effect_diffs)),
            'max_effect_diff': float(np.max(effect_diffs))
        }
        
        alpha_results[f"alpha_{alpha}"] = alpha_data
        print(f"    Summary: mean_diff={alpha_data['summary']['mean_effect_diff']:.4f} ± {alpha_data['summary']['std_effect_diff']:.4f}")
    
    # Overall summary
    print(f"\nALPHA SWEEP SUMMARY")
    for alpha_key, data in alpha_results.items():
        alpha = data['alpha']
        summary = data['summary']
        print(f"  α={alpha}: effect_diff={summary['mean_effect_diff']:.4f} ± {summary['std_effect_diff']:.4f}")
    
    # Compute overall statistics
    all_diffs = [data['summary']['mean_effect_diff'] for data in alpha_results.values()]
    
    return {
        'model': model_name,
        'trait': trait,
        'alphas': alphas,
        'n_seeds': n_seeds,
        'alpha_results': alpha_results,
        'overall_summary': {
            'mean_effect_across_alphas': float(np.mean(all_diffs)),
            'std_effect_across_alphas': float(np.std(all_diffs))
        }
    }


def plot_alpha_sweep(results: dict, output_dir: str = "figures", model_suffix: str = ""):
    """
    Plot α-sweep response curves for v vs v+v_perp.

    Args:
        results: Results dictionary from test_alpha_sweep
        output_dir: Directory to save the plot
        model_suffix: Model suffix for filename (e.g., 'qwen', 'llama')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    alphas = results['alphas']
    alpha_results = results['alpha_results']
    trait = results['trait']
    
    # Collect means and stds for each alpha
    means_v = []
    stds_v = []
    means_perp = []
    stds_perp = []
    
    for alpha_key in sorted(alpha_results.keys()):
        data = alpha_results[alpha_key]
        
        # Extract per-seed scores
        scores_v = []
        scores_perp = []
        
        for seed_data in data['seeds']:
            scores_v.append(seed_data['mean_v'])
            scores_perp.append(seed_data['mean_v_perp'])
        
        means_v.append(np.mean(scores_v))
        stds_v.append(np.std(scores_v))
        means_perp.append(np.mean(scores_perp))
        stds_perp.append(np.std(scores_perp))
    
    # Create plot: single-column ACL format (3.25 inches wide)
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    
    # Plot lines with error bands
    alphas_array = np.array(alphas)
    
    # v line with band
    ax.plot(alphas_array, means_v, 'o-', linewidth=1.5, markersize=5,
            label='v', color='#1f77b4')
    ax.fill_between(alphas_array, 
                     np.array(means_v) - np.array(stds_v),
                     np.array(means_v) + np.array(stds_v),
                     alpha=0.2, color='#1f77b4')
    
    # v+v_perp line with band
    ax.plot(alphas_array, means_perp, 's-', linewidth=1.5, markersize=5,
            label='v + v_perp', color='#ff7f0e')
    ax.fill_between(alphas_array,
                     np.array(means_perp) - np.array(stds_perp),
                     np.array(means_perp) + np.array(stds_perp),
                     alpha=0.2, color='#ff7f0e')
    
    # Labels and formatting
    ax.set_xlabel('Steering Strength (α)', fontsize=8)
    ax.set_ylabel('Formality Score', fontsize=8)
    ax.set_title(f'α-Sweep: {trait.capitalize()} Response', fontsize=9)
    ax.set_xticks(alphas_array)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=7, loc='best', framealpha=0.95)
    
    plt.tight_layout()

    # Save plot with model suffix
    suffix = f"_{model_suffix}" if model_suffix else ""
    output_pdf = f"{output_dir}/alpha_sweep_response_curves{suffix}.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PDF (paper-ready): {output_pdf}")

    # Also save as PNG for quick review
    output_file = output_pdf.replace('.pdf', '.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PNG (preview): {output_file}")

    plt.close()


def main():
    """Run alpha sweep test with command line arguments."""
    parser = argparse.ArgumentParser(description="Alpha sweep test for formality trait")
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.0, 0.5, 1.0, 2.0],
                       help='Alpha values to test')
    parser.add_argument('--n_seeds', type=int, default=10,
                       help='Number of seeds per alpha')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                       help='Model name')
    
    args = parser.parse_args()
    model_name = args.model
    alphas = sorted(args.alphas)
    n_seeds = args.n_seeds
    
    print(f"Alpha sweep configuration:")
    print(f"  Model: {model_name}")
    print(f"  Alphas: {alphas}")
    print(f"  Seeds: {n_seeds}")
    
    try:
        results = test_alpha_sweep(model_name, trait='formality', alphas=alphas, n_seeds=n_seeds)
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    import os
    os.makedirs("results", exist_ok=True)

    # Determine model suffix for filename
    if "qwen" in model_name.lower():
        model_suffix = "qwen"
    elif "llama" in model_name.lower():
        model_suffix = "llama"
    else:
        # Use a sanitized version of model name as fallback
        model_suffix = model_name.split('/')[-1].lower().replace('-', '_')

    output_file = f"results/alpha_sweep_results_{model_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] Results saved to: {output_file}")

    # Generate plot
    try:
        plot_alpha_sweep(results, output_dir="figures", model_suffix=model_suffix)
    except Exception as e:
        print(f"[WARNING] Plot generation failed: {e}")
    
    print()


if __name__ == "__main__":
    main()
