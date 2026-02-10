"""
Recreate alpha-sweep plots from existing results JSON files with ACL formatting.

Usage:
    python recreate_alpha_sweep_plots.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

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


def plot_alpha_sweep_from_json(results_file: str, output_dir: str = "figures"):
    """
    Load alpha-sweep results from JSON and create ACL-formatted plot.
    
    Args:
        results_file: Path to alpha_sweep_results JSON file
        output_dir: Directory to save the plot
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    alphas = results['alphas']
    alpha_results = results['alpha_results']
    trait = results['trait']
    model_name = results['model']
    
    # Determine model suffix
    if "qwen" in model_name.lower():
        model_suffix = "qwen"
    elif "llama" in model_name.lower():
        model_suffix = "llama"
    else:
        model_suffix = model_name.split('/')[-1].lower().replace('-', '_')
    
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
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    
    output_pdf = f"{output_dir}/alpha_sweep_response_curves_{model_suffix}.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PDF (paper-ready): {output_pdf}")
    
    output_png = output_pdf.replace('.pdf', '.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PNG (preview): {output_png}")
    
    plt.close()


def main():
    """Recreate plots from existing results files."""
    results_dir = "results"
    output_dir = "figures"
    
    # Find all alpha_sweep results files
    result_files = [
        f for f in os.listdir(results_dir)
        if f.startswith("alpha_sweep_results_") and f.endswith(".json")
    ]
    
    if not result_files:
        print(f"[ERROR] No alpha_sweep_results_*.json files found in {results_dir}/")
        return
    
    print(f"Recreating {len(result_files)} alpha-sweep plots with ACL formatting...\n")
    
    for result_file in sorted(result_files):
        result_path = os.path.join(results_dir, result_file)
        print(f"Processing: {result_file}")
        plot_alpha_sweep_from_json(result_path, output_dir)
        print()
    
    print("Done! All plots recreated with ACL formatting.")


if __name__ == "__main__":
    main()
