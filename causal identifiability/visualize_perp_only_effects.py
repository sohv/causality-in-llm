"""
Visualize orthogonal-only effects: Strongest evidence for non-identifiability.

Shows boxplot of perp_only_effects across seeds and traits.
If perp_only_effect ≈ 1.0, then v_perp alone produces equivalent effect to v.

Usage:
    python visualize_perp_only_effects.py --input results/orthogonal_test_results_10_seeds.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

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


def plot_perp_only_effects(results_file: str, output_dir: str = "figures"):
    """
    Create boxplot of perp-only effects across traits and seeds.
    
    Args:
        results_file: Path to orthogonal_test_results JSON
        output_dir: Directory to save plot
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract perp_only_effects per trait
    traits_data = {}
    for trait, data in results.items():
        if 'perp_only_effects' in data:
            traits_data[trait] = data['perp_only_effects']
    
    # Create figure: single-column ACL format (3.25 inches wide)
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    
    # Prepare data for boxplot
    trait_names = sorted(traits_data.keys())
    data_to_plot = [traits_data[trait] for trait in trait_names]
    
    # Create boxplot
    bp = ax.boxplot(data_to_plot, labels=trait_names, patch_artist=True,
                    widths=0.65, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='darkred', 
                                  markeredgecolor='darkred', markersize=6),
                    medianprops=dict(color='darkred', linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    boxprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))
    
    # Style boxes with stronger colors and no transparency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)
        patch.set_linewidth(0.8)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
              label='Equivalence (v⊥ ≡ v)', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Trait', fontsize=8)
    ax.set_ylabel('Perp-Only Effect Ratio', fontsize=8)
    ax.set_title('Non-Identifiability: Orthogonal Equivalence (n=10)', 
                fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.95, borderpad=0.8)
    
    # Set y-axis limits to focus on the data
    y_min = min([min(d) for d in data_to_plot]) - 0.1
    y_max = max([max(d) for d in data_to_plot]) + 0.1
    ax.set_ylim(y_min, y_max)
    
    # Add data points: reduced visual noise
    for i, (trait, values) in enumerate(zip(trait_names, data_to_plot)):
        x_pos = np.random.normal(i+1, 0.04, len(values))
        ax.scatter(x_pos, values, alpha=0.35, s=30, color='black')
    
    # Add dotted guide lines from each box median to y-axis
    # median_values = [np.median(values) for values in data_to_plot]
    # for i, median_val in enumerate(median_values):
    #     ax.plot([0.5, i+1], [median_val, median_val], 'k--', alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    
    # Save plot: PDF for paper submission, PNG for quick review
    output_pdf = f"{output_dir}/perp_only_effects_boxplot.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PDF (paper-ready): {output_pdf}")
    
    output_png = output_pdf.replace('.pdf', '.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PNG (preview): {output_png}")
    
    plt.close()
    
    # Print statistics
    print(f"\nPERP-ONLY EFFECT STATISTICS\n")
    for trait in trait_names:
        values = traits_data[trait]
        print(f"{trait.capitalize():15s}: mean={np.mean(values):.4f}, "
              f"std={np.std(values):.4f}, "
              f"median={np.median(values):.4f}, "
              f"range=[{np.min(values):.4f}, {np.max(values):.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Visualize perp-only effects")
    parser.add_argument('--input', type=str, default='results/orthogonal_test_results_10_seeds.json',
                       help='Path to orthogonal test results JSON')
    parser.add_argument('--output', type=str, default='figures',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.input}")
    plot_perp_only_effects(args.input, args.output)
    print()


if __name__ == "__main__":
    main()
