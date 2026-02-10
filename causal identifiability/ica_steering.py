"""
This script demonstrates how Independent Component Analysis (ICA) can be used
to extract more identifiable steering vectors compared to standard contrastive
mean difference methods.

empirically validates Proposition 2(a): Statistical independence constraints enable identifiability.
"""

import numpy as np
import torch
from sklearn.decomposition import FastICA
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import json
import os
from persona_vector_experiment import PersonaVectorExperiment
from tqdm import tqdm

class ICASteeringExtractor:
    """Extract steering vectors using ICA to ensure statistical independence."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", layer=16, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.layer = layer
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_activations(self, prompts, max_length=50, desc="Extracting activations"):
        """extract activations at specified layer for given prompts."""
        activations = []
        
        for prompt in tqdm(prompts, desc=desc, total=len(prompts)):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # get activation at final token position for specified layer
                hidden_state = outputs.hidden_states[self.layer][:, -1, :]
                activations.append(hidden_state.cpu().numpy())
      
        return np.vstack(activations)
    
    def extract_contrastive_vector(self, positive_prompts, negative_prompts):
        """Standard contrastive mean difference method."""
        print("Extracting standard contrastive vector...")
        pos_acts = self.get_activations(positive_prompts)
        neg_acts = self.get_activations(negative_prompts)
        
        # Mean difference
        v_standard = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        v_standard = v_standard / np.linalg.norm(v_standard)
        
        return v_standard
    
    def extract_ica_vectors(self, positive_prompts, negative_prompts, n_components=3):
        """ICA-based extraction ensuring statistical independence."""
        print("Extracting ICA-based vectors...")
        
        # Collect activations from both positive and negative prompts
        all_prompts = positive_prompts + negative_prompts
        activations = self.get_activations(all_prompts)
        
        # Apply ICA to recover independent components
        ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
        sources = ica.fit_transform(activations)
        mixing_matrix = ica.mixing_  # Shape: (d, n_components)
        
        # Each column of mixing_matrix is an independent steering vector
        ica_vectors = []
        for i in range(n_components):
            v = mixing_matrix[:, i]
            v = v / np.linalg.norm(v)
            ica_vectors.append(v)
        
        # Determine which component best separates positive/negative
        n_pos = len(positive_prompts)
        best_idx = 0
        best_separation = 0
        
        for i in range(n_components):
            pos_proj = sources[:n_pos, i].mean()
            neg_proj = sources[n_pos:, i].mean()
            separation = abs(pos_proj - neg_proj)
            
            if separation > best_separation:
                best_separation = separation
                best_idx = i
        
        print(f"Selected ICA component {best_idx} (separation: {best_separation:.3f})")
        
        return ica_vectors[best_idx], ica_vectors, sources
    
    def test_identifiability(self, v_standard, v_ica, test_prompts_base, trait, n_seeds=10, effect_threshold=0.01):
        """
        RIGOROUS Prop 2 test: Compare non-identifiability via perp-only effect ratio on TRAIT SCORES.
        
        Key fixes:
        1. Baseline: generate_with_steering(prompt, v, alpha=0.0) - true no-op
        2. Paired differences: reuse baseline for both v and v⊥ comparisons
        3. Effect filtering: skip seeds where abs(effect_v) < threshold (avoids ratio explosion)
        
        Measures actual behavioral change in generated outputs:
        - Generate text with steering (v or v⊥)
        - Score outputs with trait classifier
        - Effect = mean(trait_score_steered) - mean(trait_score_baseline)
        - Ratio = effect(v⊥_only) / effect(v)
        
        Non-identifiable: ratio ≈ 1.0 (null space fully encodes trait effect)
        Identifiable: ratio << 1.0 (null space has negligible effect on trait)
        
        Identifiability gap = 1 - mean(ratio)
        """
        import sys
        print(f"\n{'='*80}")
        print(f"Testing Prop 2 identifiability for {trait} via trait-score-based perp-only ratio...")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        
        print(" Loading model for semantic scoring (may take 1-2 min)...")
        sys.stdout.flush()
        experiment = PersonaVectorExperiment(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            device=self.device
        )
        print(" Model loaded\n")
        sys.stdout.flush()
        
        # Use base prompts (don't replicate - use real distinct prompts)
        test_prompts = test_prompts_base
        if len(test_prompts) < 10:
            print(f"  WARNING: Only {len(test_prompts)} test prompts. Using at least 50 is recommended.")
        print(f"  Using {len(test_prompts)} test prompts\n")
        sys.stdout.flush()
        
        # Convert numpy vectors to torch tensors for steering
        v_standard_tensor = torch.tensor(v_standard, dtype=torch.float32, device=self.device)
        v_ica_tensor = torch.tensor(v_ica, dtype=torch.float32, device=self.device)
        
        # ===== Test Standard Contrastive Vector =====
        print("  Testing standard contrastive v_std...")
        sys.stdout.flush()
        perp_ratios_std = []
        
        for seed in tqdm(range(n_seeds), desc="  Seeds (standard)", total=n_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Generate random orthogonal vector to v_std
            v_perp_std = np.random.randn(len(v_standard))
            v_perp_std = v_perp_std - np.dot(v_perp_std, v_standard) * v_standard
            v_perp_std = v_perp_std / np.linalg.norm(v_perp_std)
            v_perp_std_tensor = torch.tensor(v_perp_std, dtype=torch.float32, device=self.device)
            
            # Collect TRIPLES: (baseline, v_steered, v_perp) only when all 3 succeed
            # This ensures proper pairing - no mismatched list lengths
            triples_std = []
            
            for prompt in test_prompts:
                try:
                    # Baseline: alpha=0.0 with same vector (true no-op)
                    texts_baseline = experiment.generate_with_steering(
                        prompt, v_standard_tensor, alpha=0.0, num_return_sequences=1
                    )
                    score_baseline = np.mean([experiment.compute_semantic_score(text, trait) for text in texts_baseline])
                    
                    # Effect with v_std
                    texts_steered = experiment.generate_with_steering(
                        prompt, v_standard_tensor, alpha=1.0, num_return_sequences=1
                    )
                    score_steered = np.mean([experiment.compute_semantic_score(text, trait) for text in texts_steered])
                    
                    # Effect with v_perp_only
                    texts_perp = experiment.generate_with_steering(
                        prompt, v_perp_std_tensor, alpha=1.0, num_return_sequences=1
                    )
                    score_perp = np.mean([experiment.compute_semantic_score(text, trait) for text in texts_perp])
                    
                    # Only append if all 3 succeed
                    triples_std.append((score_baseline, score_steered, score_perp))
                except Exception as e:
                    continue
            
            # Only proceed if we have enough valid samples
            if len(triples_std) < 3:
                print(f"    Seed {seed}: Skipped (too few valid prompts, got {len(triples_std)})")
                continue
            
            # Unpack triples (now properly paired)
            baseline_scores_all = [t[0] for t in triples_std]
            scores_v_std_all = [t[1] for t in triples_std]
            scores_v_perp_all = [t[2] for t in triples_std]
            
            # Compute effects (properly paired differences)
            effect_v_std = np.mean(scores_v_std_all) - np.mean(baseline_scores_all)
            effect_v_perp = np.mean(scores_v_perp_all) - np.mean(baseline_scores_all)
            
            # Filter: skip if effect is too small (avoids ratio explosion)
            if abs(effect_v_std) < effect_threshold:
                print(f"    Seed {seed}: Skipped (effect too small: {effect_v_std:.4f})")
                continue
            
            # Ratio
            ratio = abs(effect_v_perp) / abs(effect_v_std)
            perp_ratios_std.append(ratio)
            print(f"    Seed {seed}: effect(v)={effect_v_std:.4f}, effect(v⊥)={effect_v_perp:.4f}, ratio={ratio:.3f}")
        
        # ===== Test ICA Vector =====
        print("\n  Testing ICA-based v_ica...")
        sys.stdout.flush()
        perp_ratios_ica = []
        
        for seed in tqdm(range(n_seeds), desc="  Seeds (ICA)", total=n_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Generate random orthogonal vector to v_ica
            v_perp_ica = np.random.randn(len(v_ica))
            v_perp_ica = v_perp_ica - np.dot(v_perp_ica, v_ica) * v_ica
            v_perp_ica = v_perp_ica / np.linalg.norm(v_perp_ica)
            v_perp_ica_tensor = torch.tensor(v_perp_ica, dtype=torch.float32, device=self.device)
            
            # Collect TRIPLES for ICA-extracted vectors
            triples_ica = []
            
            for prompt in test_prompts:
                try:
                    # Baseline: alpha=0.0 with same vector
                    texts_baseline = experiment.generate_with_steering(
                        prompt, v_ica_tensor, alpha=0.0, num_return_sequences=1
                    )
                    score_baseline = np.mean([experiment.compute_semantic_score(text, trait) for text in texts_baseline])
                    
                    # Effect with v_ica
                    texts_steered = experiment.generate_with_steering(
                        prompt, v_ica_tensor, alpha=1.0, num_return_sequences=1
                    )
                    score_steered = np.mean([experiment.compute_semantic_score(text, trait) for text in texts_steered])
                    
                    # Effect with v_perp_only
                    texts_perp = experiment.generate_with_steering(
                        prompt, v_perp_ica_tensor, alpha=1.0, num_return_sequences=1
                    )
                    score_perp = np.mean([experiment.compute_semantic_score(text, trait) for text in texts_perp])
                    
                    # Only append if all 3 succeed
                    triples_ica.append((score_baseline, score_steered, score_perp))
                except Exception as e:
                    continue
            
            # Only proceed if we have enough valid samples
            if len(triples_ica) < 3:
                print(f"    Seed {seed}: Skipped (too few valid prompts, got {len(triples_ica)})")
                continue
            
            # Unpack triples (now properly paired)
            baseline_scores_all = [t[0] for t in triples_ica]
            scores_v_ica_all = [t[1] for t in triples_ica]
            scores_v_perp_ica_all = [t[2] for t in triples_ica]
            
            # Compute effects (paired differences)
            effect_v_ica = np.mean(scores_v_ica_all) - np.mean(baseline_scores_all)
            effect_v_perp_ica = np.mean(scores_v_perp_ica_all) - np.mean(baseline_scores_all)
            
            # Filter: skip if effect is too small
            if abs(effect_v_ica) < effect_threshold:
                print(f"    Seed {seed}: Skipped (effect too small: {effect_v_ica:.4f})")
                continue
            
            # Ratio
            ratio = abs(effect_v_perp_ica) / abs(effect_v_ica)
            perp_ratios_ica.append(ratio)
            print(f"    Seed {seed}: effect(v_ica)={effect_v_ica:.4f}, effect(v⊥)={effect_v_perp_ica:.4f}, ratio={ratio:.3f}")
        
        # Compute identifiability gaps
        gap_std = 1.0 - np.mean(perp_ratios_std) if perp_ratios_std else 0.0
        gap_ica = 1.0 - np.mean(perp_ratios_ica) if perp_ratios_ica else 0.0
        
        return {
            'standard_perp_ratios': perp_ratios_std,
            'standard_mean_ratio': np.mean(perp_ratios_std) if perp_ratios_std else None,
            'standard_ratio_std': np.std(perp_ratios_std) if perp_ratios_std else None,
            'standard_identifiability_gap': gap_std,
            'standard_n_valid_seeds': len(perp_ratios_std),
            'ica_perp_ratios': perp_ratios_ica,
            'ica_mean_ratio': np.mean(perp_ratios_ica) if perp_ratios_ica else None,
            'ica_ratio_std': np.std(perp_ratios_ica) if perp_ratios_ica else None,
            'ica_identifiability_gap': gap_ica,
            'ica_n_valid_seeds': len(perp_ratios_ica),
            'gap_improvement': gap_ica - gap_std
        }


def create_prompts(trait):
    """Load contrastive prompts from config.json."""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(config_dir, 'config', 'config.json'), 'r') as f:
        config = json.load(f)
    
    trait_config = config['persona_prompts'].get(trait, {})
    
    # Generate positive/negative pairs from templates and topics/contexts/situations
    positive_template = trait_config.get('positive_template', '')
    negative_template = trait_config.get('negative_template', '')
    
    # Use first key that exists (topics, contexts, or situations)
    items = (trait_config.get('topics') or 
             trait_config.get('contexts') or 
             trait_config.get('situations') or [])
    
    positive_prompts = [positive_template.format(topic=item if 'topic=' in positive_template else item) 
                        if '{' in positive_template else positive_template + ' ' + item 
                        for item in items[:10]]
    
    negative_prompts = [negative_template.format(topic=item if 'topic=' in negative_template else item)
                        if '{' in negative_template else negative_template + ' ' + item 
                        for item in items[:10]]
    
    # Use all test_prompts from config (40-50 items for robust validation)
    test_prompts = config.get('test_prompts', [])
    
    return {
        'positive': positive_prompts,
        'negative': negative_prompts,
        'test': test_prompts
    }


def main():
    """Run ICA-based steering extraction and identifiability comparison."""
    
    print("=" * 80)
    print("ICA-Based Steering for Improved Identifiability")
    print("=" * 80)
    
    # Initialize extractor
    extractor = ICASteeringExtractor(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        layer=16
    )
    
    # Test for formality trait
    trait = 'formality'
    prompts = create_prompts(trait)
    
    # Extract steering vectors using both methods
    print(f"\n{'=' * 80}")
    print(f"Extracting steering vectors for: {trait.upper()}")
    print(f"{'=' * 80}")
    
    v_standard = extractor.extract_contrastive_vector(
        prompts['positive'], 
        prompts['negative']
    )
    
    v_ica, all_ica_vectors, sources = extractor.extract_ica_vectors(
        prompts['positive'],
        prompts['negative'],
        n_components=3
    )
    
    # Test identifiability
    results = extractor.test_identifiability(
        v_standard,
        v_ica,
        prompts['test'],
        trait,
        n_seeds=10
    )
    
    # Print results
    print(f"\n{'=' * 80}")
    print("PROP 2 IDENTIFIABILITY VALIDATION: Perp-Only Effect Ratio Test")
    print(f"{'=' * 80}")
    
    print(f"\nStandard Contrastive Method:")
    if results['standard_mean_ratio'] is not None:
        print(f"  Mean v⊥-only effect ratio: {results['standard_mean_ratio']:.3f} ± {results['standard_ratio_std']:.3f}")
        print(f"  Identifiability gap: {results['standard_identifiability_gap']:.3f}")
        print(f"  (Lower gap → more non-identifiable, ratio closer to 1.0)")
    else:
        print(f"  Mean v⊥-only effect ratio: NA (insufficient valid seeds: {results['standard_n_valid_seeds']})")
    
    print(f"\nICA-Based Method:")
    if results['ica_mean_ratio'] is not None:
        print(f"  Mean v⊥-only effect ratio: {results['ica_mean_ratio']:.3f} ± {results['ica_ratio_std']:.3f}")
        print(f"  Identifiability gap: {results['ica_identifiability_gap']:.3f}")
        print(f"  (Higher gap → more identifiable, ratio drops below 1.0)")
    else:
        print(f"  Mean v⊥-only effect ratio: NA (insufficient valid seeds: {results['ica_n_valid_seeds']})")
    
    if results['standard_mean_ratio'] is not None and results['ica_mean_ratio'] is not None:
        print(f"\nProp 2 Improvement:")
        print(f"  Gap increase (ICA vs Baseline): {results['gap_improvement']:.3f}")
        print(f"  Direction: {'✓ ICA improves identifiability' if results['gap_improvement'] > 0.05 else '✗ Marginal/no improvement'}")
    
    print("\nProp 2 Validation Method:")
    print("  1. For each vector (baseline & ICA), sample orthogonal components v⊥")
    print("  2. Measure effect ratio = effect(v⊥ only) / effect(v)")
    print("  3. Non-identifiable: ratio ≈ 1.0 (null space fully encodes effect)")
    print("  4. Identifiable: ratio << 1.0 (null space has minimal effect)")
    print(f"\nResult:")
    if results['gap_improvement'] > 0.05:
        print(f"  ✓ ICA-based method reduces null-space degeneracy by {results['gap_improvement']:.1%}")
        print(f"  ✓ Prop 2(a) VALIDATED: Independence constraints enable identifiability")
    else:
        print(f"  ✗ ICA shows marginal improvement ({results['gap_improvement']:.1%})")
        print(f"  ✗ Hidden states may not satisfy ICA assumptions")
    
    # Save results
    output = {
        'trait': trait,
        'method': 'ICA vs Contrastive',
        'test_type': 'perp_only_effect_ratio',
        'metric_explanation': 'ratio = effect(v_perp_only) / effect(v_full). Lower ratio = more identifiable',
        'results': {
            'baseline_contrastive': {
                'mean_perp_ratio': float(results['standard_mean_ratio']) if results['standard_mean_ratio'] is not None else None,
                'std_perp_ratio': float(results['standard_ratio_std']) if results['standard_ratio_std'] is not None else None,
                'identifiability_gap': float(results['standard_identifiability_gap']) if results['standard_identifiability_gap'] is not None else None,
                'n_valid_seeds': results['standard_n_valid_seeds']
            },
            'ica_based': {
                'mean_perp_ratio': float(results['ica_mean_ratio']) if results['ica_mean_ratio'] is not None else None,
                'std_perp_ratio': float(results['ica_ratio_std']) if results['ica_ratio_std'] is not None else None,
                'identifiability_gap': float(results['ica_identifiability_gap']) if results['ica_identifiability_gap'] is not None else None,
                'n_valid_seeds': results['ica_n_valid_seeds']
            }
        },
        'prop2_validation': {
            'gap_improvement': float(results['gap_improvement']) if results['gap_improvement'] is not None else None,
            'validated': results['gap_improvement'] > 0.05 if results['gap_improvement'] is not None else False,
            'interpretation': 'ICA reduces null-space degeneracy (lower ratio = less non-identifiable behavior)' if results['gap_improvement'] is not None and results['gap_improvement'] > 0.05 else 'Marginal/insufficient improvement - suggests LLM hidden states may not satisfy ICA assumptions or insufficient samples'
        }
    }
    
    with open('ica_identifiability_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to: ica_identifiability_results.json")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
