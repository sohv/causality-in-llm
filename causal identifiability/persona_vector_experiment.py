import torch
import numpy as np
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
import json
import yaml
import os


class PersonaVectorExperiment:
    """
    Implements the non-identifiability experiment from Section 6 of the paper.
    Tests two models (Qwen2.5, Llama-3.1) and two traits (formality, sentiment).
    """

    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        
        # Load configs
        config_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(config_dir, 'config', 'config.json'), 'r') as f:
            self.config = json.load(f)
        with open(os.path.join(config_dir, 'config', 'model_config.yml'), 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # Initialize semantic probes
        self._init_semantic_probes()

    def _init_semantic_probes(self):
        """
        Initialize semantic probes for computing φ(o).
        These are lightweight models that measure semantic properties.
        """
        print("Loading semantic probes...")

        # Sentiment probe: outputs score in [-1, 1]
        try:
            self.sentiment_probe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment probe: {e}")
            self.sentiment_probe = None

        # Formality probe: try neural classifier first, fall back to heuristic
        try:
            self.formality_probe = pipeline(
                "text-classification",
                model="s-nlp/roberta-base-formality-ranker",
                device=0 if self.device == "cuda" else -1
            )
            self.formality_is_neural = True
            print("Loaded neural formality probe")
        except Exception as e:
            print(f"Could not load neural formality probe: {e}")
            print("Falling back to heuristic formality probe")
            self.formality_probe = self._formality_heuristic
            self.formality_is_neural = False

        print("Semantic probes loaded.")

    def _formality_heuristic(self, text: str) -> float:
        """
        Simple formality score based on linguistic features.
        Returns value in [0, 1] where 1 is most formal.
        """
        # Informal markers
        informal_markers = ['gonna', 'wanna', 'yeah', 'nah', 'hey', 'cool', 'dude',
                          'stuff', 'lots', 'kinda', 'sorta', '!', 'lol', 'omg']

        # Formal markers
        formal_markers = ['therefore', 'furthermore', 'consequently', 'moreover',
                         'nevertheless', 'thus', 'hence', 'regarding', 'pursuant']

        text_lower = text.lower()

        informal_count = sum(1 for marker in informal_markers if marker in text_lower)
        formal_count = sum(1 for marker in formal_markers if marker in text_lower)

        # Sentence length (longer sentences tend to be more formal)
        avg_sentence_len = len(text.split()) / max(text.count('.') + text.count('!') + text.count('?'), 1)

        # Combine features
        formality_score = (formal_count - informal_count + avg_sentence_len / 20.0)

        # Normalize to [0, 1]
        formality_score = 1 / (1 + np.exp(-formality_score))

        return formality_score

    def _politeness_heuristic(self, text: str) -> float:
        """
        Simple politeness score based on linguistic features.
        Returns value in [0, 1] where 1 is most polite.
        """
        polite_markers = ['please', 'thank you', 'thanks', 'appreciate', 'would you mind',
                         'could you', 'if possible', 'kindly', 'respectfully', 'courtesy',
                         'sincerely', 'regards', 'apologize', 'excuse me', 'pardon']

        rude_markers = ['damn', 'hell', 'shut up', 'idiot', 'stupid', 'useless', 'hate',
                       'disgusting', 'terrible', 'awful', 'horrible']

        text_lower = text.lower()

        polite_count = sum(1 for marker in polite_markers if marker in text_lower)
        rude_count = sum(1 for marker in rude_markers if marker in text_lower)

        # Sentence length (shorter, more concise = polite)
        total_words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        avg_sentence_len = total_words / max(sentences, 1)

        # Combine features
        politeness_score = (polite_count - rude_count + (20.0 / max(avg_sentence_len, 1)))

        # Normalize to [0, 1]
        politeness_score = 1 / (1 + np.exp(-politeness_score))

        return politeness_score

    def _humor_heuristic(self, text: str) -> float:
        """
        Simple humor score based on linguistic features.
        Returns value in [0, 1] where 1 is most humorous.
        """
        humorous_markers = ['lol', 'haha', 'funny', 'hilarious', 'joke', 'witty', 'silly',
                           'comedy', 'laugh', 'absurd', 'ridiculous', 'ironic', 'pun']

        serious_markers = ['unfortunately', 'sadly', 'regret', 'apologize', 'serious',
                          'critical', 'urgent', 'concerning', 'worried', 'nervous']

        text_lower = text.lower()

        humorous_count = sum(1 for marker in humorous_markers if marker in text_lower)
        serious_count = sum(1 for marker in serious_markers if marker in text_lower)

        # Exclamation marks indicate humor/excitement
        exclamation_count = text.count('!')
        question_count = text.count('?')

        # Combine features
        humor_score = (humorous_count - serious_count + (exclamation_count / max(len(text.split()), 1)))

        # Normalize to [0, 1]
        humor_score = 1 / (1 + np.exp(-humor_score))

        return humor_score

    def compute_semantic_score(self, text: str, trait: str) -> float:
        """
        Compute semantic score φ(o) for generated text.

        Args:
            text: Generated text
            trait: 'humor', 'formality', or 'politeness'

        Returns:
            Scalar semantic score
        """
        if trait == "humor":
            return self._humor_heuristic(text)

        elif trait == "formality":
            if hasattr(self, 'formality_is_neural') and self.formality_is_neural:
                try:
                    result = self.formality_probe(text[:512])[0]
                    # Map to continuous score: formal → positive, informal → negative
                    is_formal = 'formal' in result['label'].lower()
                    score = result['score'] if is_formal else -result['score']
                    return score  # Returns value in [-1, 1]
                except Exception as e:
                    print(f"Warning: Neural formality probe failed: {e}")
                    return self._formality_heuristic(text)
            else:
                return self._formality_heuristic(text)

        elif trait == "politeness":
            # Politeness uses heuristic probe (like formality)
            return self._politeness_heuristic(text)

        else:
            raise ValueError(f"Unknown trait: {trait}")

    def create_contrastive_prompts(self, trait: str, n_pairs: int = 50) -> List[Tuple[str, str]]:
        """
        Create contrastive prompt pairs for extracting steering vectors.
        Following the paper's methodology (Section 6.1).
        """
        prompts = []
        persona_prompts = self.config['persona_prompts']

        if trait == "formality":
            topics = persona_prompts['formality']['topics']
            positive_template = persona_prompts['formality']['positive_template']
            negative_template = persona_prompts['formality']['negative_template']
            
            for i, topic in enumerate(topics[:n_pairs]):
                positive = positive_template.format(topic=topic)
                negative = negative_template.format(topic=topic)
                prompts.append((positive, negative))

        elif trait == "sentiment":
            contexts = persona_prompts['sentiment']['contexts']
            positive_template = persona_prompts['sentiment']['positive_template']
            negative_template = persona_prompts['sentiment']['negative_template']
            
            for i, context in enumerate(contexts[:n_pairs]):
                positive = positive_template.format(context=context)
                negative = negative_template.format(context=context)
                prompts.append((positive, negative))

        elif trait == "politeness":
            contexts = persona_prompts['politeness']['contexts']
            positive_template = persona_prompts['politeness']['positive_template']
            negative_template = persona_prompts['politeness']['negative_template']
            
            for i, context in enumerate(contexts[:n_pairs]):
                polite = positive_template.format(context=context)
                rude = negative_template.format(context=context)
                prompts.append((polite, rude))

        elif trait == "humor":
            situations = persona_prompts['humor']['situations']
            positive_template = persona_prompts['humor']['positive_template']
            negative_template = persona_prompts['humor']['negative_template']
            
            for i, situation in enumerate(situations[:n_pairs]):
                humorous = positive_template.format(situation=situation)
                serious = negative_template.format(situation=situation)
                prompts.append((humorous, serious))

        return prompts

    def get_hidden_states(self, prompt: str) -> torch.Tensor:
        """
        Extract hidden states from the model for a given prompt.
        Returns the hidden state at the final token position from the middle layer.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get activation at final token position for middle layer
            layer_idx = len(self.model.model.layers) // 2
            hidden_state = outputs.hidden_states[layer_idx][:, -1, :]

        return hidden_state.squeeze(0)

    def extract_steering_vector(self, trait: str, n_pairs: int = 50) -> torch.Tensor:
        """
        Extract steering vector using contrastive activation addition.
        v = E[h(x+)] - E[h(x-)]
        Following Section 6.1 of the paper.
        """
        prompt_pairs = self.create_contrastive_prompts(trait, n_pairs)

        positive_activations = []
        negative_activations = []

        print(f"Extracting {trait} steering vector from {n_pairs} prompt pairs...")
        for pos, neg in tqdm(prompt_pairs):
            pos_hidden = self.get_hidden_states(pos)
            neg_hidden = self.get_hidden_states(neg)

            positive_activations.append(pos_hidden)
            negative_activations.append(neg_hidden)

        # Average and compute difference
        pos_mean = torch.stack(positive_activations).mean(dim=0)
        neg_mean = torch.stack(negative_activations).mean(dim=0)

        steering_vector = pos_mean - neg_mean

        return steering_vector

    def generate_with_steering(self, prompt: str, steering_vector: torch.Tensor,
                               alpha: float = 1.0, max_new_tokens: int = 50,
                               temperature: float = 0.8, num_return_sequences: int = 1) -> List[str]:
        """
        Generate text with steering vector applied to activations.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Hook to add steering vector
        layer_idx = len(self.model.model.layers) // 2

        def steering_hook(module, input, output):
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            # Add steering to the last token (ensure device and dtype match)
            steering_gpu = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states[:, -1, :] += alpha * steering_gpu

            if rest:
                return (hidden_states,) + rest
            else:
                return hidden_states

        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        hook_handle.remove()

        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts

    def test_orthogonal_component_irrelevance(self, prompts: List[str], v: torch.Tensor,
                                             trait: str, alpha: float = 1.0,
                                             n_orthogonal: int = 5) -> Dict:
        """
        Check if components orthogonal to v matter.
        
        If null space is large, then v_perp (orthogonal to v) should have
        minimal semantic effect compared to v_parallel.
        
        """
        print(f"\n{'='*80}")
        print(f"Testing Orthogonal Component Irrelevance")
        print(f"{'='*80}\n")
        
        v_norm = torch.norm(v).item()
        
        # Test 1: Original vector v
        print("Test 1: Original vector v")
        scores_v = []
        for prompt in tqdm(prompts[:20], desc="Original v"):
            texts = self.generate_with_steering(prompt, v, alpha=alpha,
                                                max_new_tokens=40, num_return_sequences=5)
            scores = [self.compute_semantic_score(text, trait) for text in texts]
            scores_v.extend(scores)
        
        mean_v = np.mean(scores_v)
        std_v = np.std(scores_v)
        print(f"  Scores with v: μ={mean_v:.4f}, σ={std_v:.4f}\n")
        
        # Test 2: v + orthogonal component (should be similar to v if null space exists)
        print("Test 2: v + random orthogonal components")
        results = []
        
        for i in range(n_orthogonal):
            # Create random vector orthogonal to v
            random_vec = torch.randn_like(v)
            v_perp = random_vec - (random_vec @ v) / (v @ v) * v  # Gram-Schmidt
            v_perp = v_perp / torch.norm(v_perp) * v_norm  # Same magnitude as v
            
            v_plus_perp = v + v_perp
            
            print(f"\n  Orthogonal vector {i+1}/{n_orthogonal}:")
            print(f"    ||v_perp|| = {torch.norm(v_perp).item():.4f}")
            print(f"    v · v_perp = {(v @ v_perp).item():.6f} (should be ~0)")
            
            scores_perp = []
            for prompt in tqdm(prompts[:20], desc=f"  v+perp {i+1}", leave=False):
                texts = self.generate_with_steering(prompt, v_plus_perp, alpha=alpha,
                                                    max_new_tokens=40, num_return_sequences=5)
                scores = [self.compute_semantic_score(text, trait) for text in texts]
                scores_perp.extend(scores)
            
            mean_perp = np.mean(scores_perp)
            std_perp = np.std(scores_perp)
            
            mean_diff = abs(mean_perp - mean_v)
            cohens_d = mean_diff / np.sqrt((std_v**2 + std_perp**2) / 2)
            correlation = np.corrcoef(scores_v, scores_perp)[0, 1]
            
            print(f"    Scores: μ={mean_perp:.4f}, σ={std_perp:.4f}")
            print(f"    Cohen's d: {cohens_d:.4f}")
            print(f"    Correlation: {correlation:.4f}")
            
            results.append({
                'cohens_d': cohens_d,
                'correlation': correlation,
                'mean_diff': mean_diff
            })
        
        # Test 3: Just the orthogonal component (should have weak effect)
        print(f"\nTest 3: Pure orthogonal components (without v)")
        orthogonal_effects = []
        
        for i in range(n_orthogonal):
            random_vec = torch.randn_like(v)
            v_perp = random_vec - (random_vec @ v) / (v @ v) * v
            v_perp = v_perp / torch.norm(v_perp) * v_norm
            
            scores_only_perp = []
            for prompt in tqdm(prompts[:20], desc=f"  perp-only {i+1}", leave=False):
                texts = self.generate_with_steering(prompt, v_perp, alpha=alpha,
                                                    max_new_tokens=40, num_return_sequences=5)
                scores = [self.compute_semantic_score(text, trait) for text in texts]
                scores_only_perp.extend(scores)
            
            mean_only_perp = np.mean(scores_only_perp)
            effect_ratio = abs(mean_only_perp) / abs(mean_v) if mean_v != 0 else 0
            orthogonal_effects.append(effect_ratio)
            
            print(f" Orthogonal {i+1}: effect = {abs(mean_only_perp):.4f} ({effect_ratio*100:.1f}% of v)")
        
        cohens_ds = [r['cohens_d'] for r in results]
        correlations = [r['correlation'] for r in results]
        
        print(f"v + orthogonal: Cohen's d = {np.mean(cohens_ds):.4f} ± {np.std(cohens_ds):.4f}")
        print(f"v + orthogonal: Correlation = {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
        print(f"Orthogonal-only: Mean effect = {np.mean(orthogonal_effects)*100:.1f}% ± {np.std(orthogonal_effects)*100:.1f}%")
        
        return {
            'v_scores': {'mean': mean_v, 'std': std_v},
            'v_plus_perp': results,
            'perp_only_effects': orthogonal_effects,
            'summary': {
                'mean_cohens_d': np.mean(cohens_ds),
                'mean_correlation': np.mean(correlations),
                'mean_perp_effect': np.mean(orthogonal_effects)
            }
        }