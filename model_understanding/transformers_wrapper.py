from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForTokenClassification,
    PreTrainedModel, PreTrainedTokenizer
)
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseVisualizer(ABC):
    """Base class for all model visualizers."""

    def __init__(self, model_name: str):
        """
        Initialize the base visualizer.

        Args:
            model_name (str): Name of the pre-trained model.
        """
        self.model_name = model_name
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = self._load_model()
        self.model.to(device)
        self.model.eval()
        self.layer_outputs = []
        self.attention_weights = []

    def _reset_gradients(self):
        """Clear any gradients left over from a previous saliency computation."""
        self.model.zero_grad(set_to_none=True)

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Raw input embeddings - encoder-side for encoder-decoder models. Shared
        across all visualizers so the embedding source (position embeddings and
        LayerNorm must NOT already be applied - the model applies them itself
        when given inputs_embeds) only has to be got right in one place.

        Some models (Marian, Pegasus) store an `embed_scale` on the module that
        owns the embeddings and apply it only when given input_ids, so it is
        applied here too. Models that scale inside the embedding module itself
        (BART, M2M100) have no such attribute and need nothing extra."""
        if getattr(self.model.config, "is_encoder_decoder", False):
            owner = self.model.get_encoder()
            try:
                embeddings = owner.get_input_embeddings()(input_ids)
            except NotImplementedError:
                embeddings = owner.embed_tokens(input_ids)
        else:
            owner = self.model.base_model
            embeddings = self.model.get_input_embeddings()(input_ids)
        return embeddings * getattr(owner, "embed_scale", 1.0)

    @staticmethod
    def _non_id_inputs(inputs, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Every tokenizer output except input_ids (attention_mask,
        token_type_ids, ...), to pass alongside inputs_embeds. Passing only
        attention_mask silently drops token_type_ids, which changes the output
        for sentence-pair inputs such as QA (question, context). batch_size
        expands each tensor to match batched inputs_embeds (Captum runs all
        Integrated Gradients steps as one batch)."""
        return {k: v.expand(batch_size, -1) for k, v in inputs.items() if k != "input_ids"}

    def _tokens(self, input_ids: torch.Tensor) -> List[str]:
        """Token strings for the first (only) sequence in the batch. Indexes
        [0] rather than squeeze(), which returns a plain int (and so a plain
        string token) for a single-token input."""
        return self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    def _capture_layerwise(self, attentions, hidden_states):
        """Store per-layer attention/hidden-state tensors for the layer- and head-wise plots."""
        self.attention_weights = list(attentions) if attentions is not None else []
        self.layer_outputs = list(hidden_states) if hidden_states is not None else []

    def compute_attention_rollout(self) -> torch.Tensor:
        """Attention rollout (Abnar & Zuidema): composes per-layer attention
        (averaged over heads, mixed 50/50 with identity to account for the
        residual stream, then renormalized) across all captured layers via
        matrix multiplication, so it reflects indirect attention paths through
        earlier layers rather than one layer's raw attention in isolation. Call
        compute_attention first to populate self.attention_weights. Pure
        post-processing of already-captured data - no new forward pass.
        """
        if not self.attention_weights:
            raise ValueError("No attention weights available. Run compute_attention first.")

        seq_len = self.attention_weights[0].shape[-1]
        identity = torch.eye(seq_len)
        rollout = identity
        for attn in self.attention_weights:
            avg = attn[0].mean(dim=0).detach().cpu()  # [seq, seq], mean over heads
            avg_res = 0.5 * avg + 0.5 * identity
            avg_res = avg_res / avg_res.sum(dim=-1, keepdim=True)
            rollout = avg_res @ rollout
        return rollout

    def compute_attention_entropy_distance(self) -> List[Dict[str, Any]]:
        """Per-layer, per-head attention entropy (how focused vs. diffuse) and
        average attended distance (how local vs. long-range). Call
        compute_attention first to populate self.attention_weights. Pure
        post-processing of already-captured data - no new forward pass.
        """
        if not self.attention_weights:
            raise ValueError("No attention weights available. Run compute_attention first.")

        results = []
        for layer_idx, attn in enumerate(self.attention_weights):
            heads = attn[0].detach().cpu()  # [num_heads, seq, seq]
            seq_len = heads.shape[-1]
            positions = torch.arange(seq_len).float()
            distance_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            for head_idx in range(heads.shape[0]):
                probs = heads[head_idx]
                entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
                distance = (probs * distance_matrix).sum(dim=-1).mean().item()
                results.append({"layer": layer_idx, "head": head_idx, "entropy": entropy, "distance": distance})
        return results

    def _target_probability(self, inputs, embeddings: torch.Tensor, target: Any = None) -> Tuple[float, Any]:
        """Probability the model gives to `target` when run from `embeddings`,
        plus the target itself. With target=None, use the model's own
        prediction - the same target the subclass's compute_saliency explains.
        Used by compute_faithfulness."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _target_probability.")

    def compute_faithfulness(self, attributions: torch.Tensor, *texts: str,
                             fractions: Tuple[float, ...] = (0.1, 0.2, 0.5)) -> Dict[str, Any]:
        """Comprehensiveness and sufficiency (DeYoung et al., ERASER) of a
        per-token attribution (saliency, Integrated Gradients, ...) for the
        same input(s) it was computed on.

        For each fraction, take the top-k content tokens by attribution score
        (highest first; special tokens are never ranked or removed):
          comprehensiveness = p(full) - p(top-k removed)  -> high is faithful
          sufficiency       = p(full) - p(only top-k kept) -> low is faithful
        The target stays fixed to the full-input prediction. A token is
        "removed" by setting its input embedding to zero - the same baseline
        compute_integrated_gradients uses, which gives IG a built-in advantage
        on these metrics. k = ceil(fraction * content tokens), and at least 1
        for any fraction > 0. Opt-in: writes no shared state.
        """
        inputs = self.tokenizer(*texts, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        scores = attributions.detach().flatten().cpu()
        if scores.shape[0] != input_ids.shape[1]:
            raise ValueError(
                f"attributions has {scores.shape[0]} scores but the input has {input_ids.shape[1]} tokens."
            )

        special = torch.tensor(
            self.tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True),
            dtype=torch.bool,
        )
        content = (~special).nonzero().flatten()
        # Positions in the full sequence (not indices into `content`), highest score first.
        ranked = content[torch.argsort(scores[content], descending=True, stable=True)]
        num_content = len(content)

        with torch.no_grad():
            embeddings = self._input_embeddings(input_ids)
            full_prob, target = self._target_probability(inputs, embeddings)

            def prob_without(positions: torch.Tensor) -> float:
                masked = embeddings.clone()
                masked[0, positions.to(masked.device)] = 0.0
                return self._target_probability(inputs, masked, target)[0]

            per_fraction = []
            for fraction in fractions:
                k = 0 if fraction == 0 else min(num_content, max(1, math.ceil(round(fraction * num_content, 6))))
                per_fraction.append({
                    "fraction": fraction,
                    "k": k,
                    "comprehensiveness": full_prob - prob_without(ranked[:k]),
                    "sufficiency": full_prob - prob_without(ranked[k:]),
                })

        return {
            "full_probability": full_prob,
            "num_content_tokens": num_content,
            "per_fraction": per_fraction,
            "comprehensiveness_aopc": sum(r["comprehensiveness"] for r in per_fraction) / len(per_fraction),
            "sufficiency_aopc": sum(r["sufficiency"] for r in per_fraction) / len(per_fraction),
        }

    def _get_hidden_state_at(self, input_ids: torch.Tensor, layer_idx: int, position: int,
                              **model_kwargs) -> torch.Tensor:
        """Read the hidden state at (layer_idx, position) via a temporary forward
        hook on that transformer block, then remove the hook. Requires the
        subclass to implement _transformer_layers()."""
        captured = {}

        def hook(module, inp, output):
            captured["value"] = output[:, position, :].detach().clone()

        handle = self._transformer_layers()[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(input_ids, **model_kwargs)
        finally:
            handle.remove()
        return captured["value"]

    def _patch_hidden_state_at(self, input_ids: torch.Tensor, layer_idx: int, position: int,
                                value: torch.Tensor, **model_kwargs):
        """Run a forward pass with the hidden state at (layer_idx, position)
        overwritten with `value`, via a temporary forward hook. Requires the
        subclass to implement _transformer_layers()."""

        def hook(module, inp, output):
            output = output.clone()
            output[:, position, :] = value
            return output

        handle = self._transformer_layers()[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                return self.model(input_ids, **model_kwargs)
        finally:
            handle.remove()

    def _transformer_layers(self):
        """Return the indexable sequence of transformer blocks used by
        activation patching. Only ClassificationVisualizer and
        GenerativeVisualizer override this; other subclasses raise a clear
        error here instead of an opaque AttributeError deep inside a hook."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support activation patching "
            "(only ClassificationVisualizer and GenerativeVisualizer implement _transformer_layers)."
        )

    def compute_activation_patch_sweep(self, clean_text: str, corrupted_text: str) -> Tuple[torch.Tensor, List[str]]:
        """Run compute_activation_patch (subclass-provided) across every
        (layer, position) pair - the classic causal-tracing sweep. Returns a
        [num_layers, seq_len] grid of patch_effect plus the clean text's tokens
        for labeling.
        """
        clean_ids = self.tokenizer(clean_text)["input_ids"]
        num_layers = len(self._transformer_layers())
        seq_len = len(clean_ids)

        grid = torch.zeros(num_layers, seq_len)
        for layer_idx in range(num_layers):
            for position in range(seq_len):
                result = self.compute_activation_patch(clean_text, corrupted_text, layer_idx, position)
                grid[layer_idx, position] = result["patch_effect"]

        tokens = self.tokenizer.convert_ids_to_tokens(clean_ids)
        return grid, tokens

    def plot_activation_patch_sweep(self, grid: torch.Tensor, tokens: List[str], save_path: Optional[str] = None):
        """Visualize an activation-patching sweep as a layer-by-position heatmap:
        0 = patching had no effect, 1 = patching fully reproduced the corrupted
        prediction (see compute_activation_patch_sweep)."""
        plt.figure(figsize=(max(10, len(tokens)), 6))
        sns.heatmap(grid.numpy(), xticklabels=tokens, yticklabels=range(grid.shape[0]),
                    cmap="RdBu_r", center=0)
        plt.xlabel("Position")
        plt.ylabel("Layer")
        plt.title("Activation Patching: patch effect (0 = no effect, 1 = full corruption)")
        plt.tight_layout()
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Load the appropriate model type."""
        pass

    @abstractmethod
    def compute_saliency(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """Compute saliency scores for the input text."""
        pass

    @abstractmethod
    def compute_attention(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """Compute attention weights for the input text."""
        pass

    def _ensure_save_dir(self, save_dir: Optional[str]) -> None:
        """Ensure the save directory exists."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def plot_saliency(self, tokens: List[str], saliency_map: torch.Tensor,
                     title: str = "Saliency Map", save_path: Optional[str] = None):
        """
        Plot the saliency map for the given tokens.

        Args:
            tokens (List[str]): List of tokens.
            saliency_map (torch.Tensor): Saliency scores for each token.
            title (str): Title for the plot.
            save_path (Optional[str]): Path to save the plot.
        """
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))

        # Handle multi-dimensional saliency maps by taking mean across last dimension
        if len(saliency_map.shape) > 1:
            saliency_map = saliency_map.mean(dim=-1)

        plt.figure(figsize=(10, 5))
        plt.bar(tokens, saliency_map.detach().cpu().numpy())
        plt.xticks(rotation=90)
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_attention(self, tokens: List[str], attention_weights: torch.Tensor,
                      title: str = "Attention Weights", save_path: Optional[str] = None):
        """
        Plot attention weights as a heatmap.

        Args:
            tokens (List[str]): List of tokens.
            attention_weights (torch.Tensor): Attention weights (seq_len x seq_len).
            title (str): Title for the plot.
            save_path (Optional[str]): Path to save the plot.
        """
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens,
                   cmap="Blues", annot=False)
        plt.title(title)
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.xticks(rotation=90)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_layer_attention(self, tokens: List[str], save_path: Optional[str] = None):
        """Visualize attention patterns across all layers."""
        if not self.attention_weights:
            print("No attention weights available. Run compute_attention first.")
            return

        num_layers = len(self.attention_weights)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 5*num_layers))
        if num_layers == 1:
            axes = [axes]

        for i, attn in enumerate(self.attention_weights):
            if not isinstance(attn, torch.Tensor):
                print(f"Warning: Layer {i+1} attention weights are not a tensor, skipping...")
                continue

            # Average across heads and batch dimension
            try:
                attn_avg = attn.mean(dim=(0, 1)).detach().cpu()

                # Ensure the attention matrix is square and matches token length
                seq_len = min(len(tokens), attn_avg.shape[0], attn_avg.shape[1])
                attn_avg = attn_avg[:seq_len, :seq_len]

                sns.heatmap(attn_avg, xticklabels=tokens[:seq_len],
                           yticklabels=tokens[:seq_len],
                           cmap="Blues", ax=axes[i])
                axes[i].set_title(f"Layer {i+1} Attention")
            except Exception as e:
                print(f"Warning: Could not plot attention for layer {i+1}: {str(e)}")
                continue

        plt.tight_layout()
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    def plot_attention_heads(self, tokens: List[str], layer: int = -1,
                           save_path: Optional[str] = None):
        """Visualize individual attention heads in a specific layer."""
        if not self.attention_weights:
            print("No attention weights available. Run compute_attention first.")
            return

        layer_attn = self.attention_weights[layer]
        if not isinstance(layer_attn, torch.Tensor):
            print(f"Warning: Layer {layer+1} attention weights are not a tensor, skipping...")
            return

        # Get the number of attention heads
        num_heads = layer_attn.shape[1]
        seq_len = len(tokens)

        # Create a grid of subplots
        n_rows = 2
        n_cols = (num_heads + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        axes = axes.flatten()

        for head in range(num_heads):
            # Real per-head attention matrix: [seq_len, seq_len]
            head_attn = layer_attn[0, head, :seq_len, :seq_len].detach().cpu()

            try:
                sns.heatmap(head_attn, xticklabels=tokens,
                           yticklabels=tokens, cmap="Blues", ax=axes[head])
                axes[head].set_title(f"Head {head+1}")
            except Exception as e:
                print(f"Warning: Could not plot attention for head {head+1}: {str(e)}")
                continue

        # Hide empty subplots if any
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    def plot_layer_activations(self, save_path: Optional[str] = None):
        """Visualize activation patterns across different layers."""
        if not self.layer_outputs:
            print("No layer outputs available. Run compute_saliency or compute_attention first.")
            return

        num_layers = len(self.layer_outputs)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 5*num_layers))
        if num_layers == 1:
            axes = [axes]

        for i, activation in enumerate(self.layer_outputs):
            if isinstance(activation, tuple):
                activation = activation[0]  # Handle tuple outputs
            mean_activation = activation.mean(dim=1).detach().cpu()
            sns.heatmap(mean_activation, ax=axes[i], cmap="viridis")
            axes[i].set_title(f"Layer {i+1} Activation Pattern")

        plt.tight_layout()
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    def plot_gradient_flow(self, save_path: Optional[str] = None):
        """Visualize gradient flow through model layers."""
        ave_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())

        if not ave_grads:
            print("No gradients available. Run compute_saliency first.")
            return

        plt.figure(figsize=(15, 10))
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
        plt.xticks(range(len(ave_grads)), layers, rotation=45)
        plt.title("Gradient Flow")

        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    def plot_logit_lens(self, results: List[Tuple[str, float]], save_path: Optional[str] = None):
        """Visualize logit lens results: top predicted next token and its
        probability at each layer (see GenerativeVisualizer.compute_logit_lens)."""
        layers = list(range(len(results)))
        tokens = [r[0] for r in results]
        probs = [r[1] for r in results]

        plt.figure(figsize=(10, 5))
        plt.bar(layers, probs)
        plt.xticks(layers, [f"L{i}: {t!r}" for i, t in zip(layers, tokens)], rotation=90)
        plt.ylabel("Top-1 probability")
        plt.title("Logit Lens: top predicted token per layer")
        plt.tight_layout()
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    def plot_attention_entropy_distance(self, results: List[Dict[str, Any]], save_path: Optional[str] = None):
        """Visualize per-layer, per-head attention entropy and average attended
        distance (see compute_attention_entropy_distance) as two layer-by-head heatmaps."""
        num_layers = max(r["layer"] for r in results) + 1
        num_heads = max(r["head"] for r in results) + 1

        entropy_grid = torch.zeros(num_layers, num_heads)
        distance_grid = torch.zeros(num_layers, num_heads)
        for r in results:
            entropy_grid[r["layer"], r["head"]] = r["entropy"]
            distance_grid[r["layer"], r["head"]] = r["distance"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(entropy_grid, ax=axes[0], cmap="viridis")
        axes[0].set_title("Attention Entropy")
        axes[0].set_xlabel("Head")
        axes[0].set_ylabel("Layer")

        sns.heatmap(distance_grid, ax=axes[1], cmap="viridis")
        axes[1].set_title("Attention Distance")
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Layer")

        plt.tight_layout()
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    def plot_faithfulness(self, results: Dict[str, Any], title: str = "Faithfulness",
                          save_path: Optional[str] = None):
        """Plot comprehensiveness and sufficiency against the fraction of
        content tokens (see compute_faithfulness)."""
        fractions = [r["fraction"] for r in results["per_fraction"]]
        plt.figure(figsize=(8, 5))
        plt.plot(fractions, [r["comprehensiveness"] for r in results["per_fraction"]], marker="o",
                 label=f"Comprehensiveness (AOPC {results['comprehensiveness_aopc']:.3f}, higher is better)")
        plt.plot(fractions, [r["sufficiency"] for r in results["per_fraction"]], marker="o",
                 label=f"Sufficiency (AOPC {results['sufficiency_aopc']:.3f}, lower is better)")
        plt.axhline(0, color="k", linewidth=0.5)
        plt.xlabel("Fraction of content tokens")
        plt.ylabel("p(full) - p(perturbed)")
        plt.title(f"{title} (p(full) = {results['full_probability']:.3f})")
        plt.legend()
        plt.tight_layout()
        if save_path:
            self._ensure_save_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
        plt.close()

    def analyze(self, text: str, save_dir: Optional[str] = None):
        """
        Perform comprehensive analysis for a given text.

        Args:
            text (str): Input text to analyze.
            save_dir (Optional[str]): Directory to save visualizations.
        """
        # Clear previous outputs
        self.attention_weights = []
        self.layer_outputs = []

        # Basic analysis
        tokens, saliency_map = self.compute_saliency(text)
        print("\nSaliency Analysis:")
        save_path = f"{save_dir}/saliency_map.png" if save_dir else None
        self.plot_saliency(tokens, saliency_map, save_path=save_path)

        tokens, attention_weights = self.compute_attention(text)
        print("\nAttention Analysis:")
        save_path = f"{save_dir}/attention_map.png" if save_dir else None
        self.plot_attention(tokens, attention_weights, save_path=save_path)

        # Additional visualizations
        print("\nLayer-wise Attention Analysis:")
        save_path = f"{save_dir}/layer_attention.png" if save_dir else None
        self.plot_layer_attention(tokens, save_path=save_path)

        print("\nAttention Head Analysis:")
        save_path = f"{save_dir}/attention_heads.png" if save_dir else None
        self.plot_attention_heads(tokens, save_path=save_path)

        print("\nLayer Activation Analysis:")
        save_path = f"{save_dir}/layer_activations.png" if save_dir else None
        self.plot_layer_activations(save_path=save_path)

        print("\nGradient Flow Analysis:")
        save_path = f"{save_dir}/gradient_flow.png" if save_dir else None
        self.plot_gradient_flow(save_path=save_path)

class ClassificationVisualizer(BaseVisualizer):
    """Visualizer for classification models."""

    def _load_model(self) -> PreTrainedModel:
        # SDPA/flash attention implementations don't materialize attention
        # weights, so output_attentions=True would silently return None.
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, attn_implementation="eager"
        )

    def compute_saliency(self, text: str) -> Tuple[List[str], torch.Tensor]:
        self._reset_gradients()
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embeddings = self._input_embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs))
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1)

        logits[0, predicted_class].backward()

        gradients = embeddings.grad[0].abs().mean(dim=-1)
        saliency_map = gradients / gradients.max()

        tokens = self._tokens(input_ids)
        return tokens, saliency_map

    def _target_probability(self, inputs, embeddings, target=None):
        logits = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs)).logits
        probs = torch.softmax(logits[0], dim=-1)
        if target is None:
            target = probs.argmax().item()
        return probs[target].item(), target

    def compute_attention(self, text: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        self._capture_layerwise(outputs.attentions, outputs.hidden_states)

        attention_weights = outputs.attentions[-1][0].detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self._tokens(input_ids)
        return tokens, avg_attention

    def compute_integrated_gradients(self, text: str, n_steps: int = 50) -> Tuple[List[str], torch.Tensor, float]:
        """Integrated Gradients attribution (Sundararajan et al.) against a
        zero-embedding baseline, targeting the same predicted-class logit
        compute_saliency does. More faithful than raw input-gradient saliency:
        satisfies a completeness axiom (sum(attributions) == F(input) -
        F(baseline)), which the returned convergence delta directly measures.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embeddings = self._input_embeddings(input_ids)
        baseline = torch.zeros_like(embeddings)

        with torch.no_grad():
            target = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs)).logits.argmax(dim=1).item()

        def forward_func(embeds):
            return self.model(inputs_embeds=embeds, **self._non_id_inputs(inputs, batch_size=embeds.shape[0])).logits

        ig = IntegratedGradients(forward_func)
        attributions, delta = ig.attribute(
            embeddings, baseline, target=target, n_steps=n_steps, return_convergence_delta=True
        )
        attributions = attributions[0].sum(dim=-1).detach()

        tokens = self._tokens(input_ids)
        return tokens, attributions, delta.item()

    def _transformer_layers(self):
        return self.model.base_model.encoder.layer

    def compute_activation_patch(self, clean_text: str, corrupted_text: str,
                                  layer: int, position: int) -> Dict[str, float]:
        """Activation patching (noising direction): cache the corrupted run's
        hidden state at (layer, position), splice it into the clean run at the
        same (layer, position), and measure how far the predicted class'
        probability moves from the clean value toward the corrupted value.
        clean_text and corrupted_text must tokenize to the same number of
        tokens, since position-based patching only makes sense when positions
        line up 1:1 between the two runs.
        """
        clean_inputs = self.tokenizer(clean_text, return_tensors="pt").to(device)
        corrupted_inputs = self.tokenizer(corrupted_text, return_tensors="pt").to(device)
        if clean_inputs["input_ids"].shape[1] != corrupted_inputs["input_ids"].shape[1]:
            raise ValueError(
                "clean_text and corrupted_text must tokenize to the same number of tokens "
                "for position-based activation patching."
            )

        with torch.no_grad():
            clean_logits = self.model(**clean_inputs).logits
            corrupted_logits = self.model(**corrupted_inputs).logits
        predicted_class = clean_logits.argmax(dim=-1).item()

        corrupted_value = self._get_hidden_state_at(
            corrupted_inputs["input_ids"], layer, position, **self._non_id_inputs(corrupted_inputs)
        )
        patched_outputs = self._patch_hidden_state_at(
            clean_inputs["input_ids"], layer, position, corrupted_value,
            **self._non_id_inputs(clean_inputs)
        )

        clean_prob = torch.softmax(clean_logits, dim=-1)[0, predicted_class].item()
        corrupted_prob = torch.softmax(corrupted_logits, dim=-1)[0, predicted_class].item()
        patched_prob = torch.softmax(patched_outputs.logits, dim=-1)[0, predicted_class].item()

        denom = clean_prob - corrupted_prob
        patch_effect = (clean_prob - patched_prob) / denom if abs(denom) > 1e-8 else 0.0

        return {
            "clean_prob": clean_prob,
            "corrupted_prob": corrupted_prob,
            "patched_prob": patched_prob,
            "patch_effect": patch_effect,
        }

class GenerativeVisualizer(BaseVisualizer):
    """Visualizer for generative models."""

    def _load_model(self) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            self.model_name, attn_implementation="eager"
        )

    def compute_saliency(self, prompt: str) -> Tuple[List[str], torch.Tensor]:
        self._reset_gradients()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embeddings = self._input_embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs))
        logits = outputs.logits

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        predicted_logit = logits[:, -1, next_token_id]
        predicted_logit.backward()

        gradients = embeddings.grad[0].abs().mean(dim=-1)
        saliency_map = gradients / gradients.max()

        tokens = self._tokens(input_ids)
        return tokens, saliency_map

    def _target_probability(self, inputs, embeddings, target=None):
        logits = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs)).logits
        probs = torch.softmax(logits[0, -1], dim=-1)
        if target is None:
            target = probs.argmax().item()
        return probs[target].item(), target

    def compute_attention(self, prompt: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        self._capture_layerwise(outputs.attentions, outputs.hidden_states)

        attention_weights = outputs.attentions[-1][0].detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self._tokens(input_ids)
        return tokens, avg_attention

    def compute_integrated_gradients(self, prompt: str, n_steps: int = 50) -> Tuple[List[str], torch.Tensor, float]:
        """Integrated Gradients attribution against a zero-embedding baseline,
        targeting the same next-token logit compute_saliency does. See
        ClassificationVisualizer.compute_integrated_gradients for the rationale."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embeddings = self._input_embeddings(input_ids)
        baseline = torch.zeros_like(embeddings)

        with torch.no_grad():
            target = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs)).logits[:, -1, :].argmax(dim=-1).item()

        def forward_func(embeds):
            return self.model(inputs_embeds=embeds, **self._non_id_inputs(inputs, batch_size=embeds.shape[0])).logits[:, -1, :]

        ig = IntegratedGradients(forward_func)
        attributions, delta = ig.attribute(
            embeddings, baseline, target=target, n_steps=n_steps, return_convergence_delta=True
        )
        attributions = attributions[0].sum(dim=-1).detach()

        tokens = self._tokens(input_ids)
        return tokens, attributions, delta.item()

    def compute_logit_lens(self, prompt: str) -> List[Tuple[str, float]]:
        """For each captured hidden-state layer, the top predicted next token
        (at the final input position) and its probability, obtained by
        projecting that layer's hidden state through the model's own final
        norm + unembedding - i.e. "what would this model predict if generation
        stopped here." Only guaranteed correct for GPT-2-style architectures
        with a `ln_f` final norm (skipped, not faked, for anything else). The
        final layer's result is not an approximation - it reconstructs the
        model's actual next-token logits exactly.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        unembed = self.model.get_output_embeddings()
        final_norm = getattr(self.model.base_model, "ln_f", None)
        last_layer = len(outputs.hidden_states) - 1

        results = []
        with torch.no_grad():
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                h = hidden[:, -1, :]
                # The last hidden_states entry already has the final norm
                # applied (GPT-2 appends it after ln_f), so apply it only to
                # the earlier, un-normed layers.
                if final_norm is not None and layer_idx != last_layer:
                    h = final_norm(h)
                probs = torch.softmax(unembed(h), dim=-1)
                top_prob, top_id = probs[0].max(dim=-1)
                results.append((self.tokenizer.decode(top_id), top_prob.item()))
        return results

    def _transformer_layers(self):
        return self.model.base_model.h

    def compute_activation_patch(self, clean_text: str, corrupted_text: str,
                                  layer: int, position: int) -> Dict[str, float]:
        """Activation patching (noising direction): cache the corrupted run's
        hidden state at (layer, position), splice it into the clean run at the
        same (layer, position), and measure how far the predicted next token's
        probability moves from the clean value toward the corrupted value.
        clean_text and corrupted_text must tokenize to the same number of
        tokens, since position-based patching only makes sense when positions
        line up 1:1 between the two runs.
        """
        clean_inputs = self.tokenizer(clean_text, return_tensors="pt").to(device)
        corrupted_inputs = self.tokenizer(corrupted_text, return_tensors="pt").to(device)
        if clean_inputs["input_ids"].shape[1] != corrupted_inputs["input_ids"].shape[1]:
            raise ValueError(
                "clean_text and corrupted_text must tokenize to the same number of tokens "
                "for position-based activation patching."
            )

        with torch.no_grad():
            clean_logits = self.model(**clean_inputs).logits[:, -1, :]
            corrupted_logits = self.model(**corrupted_inputs).logits[:, -1, :]
        predicted_token = clean_logits.argmax(dim=-1).item()

        corrupted_value = self._get_hidden_state_at(corrupted_inputs["input_ids"], layer, position)
        patched_outputs = self._patch_hidden_state_at(clean_inputs["input_ids"], layer, position, corrupted_value)

        clean_prob = torch.softmax(clean_logits, dim=-1)[0, predicted_token].item()
        corrupted_prob = torch.softmax(corrupted_logits, dim=-1)[0, predicted_token].item()
        patched_prob = torch.softmax(patched_outputs.logits[:, -1, :], dim=-1)[0, predicted_token].item()

        denom = clean_prob - corrupted_prob
        patch_effect = (clean_prob - patched_prob) / denom if abs(denom) > 1e-8 else 0.0

        return {
            "clean_prob": clean_prob,
            "corrupted_prob": corrupted_prob,
            "patched_prob": patched_prob,
            "patch_effect": patch_effect,
        }

class Seq2SeqVisualizer(BaseVisualizer):
    """Visualizer for sequence-to-sequence models."""

    def _load_model(self) -> PreTrainedModel:
        return AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, attn_implementation="eager"
        )

    def _decoder_start_ids(self, batch_size: int = 1) -> torch.Tensor:
        """Build a single-token decoder_input_ids seeded with the model's start token."""
        start_id = self.model.config.decoder_start_token_id
        if start_id is None:
            start_id = self.tokenizer.pad_token_id
        return torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)

    def compute_saliency(self, text: str) -> Tuple[List[str], torch.Tensor]:
        self._reset_gradients()
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embeddings = self._input_embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        decoder_input_ids = self._decoder_start_ids(batch_size=input_ids.shape[0])
        outputs = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs),
                              decoder_input_ids=decoder_input_ids)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Use the first predicted token for saliency
        predicted_logit = logits[:, 0, predicted_ids[0, 0]]
        predicted_logit.backward()

        gradients = embeddings.grad[0].abs().mean(dim=-1)
        saliency_map = gradients / gradients.max()

        tokens = self._tokens(input_ids)
        return tokens, saliency_map

    def _target_probability(self, inputs, embeddings, target=None):
        logits = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs),
                            decoder_input_ids=self._decoder_start_ids()).logits
        probs = torch.softmax(logits[0, 0], dim=-1)
        if target is None:
            target = probs.argmax().item()
        return probs[target].item(), target

    def compute_attention(self, text: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        decoder_input_ids = self._decoder_start_ids(batch_size=input_ids.shape[0])

        with torch.no_grad():
            outputs = self.model(**inputs,
                                  decoder_input_ids=decoder_input_ids,
                                  output_attentions=True, output_hidden_states=True)
        self._capture_layerwise(outputs.encoder_attentions, outputs.encoder_hidden_states)

        # Use encoder attention for input analysis
        attention_weights = outputs.encoder_attentions[-1][0].detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self._tokens(input_ids)
        return tokens, avg_attention

class TokenClassificationVisualizer(BaseVisualizer):
    """Visualizer for token classification models (e.g. NER, POS tagging)."""

    def _load_model(self) -> PreTrainedModel:
        return AutoModelForTokenClassification.from_pretrained(
            self.model_name, attn_implementation="eager"
        )

    def compute_saliency(self, text: str) -> Tuple[List[str], torch.Tensor]:
        self._reset_gradients()
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embeddings = self._input_embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs))
        logits = outputs.logits  # [1, seq_len, num_labels]
        predicted_labels = torch.argmax(logits, dim=-1)

        # Sum each token's own predicted-label logit and backprop once - gives
        # one saliency score per input token, reflecting its overall influence
        # on the full sequence-labeling decision rather than one label at a time.
        selected_logits = logits.gather(-1, predicted_labels.unsqueeze(-1)).squeeze(-1)
        selected_logits.sum().backward()

        gradients = embeddings.grad[0].abs().mean(dim=-1)
        saliency_map = gradients / gradients.max()

        tokens = self._tokens(input_ids)
        return tokens, saliency_map

    def _target_probability(self, inputs, embeddings, target=None):
        """Mean, over tokens, of each token's probability for its own
        predicted label - the per-token labels compute_saliency explains."""
        logits = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs)).logits
        probs = torch.softmax(logits[0], dim=-1)  # [seq_len, num_labels]
        if target is None:
            target = probs.argmax(dim=-1)
        return probs.gather(-1, target.unsqueeze(-1)).mean().item(), target

    def compute_attention(self, text: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        self._capture_layerwise(outputs.attentions, outputs.hidden_states)

        attention_weights = outputs.attentions[-1][0].detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self._tokens(input_ids)
        return tokens, avg_attention

class QuestionAnsweringVisualizer(BaseVisualizer):
    """Visualizer for extractive question-answering models. Takes a
    (question, context) pair rather than a single text string, so it overrides
    analyze/compute_saliency/compute_attention with that signature instead of
    forcing the single-string abstraction the other visualizers share."""

    def _load_model(self) -> PreTrainedModel:
        return AutoModelForQuestionAnswering.from_pretrained(
            self.model_name, attn_implementation="eager"
        )

    def compute_saliency(self, question: str, context: str) -> Tuple[List[str], torch.Tensor]:
        self._reset_gradients()
        inputs = self.tokenizer(question, context, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embeddings = self._input_embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs))
        start_idx = torch.argmax(outputs.start_logits, dim=-1)
        end_idx = torch.argmax(outputs.end_logits, dim=-1)

        # Backprop jointly from the predicted answer span's start + end logits.
        target = outputs.start_logits[0, start_idx] + outputs.end_logits[0, end_idx]
        target.backward()

        gradients = embeddings.grad[0].abs().mean(dim=-1)
        saliency_map = gradients / gradients.max()

        tokens = self._tokens(input_ids)
        return tokens, saliency_map

    def _target_probability(self, inputs, embeddings, target=None):
        """Probability of the predicted (start, end) span: p(start) * p(end)."""
        outputs = self.model(inputs_embeds=embeddings, **self._non_id_inputs(inputs))
        start_probs = torch.softmax(outputs.start_logits[0], dim=-1)
        end_probs = torch.softmax(outputs.end_logits[0], dim=-1)
        if target is None:
            target = (start_probs.argmax().item(), end_probs.argmax().item())
        return (start_probs[target[0]] * end_probs[target[1]]).item(), target

    def compute_attention(self, question: str, context: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(question, context, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        self._capture_layerwise(outputs.attentions, outputs.hidden_states)

        attention_weights = outputs.attentions[-1][0].detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self._tokens(input_ids)
        return tokens, avg_attention

    def analyze(self, question: str, context: str, save_dir: Optional[str] = None):
        """Perform comprehensive analysis for a (question, context) pair. Mirrors
        BaseVisualizer.analyze - duplicated rather than reused because the
        (question, context) signature doesn't fit the single-text abstraction."""
        self.attention_weights = []
        self.layer_outputs = []

        tokens, saliency_map = self.compute_saliency(question, context)
        print("\nSaliency Analysis:")
        save_path = f"{save_dir}/saliency_map.png" if save_dir else None
        self.plot_saliency(tokens, saliency_map, save_path=save_path)

        tokens, attention_weights = self.compute_attention(question, context)
        print("\nAttention Analysis:")
        save_path = f"{save_dir}/attention_map.png" if save_dir else None
        self.plot_attention(tokens, attention_weights, save_path=save_path)

        print("\nLayer-wise Attention Analysis:")
        save_path = f"{save_dir}/layer_attention.png" if save_dir else None
        self.plot_layer_attention(tokens, save_path=save_path)

        print("\nAttention Head Analysis:")
        save_path = f"{save_dir}/attention_heads.png" if save_dir else None
        self.plot_attention_heads(tokens, save_path=save_path)

        print("\nLayer Activation Analysis:")
        save_path = f"{save_dir}/layer_activations.png" if save_dir else None
        self.plot_layer_activations(save_path=save_path)

        print("\nGradient Flow Analysis:")
        save_path = f"{save_dir}/gradient_flow.png" if save_dir else None
        self.plot_gradient_flow(save_path=save_path)
