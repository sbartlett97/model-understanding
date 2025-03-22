from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForTokenClassification,
    AutoModelForMultipleChoice, PreTrainedModel, PreTrainedTokenizer
)
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
        self.setup_hooks()

    def setup_hooks(self):
        """Setup hooks to capture intermediate outputs."""
        def hook_fn(module, input, output):
            self.layer_outputs.append(output)
        
        def attention_hook_fn(module, input, output):
            # Debug print to understand the output structure
            print(f"Attention hook output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"Tuple length: {len(output)}")
                print(f"First element type: {type(output[0])}")
            
            # Handle different types of attention outputs
            if isinstance(output, tuple):
                # For attention outputs, we want the attention weights which are typically the first element
                if isinstance(output[0], torch.Tensor):
                    print(f"Found tensor with shape: {output[0].shape}")
                    self.attention_weights.append(output[0])
                elif isinstance(output[0], tuple) and isinstance(output[0][0], torch.Tensor):
                    print(f"Found nested tensor with shape: {output[0][0].shape}")
                    self.attention_weights.append(output[0][0])
            elif isinstance(output, torch.Tensor):
                print(f"Found direct tensor with shape: {output.shape}")
                self.attention_weights.append(output)

        # Register hooks for all transformer layers
        for name, module in self.model.named_modules():
            if "layer" in name and "attention" in name:
                print(f"Registering attention hook for: {name}")
                module.register_forward_hook(attention_hook_fn)
            elif "layer" in name:
                module.register_forward_hook(hook_fn)

    def clear_hooks(self):
        """Clear stored intermediate outputs."""
        self.layer_outputs = []
        self.attention_weights = []

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

        print(f"Number of attention weights collected: {len(self.attention_weights)}")
        for i, attn in enumerate(self.attention_weights):
            print(f"Layer {i+1} attention shape: {attn.shape}")

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
                print(f"Layer {i+1} averaged attention shape: {attn_avg.shape}")
                
                # Ensure the attention matrix is square and matches token length
                seq_len = min(len(tokens), attn_avg.shape[0], attn_avg.shape[1])
                attn_avg = attn_avg[:seq_len, :seq_len]
                
                sns.heatmap(attn_avg, xticklabels=tokens[:seq_len], 
                           yticklabels=tokens[:seq_len], 
                           cmap="Blues", ax=axes[i])
                axes[i].set_title(f"Layer {i+1} Attention")
            except Exception as e:
                print(f"Warning: Could not plot attention for layer {i+1}: {str(e)}")
                print(f"Attention tensor shape: {attn.shape}")
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
        
        # Create a grid of subplots
        n_rows = 2
        n_cols = (num_heads + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        axes = axes.flatten()
        
        for head in range(num_heads):
            # Get attention weights for this head
            head_attn = layer_attn[0, head].detach().cpu()  # [seq_len]
            
            # Create a square attention matrix by repeating the weights
            attn_matrix = torch.zeros((len(tokens), len(tokens)))
            attn_matrix.fill_diagonal_(head_attn[:len(tokens)])
            
            try:
                sns.heatmap(attn_matrix, xticklabels=tokens,
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

    def plot_prediction_confidence(self, tokens: List[str], logits: torch.Tensor, 
                                 top_k: int = 5, save_path: Optional[str] = None):
        """Visualize confidence scores for top-k predictions."""
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=range(top_k), y=top_probs[0].detach().cpu())
        plt.xticks(range(top_k), [self.tokenizer.decode(idx) for idx in top_indices[0]])
        plt.title("Top-k Prediction Probabilities")
        
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

    def analyze(self, text: str, save_dir: Optional[str] = None):
        """
        Perform comprehensive analysis for a given text.

        Args:
            text (str): Input text to analyze.
            save_dir (Optional[str]): Directory to save visualizations.
        """
        # Clear previous outputs
        self.clear_hooks()
        
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
        return AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def compute_saliency(self, text: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        embeddings = self.model.base_model.embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1)

        logits[0, predicted_class].backward()

        gradients = embeddings.grad.abs().mean(dim=-1).squeeze()
        saliency_map = gradients / gradients.max()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, saliency_map

    def compute_attention(self, text: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
        attention_weights = outputs.attentions[-1].squeeze().detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, avg_attention

class GenerativeVisualizer(BaseVisualizer):
    """Visualizer for generative models."""
    
    def _load_model(self) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(self.model_name)

    def compute_saliency(self, prompt: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        predicted_logit = logits[:, -1, next_token_id]
        predicted_logit.backward()

        gradients = embeddings.grad.abs().squeeze()
        saliency_map = gradients / gradients.max()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, saliency_map

    def compute_attention(self, prompt: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        outputs = self.model(input_ids, output_attentions=True)
        attention_weights = outputs.attentions[-1].squeeze().detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, avg_attention

class Seq2SeqVisualizer(BaseVisualizer):
    """Visualizer for sequence-to-sequence models."""
    
    def _load_model(self) -> PreTrainedModel:
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def compute_saliency(self, text: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        embeddings = self.model.get_encoder().embed_tokens(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Use the first predicted token for saliency
        predicted_logit = logits[:, 0, predicted_ids[0, 0]]
        predicted_logit.backward()

        gradients = embeddings.grad.abs().mean(dim=-1).squeeze()
        saliency_map = gradients / gradients.max()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, saliency_map

    def compute_attention(self, text: str) -> Tuple[List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
        # Use encoder attention for input analysis
        attention_weights = outputs.encoder_attentions[-1].squeeze().detach().cpu()
        avg_attention = attention_weights.mean(dim=0)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, avg_attention
