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

    def analyze(self, text: str, save_dir: Optional[str] = None):
        """
        Perform both saliency and attention analysis for a given text.

        Args:
            text (str): Input text to analyze.
            save_dir (Optional[str]): Directory to save visualizations.
        """
        tokens, saliency_map = self.compute_saliency(text)
        print("\nSaliency Analysis:")
        save_path = f"{save_dir}/saliency_map.png" if save_dir else None
        self.plot_saliency(tokens, saliency_map, save_path=save_path)

        tokens, attention_weights = self.compute_attention(text)
        print("\nAttention Analysis:")
        save_path = f"{save_dir}/attention_map.png" if save_dir else None
        self.plot_attention(tokens, attention_weights, save_path=save_path)

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
