import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, LlamaForCausalLM

device = torch.device("cuda")

#TODO: Finish condensing re-used code into a base class
class BaseVisualiser:
    model = None
    tokenizer = None

    def compute_saliency(self, text):
        pass


class ModelVisualizer:
    def __init__(self, model_name):
        """
        Initialize the visualizer with a pre-trained model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model (e.g., 'bert-base-uncased').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    def compute_saliency(self, text):
        """
        Compute the saliency map for a given input text.

        Args:
            text (str): Input text to analyze.

        Returns:
            tokens (list): List of tokens.
            saliency_map (torch.Tensor): Saliency scores for each token.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Extract embeddings from the model
        embeddings = self.model.base_model.embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_()

        # Forward pass
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1)

        logits[0, predicted_class].backward()

        gradients = embeddings.grad.abs().mean(dim=-1).squeeze()
        saliency_map = gradients / gradients.max()  # Normalize

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        return tokens, saliency_map

    def plot_saliency(self, tokens, saliency_map):
        """
        Plot the saliency map for the given tokens.

        Args:
            tokens (list): List of tokens.
            saliency_map (torch.Tensor): Saliency scores for each token.
        """
        plt.figure(figsize=(10, 5))
        plt.bar(tokens, saliency_map.detach().cpu().numpy())
        plt.xticks(rotation=90)
        plt.title("Saliency Map")
        plt.savefig("classification_saliency_map.png")
        plt.close()

    def compute_attention(self, text):
        """
        Compute attention weights for a given input text.

        Args:
            text (str): Input text to analyze.

        Returns:
            tokens (list): List of tokens.
            attention_weights (torch.Tensor): Average attention weights (seq_len x seq_len).
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
        attention_weights = outputs.attentions

        last_layer_attention = attention_weights[-1].squeeze().detach().cpu()

        avg_attention = last_layer_attention.mean(dim=0)  # (seq_len, seq_len)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        return tokens, avg_attention

    def plot_attention(self, tokens, attention_weights):
        """
        Plot attention weights as a heatmap.

        Args:
            tokens (list): List of tokens.
            attention_weights (torch.Tensor): Attention weights (seq_len x seq_len).
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=False)
        plt.title("Attention Weights")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.xticks(rotation=90)
        plt.savefig("classification_attention_fig.png")
        plt.close()

    def analyze(self, text):
        """
        Perform both saliency and attention analysis for a given text.

        Args:
            text (str): Input text to analyze.
        """
        tokens, saliency_map = self.compute_saliency(text)
        print("\nSaliency Analysis:")
        self.plot_saliency(tokens, saliency_map)

        tokens, attention_weights = self.compute_attention(text)
        print("\nAttention Analysis:")
        self.plot_attention(tokens, attention_weights)


class GenerativeModelVisualizer:
    def __init__(self, model_name):
        """
        Initialize the visualizer for generative models.

        Args:
            model_name (str): Name of the pre-trained model (e.g., 'gpt2').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: AutoModelForCausalLM | LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def compute_saliency(self, prompt):
        """
        Compute the saliency map for a given prompt in a generative model.

        Args:
            prompt (str): Input text prompt to analyze.

        Returns:
            tokens (list): List of tokens.
            saliency_map (torch.Tensor): Saliency scores for each token in the prompt.
        """
        # Tokenize input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        embedding_layer = self.model.get_input_embeddings()

        embeddings = embedding_layer(input_ids)

        embeddings.retain_grad()
        embeddings.requires_grad_()

        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        predicted_logit = logits[:, -1, next_token_id]

        next_token = self.tokenizer.decode(next_token_id)
        print(next_token)
        predicted_logit.backward()

        gradients = embeddings.grad.abs().squeeze()
        saliency_map = gradients / gradients.max()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        tokens += self.tokenizer.convert_ids_to_tokens([next_token_id])
        gen_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens))
        return tokens, saliency_map, gen_text

    def compute_attention(self, prompt):
        """
        Compute attention weights for a generative model prompt.

        Args:
            prompt (str): Input text prompt to analyze.

        Returns:
            tokens (list): List of tokens.
            attention_weights (torch.Tensor): Attention weights (seq_len x seq_len).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        outputs = self.model(input_ids, output_attentions=True)
        attention_weights = outputs.attentions

        last_layer_attention = attention_weights[-1].squeeze().detach().cpu()

        avg_attention = last_layer_attention.mean(dim=0)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        return tokens, avg_attention

    def plot_saliency(self, tokens, saliency_map, iteration):
        """
        Plot the saliency map for the given tokens.

        Args:
            tokens (list): List of tokens.
            saliency_map (torch.Tensor): Saliency scores for each token.
        """
        plt.figure(figsize=(10, 5))
        plt.bar(tokens, saliency_map.mean(dim=1).detach().cpu().numpy())
        plt.xticks(rotation=90)
        plt.title("Saliency Map")
        plt.savefig(f"generation_saliency_map_{iteration}.png")
        plt.close()

    def plot_attention(self, tokens, attention_weights):
        """
        Plot attention weights as a heatmap.

        Args:
            tokens (list): List of tokens.
            attention_weights (torch.Tensor): Attention weights (seq_len x seq_len).
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=False)
        plt.title("Attention Weights")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.xticks(rotation=90)
        plt.savefig("generation_attention_fig.png")
        plt.close()

    def analyze(self, prompt, max_new_tokens=50):
        """
        Perform both saliency and attention analysis for a given generative model prompt.

        Args:
            prompt (str): Input text prompt to analyze.
            max_new_tokens (int): Maximum number of generated tokens
        """
        saliency_maps = []
        token_history = []
        for _ in range(max_new_tokens):
            tokens, saliency_map, gen_text = self.compute_saliency(prompt)
            prompt = gen_text
            saliency_maps.append(saliency_map)
            token_history.append(tokens[:-1])
        print("\nSaliency Analysis:")
        for i in range(max_new_tokens):
            self.plot_saliency(token_history[i], saliency_maps[i], i)

        tokens, attention_weights = self.compute_attention(prompt)
        print("\nAttention Analysis:")
        self.plot_attention(tokens, attention_weights)
