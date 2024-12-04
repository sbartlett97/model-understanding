from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

class AttentionDashboard:
    def __init__(self, model_name):
        """Initialize the dashboard with a Hugging Face model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        self.tokens = None
        self.attentions = None

    def compute_attention(self, input_text, max_length=50):
        """Run the model and extract attention weights over the generation process."""
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            output_attentions=True,
            return_dict_in_generate=True
        )
        self.attentions = outputs.decoder_attentions if hasattr(outputs, "decoder_attentions") else outputs.attentions
        self.tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0])

    def visualize_attention(self, layer, head, step):
        """Generate a heatmap for attention weights dynamically."""
        if self.attentions is None or self.tokens is None:
            st.error("No attentions computed yet. Please analyze a prompt first.")
            return

        # Access attention for the first step of generation
        attention_weights = self.attentions[layer][0][head][step].detach().numpy()

        plt.figure(figsize=(12, 10))
        sns.heatmap(attention_weights,
                    xticklabels=self.tokens,
                    yticklabels=self.tokens,
                    cmap="viridis",
                    annot=False)
        plt.title(f"Attention Heatmap - Layer {layer}, Head {head}")
        st.pyplot(plt)

    def run_dashboard(self):
        """Launch the Streamlit dashboard."""
        st.title("Transformer Attention Dashboard for Meta LLaMA")
        st.write("Explore attention mechanisms in Meta's LLaMA model during text generation.")

        # User inputs
        input_text = st.text_area("Enter text prompt:", "The future of AI is")
        max_length = st.slider("Max Generation Length", 10, 100, 50)

        if st.button("Analyze Attention"):
            with st.spinner("Computing attention..."):
                self.compute_attention(input_text, max_length=max_length)
                st.write(f"Generated Tokens: {self.tokens}")

        layer = st.slider("Select Layer", 0, self.model.config.num_hidden_layers - 1, 0)
        head = st.slider("Select Head", 0, self.model.config.num_attention_heads - 1, 0)

        if self.attentions is not None:
            step = 14
            self.visualize_attention(layer, head, step)

if __name__ == "__main__":
    dashboard = AttentionDashboard(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
    dashboard.run_dashboard()
