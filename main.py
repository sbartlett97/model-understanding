import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import visualization as viz, LayerIntegratedGradients
import torch
import seaborn as sns
import matplotlib.pyplot as plt

class TransformerAttentionApp:
    def __init__(self):
        st.title("Transformer Attention and Embedding Visualization")
        self.initialize_session()

    def initialize_session(self):
        if 'model' not in st.session_state:
            st.session_state['model'] = None
        if 'tokenizer' not in st.session_state:
            st.session_state['tokenizer'] = None
        if 'attentions' not in st.session_state:
            st.session_state['attentions'] = None
        if 'tokens' not in st.session_state:
            st.session_state['tokens'] = None
        if 'input_text' not in st.session_state:
            st.session_state['input_text'] = ""

    def load_model(self, model_path):
        """Load model and tokenizer."""
        try:
            st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(model_path)
            st.session_state['model'] = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True).to("cuda")
            st.session_state['model'].eval()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    def compute_attention(self, input_text, max_new_tokens=50, temperature=1.0):
        """Generate output and capture attention states."""
        tokenizer = st.session_state['tokenizer']
        model = st.session_state['model']

        outputs = model.generate(
            input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_attentions=True,
            return_dict_in_generate=True
        )
        st.session_state['attentions'] = outputs.attentions
        st.session_state['tokens'] = tokenizer.convert_ids_to_tokens(outputs.sequences[0])

        # Decode and store raw output
        st.session_state['output_text'] = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    def visualize_attention(self):
        """Visualize attention weights for each layer and head."""
        attentions = st.session_state['attentions']
        tokens = st.session_state['tokens']

        if attentions is None or tokens is None:
            st.error("No attentions computed. Please run a sequence first.")
            return
        layer = st.slider("Select Layer", 0, st.session_state["model"].config.num_hidden_layers - 1, 0)
        head = st.slider("Select Head", 0, st.session_state["model"].config.num_attention_heads - 1, 0)

        attention_weights = attentions[layer][0][head].detach().cpu().numpy()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_weights[-1],
            xticklabels=tokens, 
            yticklabels=tokens, 
            cmap="viridis",
            annot=False
        )
        plt.title(f"Attention Heatmap - Layer {layer}, Head {head}")
        st.pyplot(plt)

    def visualize_word_importance(self):
        """Visualize word importance using LayerIntegratedGradients from Captum."""
        torch.cuda.empty_cache()
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']
        input_text = st.session_state['input_text']

        if input_text == "" or model is None or tokenizer is None:
            st.error("No input text or model to visualize. Please load a model and enter a sequence.")
            return

        def forward_func(inputs):
            return model(inputs).logits

        inputs = st.session_state["inputs"]
        lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
        attributions, _ = lig.attribute(inputs, return_convergence_delta=True)

        # Visualize word importance
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        plt.figure(figsize=(12, 8))
        sns.barplot(x=tokens, y=attributions_sum, palette="coolwarm")
        plt.title("Word Importance Visualization")
        plt.xlabel("Tokens")
        plt.ylabel("Attribution Scores")
        plt.xticks(rotation=90)
        st.pyplot(plt)

    def run(self):
        """Run the Streamlit app."""
        model_path = st.text_input("Enter the path to a Hugging Face model:", value="")

        if st.button("Load Model") and model_path:
            with st.spinner("Loading model..."):
                self.load_model(model_path)

        if st.session_state['model'] is not None:
            st.session_state['input_text'] = st.text_area("Enter text prompt:", st.session_state['input_text'])
            max_new_tokens = st.slider("Max New Tokens", 10, 100, 50)
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, step=0.1)

            if st.button("Generate Attention"):
                with st.spinner("Generating and computing attentions..."):
                    messages = [{"role": "user", "content": st.session_state['input_text']}]
                    input_text = st.session_state['tokenizer'].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = st.session_state['tokenizer'].encode(input_text, return_tensors="pt").to("cuda")
                    st.session_state["inputs"] = inputs
                    self.compute_attention(inputs, max_new_tokens=max_new_tokens, temperature=temperature)
                    st.write(f"Generated Output: {st.session_state['output_text']}")

            # Visualizations
            self.visualize_attention()
            self.visualize_word_importance()

if __name__ == "__main__":
    app = TransformerAttentionApp()
    app.run()