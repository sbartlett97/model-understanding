from model_understanding.transformers_wrapper import (
    ClassificationVisualizer,
    GenerativeVisualizer,
    Seq2SeqVisualizer
)
import os

def main():
    # Create base visualization directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Create model-specific directories
    os.makedirs("visualizations/classification", exist_ok=True)
    os.makedirs("visualizations/generative", exist_ok=True)
    os.makedirs("visualizations/seq2seq", exist_ok=True)

    # Classification Example
    print("\nAnalyzing classification model...")
    classifier = ClassificationVisualizer("prajjwal1/bert-tiny")
    input_text = "The movie was absolutely fantastic!"
    classifier.analyze(input_text, save_dir="visualizations/classification")

    # Generative Example
    print("\nAnalyzing generative model...")
    generator = GenerativeVisualizer("distilgpt2")
    prompt = "Once upon a time in a land far away,"
    generator.analyze(prompt, save_dir="visualizations/generative")

    # Seq2Seq Example
    print("\nAnalyzing sequence-to-sequence model...")
    translator = Seq2SeqVisualizer("facebook/nllb-200-distilled-600M")
    input_text = "translate English to French: Hello, how are you?"
    translator.analyze(input_text, save_dir="visualizations/seq2seq")

if __name__ == "__main__":
    main()
