from model_understanding import ModelVisualizer, GenerativeModelVisualizer

if __name__ == "__main__":
    # Classification Example
    visualizer = ModelVisualizer("bert-base-uncased")
    input_text = "The movie was absolutely fantastic!"
    visualizer.analyze(input_text)

    # Generative Example
    gen_visualizer = GenerativeModelVisualizer("HuggingFaceTB/SmolLM2-360M")
    prompt = "Once upon a time in a land far away,"
    gen_visualizer.analyze(prompt, max_new_tokens=10)
