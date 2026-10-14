from model_understanding.transformers_wrapper import (
    ClassificationVisualizer,
    GenerativeVisualizer,
    Seq2SeqVisualizer,
    TokenClassificationVisualizer,
    QuestionAnsweringVisualizer,
)
import os
import torch

def main():
    # Create base visualization directory
    os.makedirs("visualizations", exist_ok=True)

    # Create model-specific directories
    for subdir in ("classification", "generative", "seq2seq", "token_classification", "question_answering"):
        os.makedirs(f"visualizations/{subdir}", exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Classification Example
    print("\nAnalyzing classification model...")
    try:
        classifier = ClassificationVisualizer("hf-internal-testing/tiny-random-bert")
        input_text = "The movie was absolutely fantastic!"
        classifier.analyze(input_text, save_dir="visualizations/classification")

        print("\nIntegrated Gradients (classification)...")
        tokens, attributions, delta = classifier.compute_integrated_gradients(input_text)
        print(f"IG convergence delta: {delta:.2e}")
        classifier.plot_saliency(
            tokens, attributions, title="Integrated Gradients",
            save_path="visualizations/classification/integrated_gradients.png"
        )

        print("\nAttention Rollout + entropy/distance (classification)...")
        rollout = classifier.compute_attention_rollout()
        classifier.plot_attention(
            tokens, rollout, title="Attention Rollout",
            save_path="visualizations/classification/attention_rollout.png"
        )
        entropy_distance = classifier.compute_attention_entropy_distance()
        classifier.plot_attention_entropy_distance(
            entropy_distance, save_path="visualizations/classification/attention_entropy_distance.png"
        )

        print("\nActivation Patching sweep (classification)...")
        patch_grid, patch_tokens = classifier.compute_activation_patch_sweep(
            "The movie was great", "The movie was awful"
        )
        classifier.plot_activation_patch_sweep(
            patch_grid, patch_tokens, save_path="visualizations/classification/activation_patch_sweep.png"
        )
    except Exception as e:
        print(f"Classification analysis failed: {e}")

    # Token Classification Example
    print("\nAnalyzing token classification model...")
    try:
        token_classifier = TokenClassificationVisualizer("hf-internal-testing/tiny-random-bert")
        input_text = "John lives in New York."
        token_classifier.analyze(input_text, save_dir="visualizations/token_classification")
    except Exception as e:
        print(f"Token classification analysis failed: {e}")

    # Question Answering Example
    print("\nAnalyzing question answering model...")
    try:
        qa = QuestionAnsweringVisualizer("hf-internal-testing/tiny-random-bert")
        question = "Where does John live?"
        context = "John lives in New York."
        qa.analyze(question, context, save_dir="visualizations/question_answering")
    except Exception as e:
        print(f"Question answering analysis failed: {e}")

    # Generative Example
    print("\nAnalyzing generative model...")
    try:
        generator = GenerativeVisualizer("distilgpt2")
        prompt = "Once upon a time in a land far away,"
        generator.analyze(prompt, save_dir="visualizations/generative")

        print("\nIntegrated Gradients (generative)...")
        tokens, attributions, delta = generator.compute_integrated_gradients(prompt)
        print(f"IG convergence delta: {delta:.2e}")
        generator.plot_saliency(
            tokens, attributions, title="Integrated Gradients",
            save_path="visualizations/generative/integrated_gradients.png"
        )

        print("\nFaithfulness: saliency vs Integrated Gradients vs random (generative)...")
        # Run on distilgpt2, not the tiny random checkpoints: their output
        # probabilities are near-uniform, so their faithfulness scores are
        # noise. A random ranking is the reference point - a useful
        # attribution has higher comprehensiveness and lower sufficiency.
        # IG gets a built-in advantage here: its zero-embedding baseline is
        # the same perturbation compute_faithfulness uses to remove a token.
        _, saliency = generator.compute_saliency(prompt)
        for name, scores in (("saliency", saliency),
                             ("integrated_gradients", attributions),
                             ("random", torch.rand(len(tokens)))):
            faithfulness = generator.compute_faithfulness(scores, prompt)
            print(f"{name}: comprehensiveness AOPC {faithfulness['comprehensiveness_aopc']:.3f}, "
                  f"sufficiency AOPC {faithfulness['sufficiency_aopc']:.3f}, "
                  f"k per fraction {[r['k'] for r in faithfulness['per_fraction']]}")
            generator.plot_faithfulness(
                faithfulness, title=f"Faithfulness ({name})",
                save_path=f"visualizations/generative/faithfulness_{name}.png"
            )

        print("\nLogit Lens (generative)...")
        logit_lens_results = generator.compute_logit_lens(prompt)
        generator.plot_logit_lens(
            logit_lens_results, save_path="visualizations/generative/logit_lens.png"
        )

        print("\nActivation Patching sweep (generative)...")
        # distilgpt2 is small/undertrained enough that its top next-token
        # prediction for this prompt is generic ("the"), not a confident
        # factual-recall answer, so the resulting signal is real but noisy -
        # this demonstrates the mechanism correctly without overclaiming a
        # clean result. Causal-tracing experiments in the literature typically
        # use much larger models for exactly this reason.
        patch_grid, patch_tokens = generator.compute_activation_patch_sweep(
            "Paris is the capital of", "London is the capital of"
        )
        generator.plot_activation_patch_sweep(
            patch_grid, patch_tokens, save_path="visualizations/generative/activation_patch_sweep.png"
        )
    except Exception as e:
        print(f"Generative analysis failed: {e}")

    # Seq2Seq Example
    print("\nAnalyzing sequence-to-sequence model...")
    try:
        translator = Seq2SeqVisualizer("Helsinki-NLP/opus-mt-en-fr")
        input_text = "Hello, how are you?"
        translator.analyze(input_text, save_dir="visualizations/seq2seq")
    except Exception as e:
        print(f"Seq2seq analysis failed: {e}")

if __name__ == "__main__":
    main()
