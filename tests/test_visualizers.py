import math

import pytest
import torch

from model_understanding import (
    ClassificationVisualizer,
    GenerativeVisualizer,
    Seq2SeqVisualizer,
    TokenClassificationVisualizer,
    QuestionAnsweringVisualizer,
)

# Tiny checkpoints kept small on purpose so the suite runs quickly and
# deterministically without pulling multi-GB weights.
CLASSIFICATION_MODEL = "hf-internal-testing/tiny-random-bert"
GENERATIVE_MODEL = "sshleifer/tiny-gpt2"
SEQ2SEQ_MODEL = "patrickvonplaten/t5-tiny-random"
# T5 does not scale its input embeddings, so it cannot catch a missing
# embed_scale. Marian does (scale_embedding=True). The tiny random Marian
# checkpoint does not load under transformers v5, so use the real one.
SCALED_SEQ2SEQ_MODEL = "Helsinki-NLP/opus-mt-en-fr"

# (visualizer class, checkpoint, positional inputs) for every visualizer type.
SALIENCY_CASES = [
    (ClassificationVisualizer, CLASSIFICATION_MODEL, ("A short test sentence.",)),
    (GenerativeVisualizer, GENERATIVE_MODEL, ("Once upon a time",)),
    (Seq2SeqVisualizer, SEQ2SEQ_MODEL, ("translate English to German: hello",)),
    (Seq2SeqVisualizer, SCALED_SEQ2SEQ_MODEL, ("Hello, how are you?",)),
    (TokenClassificationVisualizer, CLASSIFICATION_MODEL, ("John lives in New York.",)),
    (QuestionAnsweringVisualizer, CLASSIFICATION_MODEL, ("Where does John live?", "John lives in New York.")),
]
SALIENCY_IDS = ["classification", "generative", "seq2seq-t5", "seq2seq-marian", "token-classification", "qa"]


def _expected_pngs(save_dir):
    return [
        save_dir / "saliency_map.png",
        save_dir / "attention_map.png",
        save_dir / "layer_attention.png",
        save_dir / "attention_heads.png",
        save_dir / "layer_activations.png",
        save_dir / "gradient_flow.png",
    ]


def test_classification_analyze_completes(tmp_path):
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    viz.analyze("This is a test sentence.", save_dir=str(tmp_path))
    for png in _expected_pngs(tmp_path):
        assert png.exists(), f"expected {png} to be written"


def test_generative_analyze_completes(tmp_path):
    viz = GenerativeVisualizer(GENERATIVE_MODEL)
    viz.analyze("Once upon a time", save_dir=str(tmp_path))
    for png in _expected_pngs(tmp_path):
        assert png.exists(), f"expected {png} to be written"


def test_seq2seq_analyze_completes(tmp_path):
    viz = Seq2SeqVisualizer(SEQ2SEQ_MODEL)
    viz.analyze("translate English to German: hello", save_dir=str(tmp_path))
    for png in _expected_pngs(tmp_path):
        assert png.exists(), f"expected {png} to be written"


def test_token_classification_analyze_completes(tmp_path):
    viz = TokenClassificationVisualizer(CLASSIFICATION_MODEL)
    viz.analyze("John lives in New York.", save_dir=str(tmp_path))
    for png in _expected_pngs(tmp_path):
        assert png.exists(), f"expected {png} to be written"


def test_question_answering_analyze_completes(tmp_path):
    viz = QuestionAnsweringVisualizer(CLASSIFICATION_MODEL)
    viz.analyze("Where does John live?", "John lives in New York.", save_dir=str(tmp_path))
    for png in _expected_pngs(tmp_path):
        assert png.exists(), f"expected {png} to be written"


def test_classification_saliency_matches_plain_forward_pass():
    """Regression test for the embedding double-processing bug: the exact
    embedding tensor compute_saliency feeds into the model (via the shared
    ClassificationVisualizer._input_embeddings helper) must produce the same
    logits as a plain input_ids forward pass on the same text. Reverting that
    helper to `self.model.base_model.embeddings(input_ids)` must fail this."""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    text = "A short test sentence."
    inputs = viz.tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        reference_logits = viz.model(**inputs).logits

        embeddings = viz._input_embeddings(inputs["input_ids"])
        saliency_path_logits = viz.model(
            inputs_embeds=embeddings, attention_mask=inputs["attention_mask"]
        ).logits

        # The old, buggy embedding source (position encoding + LayerNorm applied
        # a second time by the model) is not a near-miss - it's a substantially
        # different tensor, which is why the bug corrupted saliency scores rather
        # than merely nudging them.
        double_processed_embeddings = viz.model.base_model.embeddings(inputs["input_ids"])

    assert torch.allclose(reference_logits, saliency_path_logits, atol=1e-4)
    assert not torch.allclose(embeddings, double_processed_embeddings, atol=1e-2)


@pytest.mark.parametrize("viz_cls, model_name, texts", SALIENCY_CASES, ids=SALIENCY_IDS)
def test_saliency_forward_pass_matches_plain_forward_pass(viz_cls, model_name, texts):
    """The forward pass that compute_saliency runs from inputs_embeds must give
    the same output as a plain forward pass on the full tokenizer output. This
    catches every way the two paths can differ: double-applied position
    encoding (ASSESSMENT bug #1), a missing embed_scale (Marian), or dropped
    tokenizer outputs such as token_type_ids (QA question/context pairs)."""
    viz = viz_cls(model_name)
    if model_name == SCALED_SEQ2SEQ_MODEL:
        assert viz.model.config.scale_embedding, "this case exists to cover scaled embeddings"

    captured = {}

    def hook(module, args, kwargs, output):
        if kwargs.get("inputs_embeds") is not None:
            captured["output"] = output[0].detach()

    handle = viz.model.register_forward_hook(hook, with_kwargs=True)
    try:
        viz.compute_saliency(*texts)
    finally:
        handle.remove()

    inputs = viz.tokenizer(*texts, return_tensors="pt")
    extra = {"decoder_input_ids": viz._decoder_start_ids()} if viz_cls is Seq2SeqVisualizer else {}
    with torch.no_grad():
        reference = viz.model(**inputs, **extra)[0]

    assert torch.allclose(captured["output"], reference, atol=1e-4), (
        f"max diff {(captured['output'] - reference).abs().max().item()}"
    )


@pytest.mark.parametrize("viz_cls, model_name, texts", SALIENCY_CASES, ids=SALIENCY_IDS)
def test_saliency_returns_one_score_per_token(viz_cls, model_name, texts):
    viz = viz_cls(model_name)
    tokens, saliency = viz.compute_saliency(*texts)
    assert isinstance(tokens, list)
    assert saliency.shape == (len(tokens),)


def test_single_token_input_returns_token_list():
    viz = GenerativeVisualizer(GENERATIVE_MODEL)
    tokens, saliency = viz.compute_saliency("Hello")
    assert tokens == ["Hello"]
    assert saliency.shape == (1,)


def test_gradients_do_not_accumulate_across_saliency_calls():
    """Regression test for the missing zero_grad bug: calling compute_saliency
    twice on identical input must not double the resulting gradients."""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    text = "A short test sentence."

    viz.compute_saliency(text)
    param = next(p for p in viz.model.parameters() if p.grad is not None)
    grad_after_first_call = param.grad.clone()

    viz.compute_saliency(text)
    grad_after_second_call = param.grad.clone()

    assert torch.allclose(grad_after_first_call, grad_after_second_call, atol=1e-6)


def test_integrated_gradients_satisfies_completeness_axiom():
    """Integrated Gradients must (approximately) satisfy the completeness
    axiom: sum(attributions) == F(input) - F(baseline). Captum's convergence
    delta directly measures the gap; empirically ~1e-8 at n_steps=50 for these
    tiny checkpoints, so 1e-3 is a generous but still meaningful bound."""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    tokens, attributions, delta = viz.compute_integrated_gradients("A short test sentence.")
    assert len(tokens) == attributions.shape[0]
    assert abs(delta) < 1e-3


def test_generative_integrated_gradients_satisfies_completeness_axiom():
    viz = GenerativeVisualizer(GENERATIVE_MODEL)
    tokens, attributions, delta = viz.compute_integrated_gradients("Once upon a time")
    assert len(tokens) == attributions.shape[0]
    assert abs(delta) < 1e-3


def test_logit_lens_final_layer_matches_real_prediction():
    """The logit lens result at the final captured layer must exactly
    reproduce the model's real next-token prediction - it reconstructs the
    same computation (final hidden state -> final norm -> unembedding) that
    produces the model's actual output logits, not an approximation of it.

    Uses distilgpt2, not tiny-gpt2: tiny-gpt2 has a hidden size of 2, and a
    LayerNorm over 2 values always gives +/-1, so applying ln_f twice changes
    almost nothing. That hid a real double-norm bug (GPT-2's last
    hidden_states entry already has ln_f applied)."""
    viz = GenerativeVisualizer("distilgpt2")
    prompt = "Once upon a time"

    results = viz.compute_logit_lens(prompt)

    inputs = viz.tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        real_logits = viz.model(**inputs).logits[0, -1, :]
    real_probs = torch.softmax(real_logits, dim=-1)
    top_prob, top_id = real_probs.max(dim=-1)
    real_token = viz.tokenizer.decode(top_id)

    final_token, final_prob = results[-1]
    assert final_token == real_token
    assert abs(final_prob - top_prob.item()) < 1e-5


def test_attention_rollout_rows_are_probability_distributions():
    """Attention rollout is a chain of renormalized convex combinations, so
    each output row must sum to 1 (a hard invariant of the construction) and
    the shape must match the token sequence length."""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    text = "A short test sentence."
    tokens, _ = viz.compute_attention(text)

    rollout = viz.compute_attention_rollout()

    assert rollout.shape == (len(tokens), len(tokens))
    assert torch.allclose(rollout.sum(dim=-1), torch.ones(len(tokens)), atol=1e-4)


def test_attention_entropy_matches_known_distributions():
    """Synthetic-tensor check (no model needed): uniform attention has entropy
    log(seq_len) exactly; one-hot (fully concentrated) attention has entropy 0
    exactly. These are hand-computable ground truths, not approximations."""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    seq_len = 4

    uniform = torch.full((1, 2, seq_len, seq_len), 1.0 / seq_len)
    viz.attention_weights = [uniform]
    uniform_results = viz.compute_attention_entropy_distance()
    assert all(abs(r["entropy"] - math.log(seq_len)) < 1e-5 for r in uniform_results)

    one_hot = torch.zeros((1, 2, seq_len, seq_len))
    one_hot[:, :, :, 0] = 1.0
    viz.attention_weights = [one_hot]
    one_hot_results = viz.compute_attention_entropy_distance()
    assert all(abs(r["entropy"]) < 1e-5 for r in one_hot_results)


def test_classification_activation_patch_identity_has_no_effect():
    """Regression test for the patching mechanism itself: patching a run with
    its OWN hidden state at every (layer, position) must leave the output
    exactly unchanged (torch.equal, not just close) - proves the forward hook
    only overwrites the intended slice and nothing else in the computation.
    Exercises _get_hidden_state_at/_patch_hidden_state_at directly (rather than
    via compute_activation_patch's clean==corrupted case, where clean_prob ==
    corrupted_prob makes patch_effect's zero-division guard return 0.0
    regardless of whether the hook actually did anything)."""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    text = "A short test sentence."
    inputs = viz.tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        clean_logits = viz.model(**inputs).logits

    num_layers = len(viz._transformer_layers())
    seq_len = input_ids.shape[1]
    for layer in range(num_layers):
        for position in range(seq_len):
            value = viz._get_hidden_state_at(input_ids, layer, position, attention_mask=attention_mask)
            patched = viz._patch_hidden_state_at(input_ids, layer, position, value, attention_mask=attention_mask)
            assert torch.equal(clean_logits, patched.logits)


def test_generative_activation_patch_identity_has_no_effect():
    viz = GenerativeVisualizer(GENERATIVE_MODEL)
    prompt = "The cat sat on the"
    input_ids = torch.tensor([viz.tokenizer(prompt)["input_ids"]])

    with torch.no_grad():
        clean_logits = viz.model(input_ids).logits

    num_layers = len(viz._transformer_layers())
    seq_len = input_ids.shape[1]
    for layer in range(num_layers):
        for position in range(seq_len):
            value = viz._get_hidden_state_at(input_ids, layer, position)
            patched = viz._patch_hidden_state_at(input_ids, layer, position, value)
            assert torch.equal(clean_logits, patched.logits)


def test_classification_activation_patch_sweep_completes(tmp_path):
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    grid, tokens = viz.compute_activation_patch_sweep("The movie was great", "The movie was awful")
    assert grid.shape == (len(viz._transformer_layers()), len(tokens))

    save_path = str(tmp_path / "activation_patch_sweep.png")
    viz.plot_activation_patch_sweep(grid, tokens, save_path=save_path)
    assert (tmp_path / "activation_patch_sweep.png").exists()


def test_generative_activation_patch_sweep_completes(tmp_path):
    viz = GenerativeVisualizer(GENERATIVE_MODEL)
    grid, tokens = viz.compute_activation_patch_sweep("The cat sat on the", "The dog sat on the")
    assert grid.shape == (len(viz._transformer_layers()), len(tokens))

    save_path = str(tmp_path / "activation_patch_sweep.png")
    viz.plot_activation_patch_sweep(grid, tokens, save_path=save_path)
    assert (tmp_path / "activation_patch_sweep.png").exists()


def test_activation_patch_rejects_mismatched_lengths():
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    try:
        viz.compute_activation_patch("short", "a much longer sentence than that", 0, 0)
        assert False, "expected ValueError for mismatched token lengths"
    except ValueError:
        pass


def _content_positions(viz, *texts):
    ids = viz.tokenizer(*texts)["input_ids"]
    special = viz.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
    return [i for i, s in enumerate(special) if not s], len(ids)


def test_faithfulness_removes_the_top_ranked_content_token():
    """Ranking and masking check. A special token gets the highest score and
    one content token the next highest. Comprehensiveness must zero exactly
    that content token's embedding; sufficiency must keep exactly that token
    plus the special tokens. A reversed sort, or a content-token index that
    skips the [CLS] offset, zeroes a different position and fails here. (The
    bookkeeping identities in the next test cannot catch either bug.)"""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    text = "A short test sentence."
    content, seq_len = _content_positions(viz, text)
    special = [i for i in range(seq_len) if i not in content]
    assert special == [0, seq_len - 1]

    top = content[1]
    scores = torch.linspace(0.0, 1.0, seq_len)
    scores[top] = 5.0
    scores[0] = 10.0  # special: must be ignored, not ranked first

    zeroed_sets = []

    def hook(module, args, kwargs, output):
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None:
            zeroed = (embeds[0] == 0).all(dim=-1).nonzero().flatten().tolist()
            zeroed_sets.append(set(zeroed))

    handle = viz.model.register_forward_hook(hook, with_kwargs=True)
    try:
        viz.compute_faithfulness(scores, text, fractions=(1 / len(content),))  # k = 1
    finally:
        handle.remove()

    assert {top} in zeroed_sets, f"comprehensiveness did not remove exactly position {top}: {zeroed_sets}"
    kept = {top, *special}
    assert set(range(seq_len)) - kept in zeroed_sets, f"sufficiency did not keep exactly {kept}: {zeroed_sets}"


@pytest.mark.parametrize("viz_cls, model_name, texts", SALIENCY_CASES, ids=SALIENCY_IDS)
def test_faithfulness_bookkeeping_identities(viz_cls, model_name, texts):
    """Exact identities that hold for any ranking: removing nothing changes
    nothing; keeping everything changes nothing; removing every content token
    and keeping no content token are the same input. All probabilities go
    through the same inputs_embeds path, so these are exact (==)."""
    viz = viz_cls(model_name)
    tokens, saliency = viz.compute_saliency(*texts)
    results = viz.compute_faithfulness(saliency, *texts, fractions=(0.0, 0.5, 1.0))

    by_fraction = {r["fraction"]: r for r in results["per_fraction"]}
    assert by_fraction[0.0]["comprehensiveness"] == 0.0
    assert by_fraction[1.0]["sufficiency"] == 0.0
    assert by_fraction[1.0]["comprehensiveness"] == by_fraction[0.0]["sufficiency"]
    assert by_fraction[0.0]["k"] == 0
    assert by_fraction[1.0]["k"] == results["num_content_tokens"]


@pytest.mark.parametrize("viz_cls, model_name, texts", SALIENCY_CASES, ids=SALIENCY_IDS)
def test_faithfulness_full_input_probability_matches_plain_forward_pass(viz_cls, model_name, texts):
    """p(full), the reference point for both metrics, must be the real model's
    probability for its own prediction. This crosses two code paths
    (inputs_embeds vs input_ids), so it is allclose, not exact."""
    viz = viz_cls(model_name)
    tokens, saliency = viz.compute_saliency(*texts)
    results = viz.compute_faithfulness(saliency, *texts)

    inputs = viz.tokenizer(*texts, return_tensors="pt")
    extra = {"decoder_input_ids": viz._decoder_start_ids()} if viz_cls is Seq2SeqVisualizer else {}
    with torch.no_grad():
        outputs = viz.model(**inputs, **extra)

    if viz_cls is QuestionAnsweringVisualizer:
        expected = (torch.softmax(outputs.start_logits[0], -1).max() * torch.softmax(outputs.end_logits[0], -1).max()).item()
    elif viz_cls is TokenClassificationVisualizer:
        expected = torch.softmax(outputs.logits[0], -1).max(dim=-1).values.mean().item()
    elif viz_cls is ClassificationVisualizer:
        expected = torch.softmax(outputs.logits[0], -1).max().item()
    else:  # generative: last position; seq2seq: first (only) decoder step
        expected = torch.softmax(outputs.logits[0, -1], -1).max().item()

    assert math.isclose(results["full_probability"], expected, abs_tol=1e-4)


def test_faithfulness_k_never_rounds_a_nonzero_fraction_to_zero():
    """round(fraction * n) gives k = 0 for a small fraction, which makes
    comprehensiveness read exactly 0 with no warning."""
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    text = "A short test sentence."
    content, _ = _content_positions(viz, text)
    fraction = 0.4 / len(content)  # round(0.4) == 0
    tokens, saliency = viz.compute_saliency(text)
    results = viz.compute_faithfulness(saliency, text, fractions=(fraction,))
    assert results["per_fraction"][0]["k"] == 1


def test_faithfulness_rejects_mismatched_attribution_length():
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    with pytest.raises(ValueError):
        viz.compute_faithfulness(torch.ones(3), "A short test sentence.")


def test_faithfulness_does_not_touch_shared_state():
    viz = ClassificationVisualizer(CLASSIFICATION_MODEL)
    text = "A short test sentence."
    viz.compute_attention(text)
    attention_before = list(viz.attention_weights)
    tokens, saliency = viz.compute_saliency(text)
    viz.compute_faithfulness(saliency, text)
    assert len(viz.attention_weights) == len(attention_before)
    assert all(a is b for a, b in zip(viz.attention_weights, attention_before))
