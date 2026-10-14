# Audit + extension plan (2026-10-14)

## Audit findings (all confirmed with real models)

1. Logit lens applies `ln_f` two times to the final layer. GPT-2's last `hidden_states` entry already has `ln_f`. On distilgpt2 the final-layer result is 0.063; the real prediction is 0.160. The test passes only because `tiny-gpt2` has an identity-like `ln_f`.
2. `_input_embeddings` omits Marian's `embed_scale` (sqrt(512)). The saliency path runs a different input than the model (max logit diff 13.9). This is the same bug class as ASSESSMENT bug #1. T5 does not scale, so the tests miss it.
3. QA drops `token_type_ids` in saliency and attention. BERT sees the context as segment A.
4. Generative saliency returns `[seq, hidden]`. All other visualizers return `[seq]`.
5. A single-token input makes `.squeeze()` return a string, not a token list.
6. Unused `AutoModelForMultipleChoice` import. Docs claim the logit lens is exact and bug #1 is resolved for all paths.

## Fix plan (test-first)

- [x] Logit-lens test: randomize `ln_f`, compare full logit vector. Confirm it fails.
- [x] Equivalence test for all visualizers: saliency path == `model(**inputs)`. Include a scaled-embedding Marian model. Confirm it fails.
- [x] Saliency-shape test for all visualizers. Confirm it fails for generative.
- [x] Fix 1-5. Confirm all tests pass.
- [x] Update CLAUDE.md, EXTENSION_PLAN.md, README if needed.
- [x] Run `main.py` end-to-end.

## Next extension slice

- [x] Ask the user which deferred item to build next. Answer: faithfulness metrics.

## Review (audit fixes)

- 30 tests pass (17 before; 13 new or changed). Each of the 5 bugs made at least one new test fail on the old code.
- `main.py` runs with no failures and writes 37 PNGs.
- The logit-lens test uses `distilgpt2`. `tiny-gpt2` (hidden size 2) cannot detect a double-applied LayerNorm.
- IG passes the extra inputs expanded to the Captum batch size. GPT-2 does not broadcast a batch-1 mask.
- Note: the generative IG delta on `distilgpt2` in `main.py` is -2.3e-3 at 50 steps. More steps reduce it.

## Faithfulness metrics: design

Metrics from DeYoung et al. 2020 (ERASER). `p` is the probability of the prediction on the full input. The target stays fixed to that prediction.

- comprehensiveness(k) = p(full) - p(input with the top-k tokens removed). High = the top tokens matter.
- sufficiency(k) = p(full) - p(input with only the top-k tokens kept). Low = the top tokens are enough.
- AOPC = the mean of each metric over the fractions (default 0.1, 0.2, 0.5).

Decisions:
- Removal sets the token's input embedding to zero. This is the same baseline that IG uses. It keeps the sequence length, so it works for all five visualizer types with no mask-token lookup.
- Special tokens ([CLS], [SEP], </s>) are never ranked and never removed.
- k = max(1, ceil(fraction x number of content tokens)) for fraction > 0, and k = 0 for fraction 0. Python's `round` gives k = 0 for 10% of 5 tokens.
- Special tokens come from `tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)`. Do not pass `return_special_tokens_mask=True`, because `_non_id_inputs` then sends that key to the model.
- Every probability (including p(full)) goes through the same `inputs_embeds` path, under `torch.no_grad()`. The method writes no shared state.
- Demo on `distilgpt2`. `tiny-random-bert` probabilities are near-uniform, so its scores are noise.
- Caveat: IG uses the same zero-embedding baseline as the removal, so IG gets a built-in advantage on these metrics. Ranking uses the given scores as-is (stable descending sort). For signed IG, positive scores support the prediction.

API:
- `BaseVisualizer.compute_faithfulness(attributions, *texts, fractions=(0.1, 0.2, 0.5))` returns per-fraction scores and AOPC values.
- Each subclass implements `_target_probability(inputs, embeddings, target=None) -> (prob, target)`. It uses the same target as its `compute_saliency`.
- `BaseVisualizer.plot_faithfulness(results, save_path=None)`.

Tests:
- [x] Mask test: a special token gets the highest score and one content token the next highest. Comprehensiveness must zero exactly that content position. Sufficiency must keep exactly that position plus the special tokens. This catches a reversed sort or a missing special-token offset.
- [x] Exact (`==`), every visualizer type: comprehensiveness(0.0) == 0, sufficiency(1.0) == 0, comprehensiveness(1.0) == sufficiency(0.0).
- [x] Close (`allclose`), every visualizer type: p(full) from `_target_probability` matches softmax of a plain `model(**inputs)`.
- [x] A length mismatch between attributions and tokens raises ValueError.

Steps:
- [x] Write the tests. Confirm they fail.
- [x] Implement. Confirm all tests pass.
- [x] Add a demo to `main.py`. Update CLAUDE.md, EXTENSION_PLAN.md, README.

## Review (faithfulness metrics)

- 46 tests pass. The 16 new faithfulness tests failed before the implementation.
- Mutation check: a reversed sort and a missing special-token offset each make the mask test fail.
- `main.py` runs with no failures and writes 40 PNGs. On `distilgpt2`: comprehensiveness AOPC is 0.113 (saliency), 0.121 (IG), and 0.073 (random).
- Two test assumptions were wrong: `tiny-random-bert` gives 18 content tokens for the test sentence, not 5. The tests now use the real count.
