# Critical Assessment

> **Status: resolved.** Findings #1–#5 below, and the dependency/README/testing gaps, were fixed in a follow-up pass (see `CLAUDE.md` for current architecture and `git log`/`git diff` for the actual changes). This document is kept as a point-in-time record of the original diagnosis, not a description of the current codebase.

This is an honest review of `model_understanding/transformers_wrapper.py` (the entire implementation) and how it's used from `main.py`. `torch`/`transformers` aren't installed in this environment and there's no venv checked in, so this isn't an empirical run — but every claim below that depends on current `transformers` internals (module names, tensor shapes, forward-signature behavior) was checked against the actual source on GitHub (`huggingface/transformers`, `main` branch: `modeling_bert.py`, `modeling_gpt2.py`, `modeling_m2m_100.py`), not assumed from memory. Confidence is noted per finding.

## Verdict

Tracing all three paths in `main.py` module-by-module: **classification runs, produces a saliency map that's silently wrong, then crashes**; **generative would likely complete but silently skips most of its extra plots**; **seq2seq crashes immediately, before a single plot is produced**. Because `main.py` has no `try/except` around any of the three `analyze()` calls, and classification runs first, none of this is visible today — running `python main.py` gets two PNGs out of classification (saliency + attention) and then dies with an unhandled exception. The generative and seq2seq examples never even execute. This reads like a project iterated on by patching symptoms as they were hit, rather than by tracing root causes.

## Bugs (highest confidence, highest impact first)

### 1. Classification saliency silently double-applies embeddings processing
`ClassificationVisualizer.compute_saliency` (transformers_wrapper.py:347):
```python
embeddings = self.model.base_model.embeddings(input_ids)
...
outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
```
`self.model.base_model.embeddings` is `BertEmbeddings` — it doesn't just do a word-embedding lookup, it also adds position/token-type embeddings and applies LayerNorm + dropout. When that fully-processed output is fed back in as `inputs_embeds`, `BertModel.forward` runs the *same* embeddings module again internally (`inputs_embeds` is supposed to be the raw word embedding, not the finished embeddings tensor) — adding position embeddings a second time and applying LayerNorm twice.

This doesn't crash. It silently changes the forward pass into something that doesn't correspond to how the model actually processes the input text, so the classification saliency map (and the `predicted_class` it's computed against) reflects a corrupted graph, not the real model. **Fix**: use `self.model.get_input_embeddings()(input_ids)` (raw word embeddings only) instead of `self.model.base_model.embeddings(input_ids)`.

Confidence: high (`BertEmbeddings.forward` unconditionally adds position embeddings and applies LayerNorm whenever it's called, whether given `input_ids` or a pre-computed `inputs_embeds`).

### 2. Seq2Seq path never supplies `decoder_input_ids` — confirmed crash on the first call
`Seq2SeqVisualizer.compute_saliency` (transformers_wrapper.py:429) calls:
```python
outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
```
with no `decoder_input_ids`, `decoder_inputs_embeds`, or `labels`. Checked against `modeling_m2m_100.py` (NLLB shares M2M100's architecture): `M2M100ForConditionalGeneration.forward` only derives `decoder_input_ids` from `shift_tokens_right(labels, ...)` when `labels is not None`; with none of the three supplied, `decoder_input_ids` stays `None` all the way down to `M2M100Decoder.forward`, which asserts `(input_ids is None) ^ (inputs_embeds is not None)` and raises `ValueError` when both are `None`. So this is not a "likely" bug — it's a guaranteed `ValueError` on the very first line of `compute_saliency`, before saliency, attention, or any plot is ever produced. `compute_attention` has the identical gap and would fail the same way if it were ever reached.

Separately, even patched, the example prompt is wrong for the chosen model: `"translate English to French: Hello, how are you?"` (transformers_wrapper.py / main.py:36) is a T5-style textual instruction prefix. NLLB doesn't take instructions in the input text — it needs `src_lang`/`tgt_lang` set on the tokenizer and a `forced_bos_token_id` for the target language passed at generation time. A "successful" run as written still wouldn't be doing a real translation-saliency analysis.

Confidence: high — traced against the current M2M100 forward chain (`M2M100ForConditionalGeneration.forward` → `M2M100Model.forward` → `M2M100Decoder.forward`).

### 3. Attention-head plot crashes on `fill_diagonal_`, taking classification's `analyze()` down with it
`plot_attention_heads` (transformers_wrapper.py:186-231):
```python
head_attn = layer_attn[0, head].detach().cpu()  # [seq_len]
attn_matrix = torch.zeros((len(tokens), len(tokens)))
attn_matrix.fill_diagonal_(head_attn[:len(tokens)])
```
`Tensor.fill_diagonal_` takes a **scalar** fill value, not a tensor. `head_attn[:len(tokens)]` has more than one element for any real input, so this call raises — it doesn't quietly draw a diagonal-only heatmap. And it sits *outside* the `try/except` a few lines below (which only wraps the `sns.heatmap` call), so the exception isn't caught — it propagates out of `plot_attention_heads`, out of `analyze()`, and (since `main.py` wraps none of its three `analyze()` calls in a `try/except`) crashes the whole script.

Concretely: this only crashes when `self.attention_weights` is non-empty, which — see #4 — is true for the classification (BERT) path but not the generative (GPT-2) path. So the actual failure sequence when running `python main.py` today is: classification's saliency + attention PNGs get written, then it dies here, inside `analyze()`'s "Attention Head Analysis" step — meaning `plot_layer_activations` and `plot_gradient_flow` never run for classification either, and the generative/seq2seq examples further down in `main()` never run at all.

Confidence: high (`fill_diagonal_`'s scalar-only contract is a documented, stable PyTorch API).

### 4. Hook-based attention capture doesn't capture attention weights, and matches inconsistently per architecture
`setup_hooks` (transformers_wrapper.py:59-64) registers hooks by substring-matching module names:
```python
for name, module in self.model.named_modules():
    if "layer" in name and "attention" in name:
        module.register_forward_hook(attention_hook_fn)
    elif "layer" in name:
        module.register_forward_hook(hook_fn)
```
Checked against current module-naming in each architecture actually used by `main.py`:

- **BERT** (`prajjwal1/bert-tiny`, classification): submodule names include `...layer.0.attention` (`BertAttention`) *and* `...layer.0.attention.self` (`BertSelfAttention`) — both contain `"layer"` and `"attention"`, so both get the attention hook. But `BertSelfAttention.forward` (checked in current `modeling_bert.py`) returns `(attn_output, attn_weights)` — a 2-tuple where **index 0 is the post-attention hidden state, not the attention weights** (the weights are at index 1, and can even be `None` under SDPA attention, which doesn't materialize them). `attention_hook_fn`'s tuple-handling branch always takes `output[0]`. So even the one submodule whose name is "correctly" matched has its hidden-state output captured and mislabeled as an attention weight — not a shape mismatch on the intended data, but the *wrong tensor entirely*. `...layer.0.attention.output` (`BertSelfOutput`, which returns a hidden state) also matches the substring filter and gets appended too. Net effect: `self.attention_weights` for BERT ends up full of hidden-state tensors, none of them actual attention probabilities.
- **GPT-2** (`distilgpt2`, generative): blocks live under `transformer.h.<i>` and the attention submodule is `transformer.h.<i>.attn` — neither `"layer"` nor `"attention"` appears as a substring anywhere in these names (checked against `modeling_gpt2.py`: the container attribute is literally named `h`, the attention attribute `attn`). **Neither hook branch ever fires.** `self.attention_weights` and `self.layer_outputs` stay empty for the entire run. This isn't a crash — `plot_layer_attention` and `plot_attention_heads` both guard on `if not self.attention_weights: print(...); return`, so they just print a "not available" message and skip — but it means two of the five plots `analyze()` claims to produce silently never get written for the generative path.
- **M2M100 / NLLB** (`facebook/nllb-200-distilled-600M`, seq2seq): attention submodules are named `self_attn` (checked in `modeling_m2m_100.py`), not `attention`, so the attention hook never fires here either — but the *generic* hook does, because layer container names look like `encoder.layers.0...` and `"layers"` contains the substring `"layer"`. So `self.layer_outputs` would get populated with real hidden-state tensors (correctly, this time) if this path ever ran a forward pass — which it doesn't, because of bug #2.

There's also no reason for this hook machinery to exist at all: `compute_attention` already gets real attention weights the documented, correct way (`output_attentions=True`, reading `outputs.attentions`), and that data isn't reused by `plot_layer_attention`/`plot_attention_heads` — they depend entirely on the broken hook path instead.

Confidence: high — all three architectures' module names and forward-return shapes were checked against current `transformers` source rather than assumed.

### 5. Gradients accumulate across repeated `analyze()` calls
No `zero_grad()` (model- or optimizer-level) is called anywhere in the file. `compute_saliency` calls `.backward()`, and `plot_gradient_flow` (transformers_wrapper.py:273-295) later reads `p.grad` off model parameters. `.eval()` doesn't stop gradients from being tracked or accumulated, so a second `analyze()` call on the *same* visualizer instance would have `.grad` values that are the sum of both calls' backward passes, silently corrupting the gradient-flow plot. `main.py` avoids this today only because it constructs three separate visualizer instances and calls `analyze()` once each — the bug is dormant, not fixed, and would surface immediately in any reuse scenario (a notebook, or the Streamlit UI implied by `requirements.txt` but never built).

Confidence: high.

## Design and code-quality issues

- **Debug `print()`s shipped in hook code** (transformers_wrapper.py:40-57): `attention_hook_fn` prints output type/shape info on every forward pass, for every module its filter matches. Fires on every `analyze()` call in the shipped example, with no logging module or verbosity flag — reads like debugging output that was never cleaned up.
- **Two independent, disagreeing mechanisms for "attention"**: the correct one (`output_attentions=True` in `compute_attention`) and the broken one (hooks, #4). There's no reason for the second to exist; deleting it and rebuilding `plot_layer_attention`/`plot_attention_heads` from `outputs.attentions` would fix #3 and #4 at once.
- **No `torch.no_grad()` anywhere**, including in `compute_attention`, which needs no gradients at all — every attention-only forward pass still builds a full autograd graph, wasted memory/compute once models are larger than the tiny examples used here.
- **Heavy duplication across the three visualizer subclasses.** `compute_saliency`/`compute_attention` in `ClassificationVisualizer`, `GenerativeVisualizer`, and `Seq2SeqVisualizer` repeat the same tokenize → embed → forward → backward → normalize skeleton three times with small variations. `BaseVisualizer` factors out plotting but not the gradient-saliency logic, which is where the real duplication (and duplicated bug surface, e.g. #1's embedding mistake could easily have been copy-pasted into the other two paths) lives.
- **Hardcoded batch-size-1 assumptions throughout** (`.squeeze()`, `[0, ...]`, `input_ids.squeeze().tolist()`). Reasonable for a single-example research script, but worth knowing before batching this.
- **Dead code**: `plot_prediction_confidence` is defined but never called from `analyze()` or anywhere else. It also calls `self.tokenizer.decode(idx)` on top-k *class* indices — meaningful for a generative/LM vocabulary, meaningless for `ClassificationVisualizer`, whose "vocabulary" is class labels, not tokens. It couldn't be dropped into that subclass's `analyze()` as-is even if someone wired it up.

## Dependencies and packaging

- `requirements.txt` lists `streamlit`, `captum`, `plotly`, `pandas`, and `tqdm` — **none of these are imported anywhere** in `main.py` or `model_understanding/`. Either there was unbuilt/abandoned scope (a Streamlit UI, Captum-based attribution) or the list was copied from a template; either way it overstates what the code does.
- Conversely, `facebook/nllb-200-distilled-600M` needs `sentencepiece` (and typically `protobuf`) for its tokenizer, and **neither is in `requirements.txt`** — a fresh `pip install -r requirements.txt && python main.py` is likely to fail on tokenizer load for the seq2seq example before it even reaches bug #2.
- No pinned upper bounds anywhere (`>=` only) — acceptable for a small research repo, but means behavior can silently shift under a `transformers` major bump. That's exactly the kind of thing that would break the hook name-matching in #4 again, in a different way, on a version upgrade.

## Documentation

- `README.md` documents a `ModelVisualizer` / `GenerativeVisualizer("gpt2")` API (`from model_understanding import ModelVisualizer`) that doesn't match the actual exported classes (`ClassificationVisualizer`, `GenerativeVisualizer`, `Seq2SeqVisualizer`). Following the README's usage instructions verbatim raises `ImportError`.
- README claims an MIT license and points to "the `LICENSE` file" — no `LICENSE` file exists in the repo.
- Nothing in the README or code signals the current broken state; a first-time user has no warning that two of the three documented examples don't run.

## Testing

There is no test suite — no unit tests for the saliency math, no smoke test asserting `analyze()` runs end-to-end on a tiny model, nothing that would have caught any of bugs #1–#4 before they shipped. Bug #1 is especially the kind of thing only a test would catch, since it doesn't crash — it just produces a plausible-looking but wrong number. A single regression test comparing `inputs_embeds`-based logits against a plain `input_ids` forward pass on the same text would have caught it immediately. A single smoke test per visualizer class (`analyze()` completes without exception on a tiny checkpoint) would have caught #2 and #3 immediately.

## Suggested priority if this is worth fixing

1. Fix #1 (embedding double-processing) and #2 (missing `decoder_input_ids`) — both corrupt or crash the primary code path.
2. Delete the hook-based attention/layer capture (#4) entirely; rebuild `plot_layer_attention` and `plot_attention_heads` from `outputs.attentions` (already computed correctly in `compute_attention`), the same way `plot_attention` does. This also removes #3's crash for free, since the diagonal-fill logic goes away with it.
3. Add `zero_grad()` before each `.backward()` (#5) and wrap attention-only forward passes in `torch.no_grad()`.
4. Trim `requirements.txt` to match actual imports, add `sentencepiece`/`protobuf` for the seq2seq path, and either fix the README's `ModelVisualizer` example or update it to the real class names.
5. Wrap `main.py`'s three `analyze()` calls so one model's failure doesn't hide the other two, and add one smoke test per visualizer class against a tiny checkpoint (`prajjwal1/bert-tiny`, `sshleifer/tiny-gpt2`, a tiny MarianMT model) asserting `analyze()` completes without exception — cheap to run in CI and would have caught #2 and #3 immediately.
