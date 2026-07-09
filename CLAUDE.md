# TransformerLens Practice — Learning Induction Heads

A self-paced course that teaches [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
through a single concrete goal: **find the induction heads in GPT-2 small and decompose the
full prev-token → induction circuit, proving the mechanism with ablation and logit attribution.**

## Learner profile

- Comfortable with PyTorch basics.
- Already understands attention internals (QKV, softmax, multi-head). We do **not** re-teach
  attention from scratch — we start at the TransformerLens mental model (hooks, activation cache,
  the residual stream).
- Goal is mechanistic interpretability, ending in a full circuit analysis.

## How we work (teach-then-practice)

Each notebook mixes two kinds of cells:

1. **Teaching cells** — fully runnable code I write, with English markdown explanations and English
   code comments. Run them top to bottom.
2. **Practice cells** — marked with `# TODO(you):`. You fill these in. When you're ready, ask me to
   review and I'll check your solution and explain anything off.

Rules of engagement:
- All explanations and comments in **English**.
- Prefer small, inspectable steps over big opaque helper functions — the point is to *see* activations.
- When a practice cell has a reference solution, I keep it in a collapsed/commented block below the
  TODO so you can self-check after trying.

## Environment (verified)

- Python 3.13, PyTorch 2.11 + CUDA (`cuda` available).
- `transformer_lens` and `circuitsvis` installed and working.
- GPT-2 small loads via `HookedTransformer.from_pretrained("gpt2")`
  → `n_layers=12`, `n_heads=12`, `d_model=768`, 144 attention heads total.

## Curriculum

| # | Notebook | Focus | Practice payoff |
|---|----------|-------|-----------------|
| 0 | `00_setup_and_model.ipynb` | Load GPT-2 small as `HookedTransformer`; tokenization; `model(tokens)` → logits; `.generate()`; reading `model.cfg` | Sanity-check the model predicts sensible next tokens |
| 1 | `01_hooks_and_cache.ipynb` | Core TL idea: `run_with_cache`, hook naming scheme, indexing the `ActivationCache`, the residual stream, pulling attention patterns out | Extract one specific head's attention pattern by hand |
| 2 | `02_attention_viz.ipynb` | Visualize attention with `circuitsvis`; learn to *read* what heads do; survey all 144 heads | Spot candidate heads visually |
| 3 | `03_induction_detection.ipynb` | Repeated-random-token sequences; per-position loss (the induction signature); why induction heads matter (in-context learning); write an **induction score** hook to rank all 144 heads | Your hook finds the real induction heads |
| 4 | `04_circuit_decomposition.ipynb` | prev-token head → induction head mechanism; **ablation** (zero/mean) to confirm necessity; **direct logit attribution**; K-composition | Ablate the circuit and watch induction break |

## Key concepts by notebook (reference map)

- **NB0:** `HookedTransformer`, `to_tokens` / `to_str_tokens`, logits shape `[batch, pos, d_vocab]`,
  greedy vs sampled generation, `cfg`.
- **NB1:** `run_with_cache`, hook names like `blocks.0.attn.hook_pattern`, `hook_resid_pre/mid/post`,
  `utils.get_act_name`, cache indexing `cache["pattern", layer]`, residual-stream read/write view.
- **NB2:** `circuitsvis.attention.attention_patterns`, interpreting attention as a `[head, dest, src]`
  tensor, previous-token / current-token / duplicate-token head shapes.
- **NB3:** constructing `[prefix, rand_seq, rand_seq]` inputs, per-token log-prob, the drop in loss on
  the *second* copy, induction attention offset (attend to the token *after* the previous occurrence),
  induction score aggregated per head → heatmap over layers × heads.
- **NB4:** prev-token heads in early layers feeding induction heads in later layers via K-composition;
  `HookPoint` ablation (patch activations to zero / dataset mean); direct logit attribution to measure
  each head's contribution to the correct next-token logit.

## Progress tracker

- [ ] NB0 — setup & model
- [ ] NB1 — hooks & cache
- [ ] NB2 — attention visualization
- [ ] NB3 — induction detection
- [ ] NB4 — circuit decomposition

(We check off a notebook once its practice cells are done and reviewed.)

## Conventions

- Notebooks live at repo root, numbered `NN_topic.ipynb`.
- Set `torch.set_grad_enabled(False)` for analysis notebooks (we're not training).
- Use a fixed seed where randomness matters (repeated-token sequences) for reproducibility.
- Device: `cuda`.
