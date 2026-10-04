# The Transformer, Drawn

**An interactive working drawing of a GPT-style decoder block.** One sentence goes through one forward pass — tokens → embeddings → attention → feed-forward → next-token probabilities — drawn step by step in ten figures on a blueprint-style schematic, with every number on the sheet coming from a real (toy) computation you can follow by hand.

![Fig. 01 — the specimen sentence and its tokens, drawn on the sheet](docs/media/fig-01-tokens.jpg)

## Why this exists

Most transformer explainers live at one of two altitudes: architecture block diagrams (too high to see the arithmetic) or tensor equations (too low to see the story). This project sits deliberately in between. The model is shrunk until every value fits on one drawing — 8-dimensional vectors, 3 attention heads, a 5-word vocabulary — so you can watch the actual mechanics of next-token prediction: what a Query is *for*, why masking exists, what softmax does to scores, and how temperature reshapes the final distribution.

It's built for students meeting attention for the first time, and for anyone who has read ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) and still wants to *see* it.

## Quick start

```bash
npm install
npm run dev
```

Open `http://localhost:5173`. No API keys, no downloads, no GPU — the whole "model" is a few hundred lines of seeded arithmetic. `npm test` runs the checks that the chain is really connected end to end.

## The ten figures

| Fig. | Stage | What you see |
|------|-------|--------------|
| 01 | **Input tokens** | The sentence split into pieces, plus a dashed slot for the token to be predicted |
| 02 | **Embeddings** | Each token as an 8-cell heat strip (amber = positive feature, blue = negative) |
| 03 | **Positional encoding** | A wave pattern added so the two "the"s stop being identical |
| 04 | **Q, K, V** | Each combined vector, layer-normalized and projected into 4-wide Query / Key / Value lanes for the selected head |
| 05 | **Attention scores** | Each head's `Q·K⊤ / √d_k + b_head` comparisons drawn as arcs; future tokens masked |
| 06 | **Softmax weights** | Scores become percentages — per head, summing to 100% |
| 07 | **Value aggregation** | Weighted Value blends flow back along the arcs, are concatenated, projected by `W_O`, and added back to the token's vector (the first residual) |
| 08 | **Feed-forward network** | Normalize, expand 8 → 16 through GELU, contract back to 8, and add the second residual |
| 09 | **Vocabulary projection** | The last token's final vector, layer-normalized, scored against each candidate's embedding row → logits |
| 10 | **Final softmax** | Logits become probabilities; the title block resolves `PREDICT ?` |

Each figure carries its governing formula in the sheet's NOTES block (`score = Q·K⊤ / √d_k + b_head`, `p = softmax(logits / T)`, …), so the drawing maps directly onto the notation you'll meet in papers.

## Things to try

- **Write your own specimen.** Type any sentence (up to 6 words) in the header and hit *Redraw*. The simulation rebuilds deterministically — the same word always gets the same toy embedding, so you can compare how the same toy weights treat different sentences.
- **Move the query.** Click a different token and watch the causal mask shift with it: token *n* can only attend to tokens 0…n. Select the first token to see why it can only attend to itself.
- **Isolate one head.** On the attention figures, click a head in the legend. The other heads dim, and one head's "opinion" — near-neighbor syntax, subject linking, positional rhythm — becomes legible on its own. On Fig. 04 the selected head also chooses which Q/K/V lanes are drawn.
- **Turn the temperature dial.** On Fig. 10, drag T from 0.2 (greedy — on the default sentence `mat` climbs to 89%) to 3.0 (near-uniform — it drops to 29% and sampling gets adventurous). This is exactly the `temperature` parameter you set in LLM APIs.
- **Export the sheet.** The *Sheet* button downloads the current figure as a standalone SVG — useful for slides, handouts, or printing.
- **Drive it from the keyboard.** `←` / `→` step through figures, `Space` plays and pauses.

![Fig. 05 — attention scores for a custom sentence, one arc per head per allowed token](docs/media/fig-05-scores.jpg)

## What's real and what's toy

Educational honesty matters. This app computes one GPT-style decoder block for real, end to end, but it does not contain a trained model:

| Real | Toy |
|------|-----|
| The full chain, each stage computed from the previous stage's output: embedding + position → LayerNorm → per-head Q/K/V projections → scaled scores, causal mask, softmax → weighted values → concat · `W_O` → residual → LayerNorm → GELU MLP → residual → final LayerNorm → tied output logits → temperature softmax | Weights are seeded, not learned: every matrix is drawn deterministically from one global `MODEL_SEED`, so the same sentence always produces the same sheet |
| All of it is computed on the numbers shown — change a position vector, `W_O` or `W₁` and the logits change (the tests check exactly this) | `d = 8`, three heads of width 4 (so `d_k · h = 12 ≠ d`; real models usually use `d_k = d / h`), one block, five candidates |
| Pre-LN layout, as in GPT-2: LayerNorm before each sublayer, residual added after it | The positional "wave" is a simplified sinusoid, not the 2017 formula (and GPT-2 learns its positions instead) |
| The output layer is tied to the token embeddings, as in GPT-2: logits are dot products with each candidate's embedding row | Each head gets a small hand-written score bias (`b_head`: near-neighbor syntax, subject link, positional rhythm) so its pattern is easy to tell apart; set `TEACHING_BIAS = false` in `src/simulationData.ts` to remove it |
| Causal masking — future tokens genuinely never contribute; softmax outputs really sum to 100%; temperature really reshapes them | `MODEL_SEED` was chosen by a one-off search (the first seed where `mat` leads by at least 0.4 logits) so that the default sentence predicts `mat`; the candidates come from a fixed list (default sentence) or a word pool chosen by sentence hash |

A real GPT block runs the same chain with learned weights instead of seeded ones, and at a much larger scale: GPT-2 small, for example, uses 768-dimensional vectors, 12 heads per layer, a 50,257-token vocabulary and 12 stacked blocks (see the "×12" note under the title block). GPT-2 uses GELU in its feed-forward layers; the original 2017 Transformer used ReLU and placed layer normalization after each residual addition rather than before each sublayer.

## How the code is organized

```
src/
├── simulationData.ts        # The entire "model": seeded params + one Pre-LN block.
│                            #   buildSimulation(sentence) → every stage, every position.
├── simulationData.test.ts   # Determinism, shapes, softmax, causal mask, chaining (Vitest).
├── components/
│   └── TransformerBoard.tsx # The SVG sheet: one small component per figure, plus the
│                            #   grid, drawing frame, NOTES block, and live title block.
├── App.tsx                  # Lesson state (step, query token, solo head, temperature),
│                            #   header/sidebar/transport UI, keyboard controls.
├── exportSheet.ts           # Clones the SVG, inlines board CSS + fonts, downloads it.
└── index.css                # The design system: palette, type, layout, board styles.
```

There is no state library and no chart library — the board is hand-drawn SVG, which is the point: every mark on the sheet is a deliberate teaching decision.

**Stack:** React 19 · Vite · TypeScript · vanilla CSS · [Lucide](https://lucide.dev/) icons.

## Design language

The UI is styled as an engineer's working drawing — we do call it the transformer *architecture*, after all. Prussian drafting paper, a fine grid, chalk-white linework, and amber/rose/green annotation pencils for the attention heads. The title block in the sheet corner tracks lesson state in drawing-title-block vernacular (`FIG. 05/10 · SCORES`, `QUERY "sat"`, `PREDICT ?`), and each figure's formula appears in the NOTES block, the way real drawings carry general notes. Type is [Big Shoulders](https://fonts.google.com/specimen/Big+Shoulders), IBM Plex Sans, and IBM Plex Mono.

Accessibility floor: keyboard-driveable, visible focus states, `prefers-reduced-motion` respected.

## Going further

If this drawing clicked for you, these pair well with it:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original paper; Fig. 05–07 here are its Figure 2, animated
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's classic visual walkthrough
- [3Blue1Brown: Visualizing Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc) — the linear-algebra intuition
- [nanoGPT](https://github.com/karpathy/nanoGPT) — when you're ready to read a real trained decoder in ~300 lines of code

## License & Author

Created by [Pedro Acosta](https://github.com/poacosta). Released under the MIT License.
