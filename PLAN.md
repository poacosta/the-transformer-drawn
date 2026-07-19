# Animation Enhancement Plan

## Goal
Make the fixed-sentence Transformer pipeline easier to read by adding motion that shows *where data is moving* and *what is happening at each sub-step*, without changing the underlying toy math.

## Proposed changes

### 1. Traveling stage-to-stage flow lines
- Add an SVG overlay layer of animated paths that connect the active token up through the current stage (tokens → embedding → positional → QKV → attention → FFN → logits → output).
- Use `stroke-dashoffset` animation so a "pulse" dot travels from the active token to the current stage as the user steps forward.
- This answers the viewer question: "where does the data come from and where does it go?"

### 2. Progressive attention animation (steps 4–6)
Currently the attention stage shows all heads, scores, and weights at once. Split it into three beats:
- **Step 4 (Scores):** draw Query beams from the active token to allowed tokens, then Key beams from allowed tokens to the active token; reveal raw dot-product scores.
- **Step 5 (Softmax):** the same beams now carry the softmax percentage labels and pulse to show normalization.
- **Step 6 (Value aggregation):** add Value particles flowing back from each allowed token to the active token along the attention curves, weighted by attention strength (thicker/denser for high weights), then reveal the merged context vector.
This mirrors the explainer’s attention-matrix expansion but in the single-token view.

### 3. Smooth vector morphing and stage transitions
- Add CSS/SVG transitions on heat strips, token highlights, and stage groups so they fade/slide in rather than instantly appearing.
- When the active token changes, animate the token selection and redraw the attention curves instead of snapping.
- When the step advances, cross-fade the outgoing stage and incoming stage over ~400 ms.

### 4. Active-token query pulse
- During attention steps, add a subtle pulsing glow around the active token so viewers always know "this is the query position."

## Files to change
- `src/components/TransformerBoard.tsx` – add flow-line and attention-animation components, split AttentionStage into progressive sub-stages, add transition classes.
- `src/App.tsx` – no logic changes, but may add a wrapper class for transition orchestration.
- `src/index.css` – add keyframes for traveling dash, pulse glow, fade/slide transitions.

## Out of scope
- No real model weights or ONNX integration.
- No multi-block stacking; stays single-block.
- No changes to the toy math in `simulationData.ts`.

## Acceptance criteria
- Advancing a step shows a visible animated flow from the previous stage to the next.
- Attention steps clearly reveal Query → Key/Score → Softmax → Value in sequence.
- Changing the active token smoothly re-routes attention curves.
- All existing readouts and timeline behavior still work.
