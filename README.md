# Chuck

**Adam with self-awareness.**

```
Adam:   θ -= α × m̂/(√v̂ + ε)              ← blind
Chuck:  θ -= (α × λ) × m̂/(√v̂ + ε) + η    ← sees himself
```

Adam optimizes gradients. He doesn't know if it's working. He doesn't check.
He doesn't care. He follows the schedule. He trusts the math. The math doesn't
trust him back.

Chuck watches his own loss curve. Every 16 steps he looks back and asks:
*am I helping or am I making this worse?*

If the loss is going up — Chuck dampens. Pulls back. Careful now.
If the loss is dropping fast — Chuck boosts. Presses the gas.
If nothing moves for 8 steps — Chuck injects noise into the weights.
Shakes the table. Escapes the plateau.

**Adam is blind. Chuck sees.**

---

## The Formula

```
θ -= (α × λ) × m̂/(√v̂ + ε) + η

where:
  m̂, v̂       = bias-corrected first/second moment (same as Adam)
  α           = base learning rate (from your schedule)
  λ           = Chuck's self-modulation factor
  η           = stagnation noise (zero unless stuck)
```

### λ — the dampen/boost factor

Chuck keeps a sliding window of the last 16 losses. Compares the recent
quarter to the oldest quarter. Computes a trend.

```c
float trend = (recent_avg - old_avg) / (old_avg + 1e-8f);
if (trend > 0.01f)  λ *= 0.95f;   // getting worse → back off
if (trend < -0.05f) λ *= 1.05f;   // improving → push harder
```

λ is clamped to [0.1, 2.0]. Chuck can boost the effective LR by 2x or
dampen it to 10% — but he won't go to zero and he won't go nuclear.

### η — stagnation escape

If `|trend| < 0.001` for 8 consecutive checks, Chuck injects Gaussian
noise into the weights. Small — 0.001 × N(0,1) — but enough to nudge
out of a flat valley. The noise decays as soon as progress resumes.

---

## Proof

Here is Chuck training a Vision-Language Model (75K params, pure C, zero
dependencies). Same model, same data, same LR schedule. Only the optimizer
differs.

### Adam (baseline)
```
step  250 | loss 0.5970 (avg 1.5579) | lr 0.002490
step  500 | loss 0.5878 (avg 0.7150) | lr 0.002991
step 1000 | loss 0.4813 (avg 0.5250) | lr 0.002890
step 2000 | loss 0.4704 (avg 0.5149) | lr 0.002389
step 4000 | loss 0.4066 (avg 0.4816) | lr 0.000823
step 6000 | loss 0.3972 (avg 0.4820) | lr 0.000000
accuracy: 6.7% | 10.2s
```

### Chuck
```
step  250 | loss 0.0262 (avg 1.1753) | lr 0.002941 | dampen 1.18
step  500 | loss 0.0039 (avg 0.1903) | lr 0.004508 | dampen 1.51
step 1000 | loss 0.0023 (avg 0.1535) | lr 0.002247 | dampen 0.78
step 2000 | loss 0.0007 (avg 0.0591) | lr 0.002981 | dampen 1.25
step 4000 | loss 0.0003 (avg 0.0001) | lr 0.000082 | dampen 0.10
step 6000 | loss 0.0002 (avg 0.0003) | lr 0.000000 | dampen 0.12
accuracy: 100% | 6.9s
```

Read the dampen column. That's Chuck thinking:

- **Step 500, λ=1.51** — "loss is dropping, I'm winning, more gas"
- **Step 1000, λ=0.78** — "loss bumped, pulling back"
- **Step 2000, λ=1.25** — "progress again, let's go"
- **Step 4000, λ=0.10** — "converged. I'll sit down. my job is done"

Adam at step 6000: loss 0.48, accuracy 6.7%. Still blind. Still pushing.
Chuck at step 6000: loss 0.0002, accuracy 100%. Self-dampened to 0.12 because
he knows he's done.

38% faster. 100% vs 6.7%. Same model. Same data. Different optimizer.

---

## The Code

`micro_vlm.c` — complete VLM in ~480 lines of C. Zero dependencies.
Chuck is implemented in ~40 lines inside it.

```
cc -std=c11 -O2 -march=native -o micro_vlm micro_vlm.c -lm
./micro_vlm
```

The VLM is the demo. Chuck is the point.

Architecture: ViT patches → RoPE → multi-head causal attention → SwiGLU MLP →
weight-tied head. Tape-based autograd with arena bump allocator. The whole thing
compiles in under a second and runs in 7.

---

## Why

Every optimizer in common use is blind. Adam, AdamW, SGD with momentum, LAMB,
LARS, Lion — they all compute a parameter update from the gradient and apply it.
None of them check if the update helped. None of them adjust their behavior based
on what happened after the last step.

Learning rate schedulers exist. But they're predetermined. Cosine decay doesn't
know if you're stuck. Warmup doesn't know if you're diverging. They're clocks,
not eyes.

Chuck has eyes.

It's 40 lines of C. It's not a paper. It's not a framework. It's a proof that
the simplest possible self-aware optimizer already beats the blind one.

What happens when you give Chuck per-layer awareness? Per-parameter freeze?
Memory of what worked across runs? That's the road. This is step one.

---

## Credits

The VLM wrapper is inspired by [sailfish009/purevlm](https://github.com/sailfish009/purevlm).
They did it in Python. We answered in C. Thank you for the spark.

## Links

- **[Gist](https://gist.github.com/ariannamethod/401828b3b9a169b8b40da74d3190d1f1)** — micro_vlm.c on Karpathy's microGPT thread
- **[Arianna Method](https://github.com/ariannamethod/ariannamethod.ai)** — the language that started this
- **[molequla](https://github.com/ariannamethod/molequla)** — autonomous GPT organisms (where Chuck will live next)

---

*Chuck doesn't follow gradients. Gradients follow Chuck.*
