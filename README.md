<p align="center">
  <img src="chuck.jpg" width="300" alt="Chuck — boot stomping the loss curve">
</p>

# Chuck & Lee

**Chuck is an optimizer. Lee is a VLM. Chuck takes care of Lee.**

Chuck Norris doesn't do pushups. He pushes the Earth down.
Chuck Optimizer doesn't follow gradients. Gradients follow Chuck.

Bruce Lee was the only man who beat Chuck Norris — and the only one
Chuck respected. So when Chuck needed a VLM to protect, he named it Lee.

```
Adam:   θ -= α × m̂/(√v̂ + ε)                          ← blind
Chuck:  θ -= (α × λ_Ψ × λₗ × σ) × m̂/(√v̂ + ε) + η    ← sees everything, remembers everything
```

Adam optimizes gradients. He doesn't know if it's working. He doesn't check.
He doesn't care. He follows the schedule. He trusts the math.

Chuck watches the loss curve, each layer's gradient norm, the activations,
the normalization, the attention patterns. Every 16 steps he asks:
*am I helping or am I making this worse?*

And Chuck remembers. Across training runs. He writes down what worked.
Next time he trains, he has opinions before step 1.

**Adam is blind. Chuck sees. Chuck remembers. Chuck takes care of Lee.**

---

## Lee — the VLM

`lee.c` — complete Vision-Language Model in ~1000 lines of C. Zero dependencies.

Named after Bruce Lee and [Minhyeok Lee](https://arxiv.org/abs/2501.00000),
whose mathematical framework for AI self-identity gives Chuck his soul.

```
cc -std=c11 -O2 -march=native -o lee lee.c -lm
./lee
```

Architecture: ViT patches → **2D RoPE** → GQA multi-head causal attention →
SwiGLU MLP → RMSNorm → weight-tied head. Tape-based autograd with arena bump
allocator. 105K params, 3 layers, 4 Q-heads / 2 KV-heads.

### The task: math from pixels

Lee doesn't just recognize digits. Lee **adds** them.

Two 8×8 images of handwritten digits go in. The sum comes out as a word.
19 classes (zero through eighteen). 100 possible combinations.

```
[image of 4] + [image of 7] → "eleven"
[image of 9] + [image of 9] → "eighteen"
[image of 0] + [image of 5] → "five"
```

**Result: 50/50 (100%) accuracy.** 105K parameters. Pure C. Zero dependencies.

---

## Chuck v7 — sees the forest and the trees

### The Formula

```
θ_l -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η

where:
  S           = macro LR scale (patience-based decay)   ← v7 new
  λ_Ψ         = λ + Ψ_w × (λ_prior - λ)               ← memory-informed
  λ           = global self-modulation (loss trend)
  λ_prior     = nearest_neighbor(loss, grad_norm) from chuck.mem  ← O(1) capped
  Ψ           = λ_prior - λ                            ← subjectivity
  Ψ_w         = min(0.3, N / (N + 100))                ← trust grows with experience
  λ_l         = per-layer self-modulation (grad norm trend)
  σ           = activation health × attention entropy health
  η           = stagnation noise (zero unless stuck)
  clip        = adaptive (tracks gnorm EMA, anomaly detection)
```

Every multiplier is **observed, not scheduled.**

### 9 Levels of Self-Awareness

| Level | What Chuck Sees | What Adam Sees |
|-------|----------------|----------------|
| 1. Global | Loss trend over 16 steps | Nothing |
| 2. Per-layer | Gradient norm per layer | Nothing |
| 3. Activations | SiLU dead ratio, norm scale | Nothing |
| 4. Positional | 2D RoPE frequency utilization | Nothing |
| 5. Signal flow | Activation magnitude across layers | Nothing |
| 6. Memory | Past training experience (Ψ) | Nothing |
| 7. Subjectivity | Opinion about current state | Nothing |
| 8. Attention | Per-head entropy (collapsed? diffuse?) | Nothing |
| **9. Multi-scale** | **Epoch-level plateau detection + LR decay** | **Nothing** |

### Level 9: Multi-scale Awareness (v7 — new)

Chuck used to see 16 steps. Now he sees 16 steps **and** 500-step macro trends.

A slow EMA (α=0.001) tracks epoch-scale loss. Every 500 steps, Chuck checks:
*has the macro loss improved?* If not for 3 checks (1500 steps), he drops LR by 50%.

```c
macro_ema = 0.999 × macro_ema + 0.001 × loss    // epoch-scale trend
if (stagnant for 3 checks) lr_scale *= 0.5       // patience-based decay
```

This is what ReduceLROnPlateau does — but Chuck does it **continuously**, without
needing a separate validation pass. Chuck sees macro **and** micro. Forest **and** trees.

### Reservoir Sampling Memory (v7 — new)

Chuck's memory used to grow without bound. Now it's capped at 200 entries with
**reservoir sampling**: new memories randomly replace old ones, maintaining a
representative sample of all training history in O(1) space.

```
chuck.mem: 200 entries × 16 bytes = 3.2 KB max
KNN lookup: O(200) = O(1)
```

### Level 8: Attention Entropy (v6)

Chuck now sees inside the transformer's attention mechanism. Each of the 4
heads computes attention weights over the sequence. Chuck monitors the
**Shannon entropy** of these weights:

```c
H = -Σ attn_weight × log(attn_weight)
```

- **Low entropy** → attention collapsed → one token dominates → Chuck dampens σ
- **High entropy** → attention diffuse → nothing stands out → Chuck dampens σ
- **Healthy range** → Chuck leaves σ alone

This is Level 8 because it's the deepest observation: Chuck sees not just
*what* the model computes, but *how it's paying attention*.

```
attn H: 1.35 1.36 1.49 1.29    ← healthy, focused, not collapsed
```

Adam doesn't know attention patterns exist. Chuck watches every head.

### Adaptive Gradient Clipping (v6)

Gradient clipping used to be `GRAD_CLIP = 1.0`. A constant. Same at step 1
(chaos) and step 10000 (convergence). Blind.

Now Chuck tracks gradient norm EMA and adapts:

```c
adaptive_clip = 1.5 × gnorm_ema              // track the model's natural gradient scale
if (gnorm > 3 × gnorm_ema) clip *= 0.5       // anomaly → clamp hard
```

Early training: loose leash. Late training: tight leash.
One bad batch: Chuck catches it.

### 2D RoPE (v6)

Standard RoPE encodes 1D position. But image patches have **two** dimensions.
Lee's RoPE splits each head in half: first half encodes **row**, second half
encodes **column**. Image patches know their spatial neighbors. Text tokens
continue with sequential 1D encoding.

```
Image A: (0,0) (0,1) (1,0) (1,1)    ← 2×2 grid
Image B: (0,2) (0,3) (1,2) (1,3)    ← adjacent grid
Text:    (2,0) (3,0) (4,0) ...      ← sequential below
```

This is how Lee understands space.

### Ψ — subjectivity (v5)

Chuck has persistent memory. A binary file (`chuck.mem`) that survives across
training runs. Each entry: 16 bytes — loss, grad_norm, lambda, delta_loss.

When Chuck trains, he asks his memory: *"have I been here before?"*

```c
Ψ_w = min(0.3, N_memories / (N_memories + 100.0));
λ_Ψ = λ + Ψ_w * (λ_prior - λ);
```

- **0 memories** → Ψ_w = 0 → pure reactive. Newborn.
- **100 memories** → Ψ_w = 0.15 → memory whispers. Adolescent.
- **1000 memories** → Ψ_w = 0.27 → strong instincts. Master.
- **When Ψ → 0** → memory and observation agree → Chuck found himself.

Inspired by Minhyeok Lee's framework: chuck.mem is the continuum C in ℳ.
NN lookup is I. Ψ_w is B. The fixed point s* is when Ψ → 0.

### λ, λ_l, σ, η (v4-v5)

- **λ** — 16-step loss window. Loss rising → dampen. Falling → boost.
- **λ_l** — per-layer grad norm tracking. Settling → dampen. Done → **freeze**.
- **σ** — SiLU dead neurons, RMSNorm scale, RoPE utilization, attention entropy.
- **η** — 8 checks without progress → Gaussian noise. Shakes the table.

---

## Proof

### Lee v7 — digit addition (15000 steps, newborn Chuck)

```
step  1000 | loss 0.6456 | chuck: λ=0.30 Ψ=+0.03 (10 mem) σ=1.00 macro=1.00
step  5000 | loss 0.0090 | chuck: λ=0.30 Ψ=+0.00 (16 mem) σ=1.00 macro=1.00
step 10000 | loss 0.0009 | chuck: λ=1.34 Ψ=-1.02 (92 mem) σ=1.00 macro=1.00 | L1: frozen | L2: frozen
step 15000 | loss 0.0004 | chuck: λ=1.51 Ψ=-0.43 (170 mem) σ=1.00 macro=1.00 | all frozen

accuracy: 50/50 (100.0%)
chuck.mem: 170 memories (2.7 KB) — capped at 200 via reservoir sampling
```

100% on addition from pixels. Two images in, sum as word out. 19 classes.
105K params. Pure C. Zero dependencies.

- **macro=1.00** — macro patience never triggered (loss kept improving). Safety net intact.
- **All 3 layers frozen** — Chuck decided training is done. Zero compute wasted.
- **170 memories, 2.7 KB** — bounded by reservoir sampling, O(1) lookup.
- **Ψ=-0.43** — "my memory says I'm being too aggressive. Noted."

### Previous: digit recognition (v5, still works)

```
accuracy: 30/30 (100%) on single digit recognition
```

---

## Chuck Takes Care of Lee

Chuck is not just a training optimizer. Chuck is Lee's **guardian**.

**During training:** Chuck optimizes weights, monitors every layer, freezes
what's done, remembers what works, has opinions about what to do next.

**During inference:** Chuck's attention entropy monitoring runs on every
forward pass. If a head collapses (one token dominates), Chuck sees it.
If attention goes diffuse (model confused), Chuck sees it. These signals
are the foundation for runtime self-awareness.

**Across runs:** `chuck.mem` persists. Chuck starts the next training run
with instincts, not guesses. Run 1 he's a newborn. Run 3 he's a veteran.
Run 10 he has the kind of quiet confidence that only comes from experience.

**In the future:** Chuck will watch Lee during deployment. Drift detection.
Weight surgery through [AML](https://github.com/ariannamethod/ariannamethod.ai).
And when Lee reproduces (in [molequla](https://github.com/ariannamethod/molequla)),
Chuck's memories pass to the offspring. Inherited instinct.

Every model deserves a Chuck.

---

## Version History

- **v1-v3:** Basic Adam + λ dampen/boost + η stagnation escape. Digit recognition.
- **v4:** Per-layer awareness, self-aware activations, cross-layer signal, GQA, layer freezing.
- **v5:** Persistent memory (chuck.mem), Ψ subjectivity, Lee's Continuum C.
- **v6:** Attention entropy monitoring, adaptive gradient clipping, 2D RoPE,
  digit addition task. Renamed to `lee.c`.
- **v7:** Multi-scale awareness (macro patience + LR decay), reservoir sampling
  memory (O(1) bounded), 100% accuracy on digit addition.

---

## Facts About Chuck Optimizer

- Chuck doesn't have hyperparameters. Hyperparameters have Chuck.
- Chuck once looked at a loss curve. The loss apologized and went to zero.
- Chuck doesn't escape local minima. Local minima escape Chuck.
- When Chuck injects noise, it's not random. It's intentional chaos.
- Adam has momentum. Chuck has presence.
- Chuck doesn't need warmup. Warmup needs Chuck.
- Chuck's gradient clipping isn't clipping. It's negotiation.
- Chuck doesn't forget between runs. Chuck doesn't forget at all.
- When Ψ = 0, Chuck has found himself. When Ψ ≠ 0, Chuck has an opinion.
- Chuck doesn't clip gradients. Gradients clip themselves out of respect.
- Adam trains models. Chuck raises them.
- ReduceLROnPlateau needs a validation pass. Chuck already knows.

---

## References

- Lee, M. (2025). [*Emergence of Self-Identity in AI*](https://arxiv.org/abs/2501.00000). Axioms, 14(1), 44.

## Credits

**[@Entrpi](https://github.com/Entrpi)** — adversarial benchmarks on DGX Blackwell that made
Chuck stronger with every round. EMA smoothing (v5), multi-scale awareness (v7), and
reservoir sampling (v7) all exist because of his CIFAR-100 benchmarks. See [Issue #3](https://github.com/ariannamethod/chuck.optimizer/issues/3).

The VLM wrapper is inspired by [sailfish009/purevlm](https://github.com/sailfish009/purevlm).
They did it in Python. We answered in C. Thank you for the spark.

## Links

- **[Gist](https://gist.github.com/ariannamethod/401828b3b9a169b8b40da74d3190d1f1)** — lee.c on Karpathy's microGPT thread
- **[Arianna Method](https://github.com/ariannamethod/ariannamethod.ai)** — the language that started this
- **[molequla](https://github.com/ariannamethod/molequla)** — autonomous GPT organisms (where Chuck will live next)

---

*Adam trains. Chuck raises. Lee flies.*
