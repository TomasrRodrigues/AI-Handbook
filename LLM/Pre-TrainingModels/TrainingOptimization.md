# Training Optimization

#### Table of Contents

1. [The Optimization Challenge](#the-optimization-challenge)
2. [Optimization Algorithms](#optimization-algorithms)
3. [Learning Rate Schedules](#learning-rate-schedules)
4. [Gradient Management](#gradient-management)
5. [Mixed Precision Training](#mixed-precision-training)
6. [Training Stability and Best Practices](#training-stability-and-best-practices)

## The Optimization Challenge


Training a large language model represents one of the most expensive computational undertakings in modern machine learning. When Meta trained Llama 3 405B, they consumed approximately fifty-four days of continuous computation across thousands of GPUs. Given these staggering costs, even small improvements in optimization efficiency translate into substantial savings. A ten percent improvement in training efficiency could save millions of dollars and weeks of time. More importantly, optimization determines not just how fast a model trains, but whether it trains at all — poor optimization choices can cause training to diverge entirely, wasting the entire compute budget.

The optimization problem in LLM pre-training differs fundamentally from traditional machine learning in both scale and character. We are navigating a loss landscape with hundreds of billions of dimensions — one for each parameter. The sheer dimensionality makes intuition from lower-dimensional problems misleading. Moreover, this landscape exhibits complex structure: many local minima, saddle points, and substantial gradient noise computed from mini-batches that can mislead the optimizer if not handled carefully.

Unlike supervised learning where we might train for multiple epochs, LLM pre-training typically involves a **single pass** through enormous datasets of trillions of tokens. There are no repeated visits to examples that help the optimizer converge — each step must extract maximal learning from the available data.

The scale amplifies every optimization challenge:
- A learning rate that works perfectly for a small model might cause a large model to diverge catastrophically.
- Gradient clipping thresholds that stabilize training at one scale may be too restrictive or too permissive at another.
- Batch sizes that provide good gradient estimates for smaller models might waste computation on larger ones due to diminishing returns.

This means optimization strategies must be carefully reconsidered and re-tuned as we scale, making small-scale experiments only partially applicable.


### The Loss Landscape of Large Language Models

Understanding the loss landscape — the topology of the function mapping parameters to training loss — provides crucial intuition for why certain optimization techniques work.

**Non-convexity and saddle points.** The landscape has many local minima and saddle points. In high dimensions, saddle points vastly outnumber local minima, meaning an optimizer is far more likely to get stuck at a saddle point (where the gradient is zero but the point is not a minimum) than to converge to a local minimum. Fortunately, modern optimizers like Adam naturally escape saddle points through momentum, which accumulates past gradients and provides inertia to push through flat regions.

**Heterogeneous curvature.** Some dimensions of the parameter space have steep gradients while others are nearly flat. This heterogeneity motivates adaptive learning rate methods like Adam that adjust the learning rate per parameter based on historical gradient information. Without adaptation, a single global learning rate would be too large for some dimensions (causing instability) while being too small for others (slowing convergence).

**Sharp vs. flat minima.** Sharp minima correspond to parameter configurations where small perturbations significantly increase loss, while flat minima tolerate more variation. Extensive empirical work shows that **generalization correlates with flatness** — models converging to flatter minima tend to perform better on unseen data. This motivates weight decay, which implicitly biases the optimizer toward flatter regions, and informs learning rate schedule design, where decaying the learning rate late in training helps settle into flatter minima.

**Loss barriers.** These are regions of high loss separating different minima. Learning rate warmup helps navigate early barriers by allowing careful exploration, while learning rate decay at the end helps the optimizer descend carefully into minima without overshooting.

> Recent theoretical work on overparameterized networks suggests that with many more parameters than constraints, the landscape has many equivalent global minima connected by low-loss paths — making optimization easier. LLMs are massively overparameterized by this definition, which contributes to the remarkable success of relatively simple optimization algorithms.


### Scaling Laws and Compute-Optimal Training

Scaling laws provide quantitative relationships between model size, dataset size, and training compute that fundamentally shape optimization decisions. The **Chinchilla scaling laws** (2022) revolutionized thinking about LLM training by demonstrating that most LLMs were undertrained — using too few training tokens for their parameter count.

The key insight: for a fixed compute budget, there exists an **optimal allocation** between model size and training duration. Chinchilla found the compute-optimal ratio to be approximately **20 training tokens per model parameter**. A 100B parameter model should be trained on roughly 2 trillion tokens for optimal performance.

**Implications for optimization:**
- If training with **fewer tokens than compute-optimal** (data-constrained), use higher learning rates and more aggressive optimization to extract maximum learning from each token.
- If training with **more tokens than compute-optimal** (compute-constrained), use more conservative optimization with stronger regularization to prevent overfitting.

Scaling laws also reveal that larger models become **more sample-efficient** — extracting more learning per token. This typically enables larger peak learning rates and more aggressive decay, at the cost of increased instability risk.


### The Convergence-Stability Trade-off

Every optimization decision in LLM training involves balancing **convergence speed** against **stability**. This fundamental trade-off shapes every aspect of optimization design.

| Technique | Convergence benefit | Stability risk |
|---|---|---|
| Higher learning rate | Learns faster per update | May overshoot minima or diverge |
| Smaller batch size | More updates per compute; implicit regularization | Noisier gradients |
| Less gradient clipping | Preserves gradient information | Allows dangerous spikes through |
| Less weight decay | Faster convergence toward gradient signal | Risk of overfitting, sharp minima |

These trade-offs are not static — they evolve during training:
- **Early training**: stability is paramount. Warmup starts with a very low learning rate while the model is far from any good solution.
- **Late training**: convergence precision matters more. Learning rate decay helps the model settle carefully into a minimum.
- **At scale**: larger models tolerate higher learning rates due to overparameterization, but also face unique instabilities from numerical precision limitations.

> The goal is not to maximize instantaneous training speed, but to maximize the **probability of successfully training a good model** given finite resources and time.

## Optimization Algorithms

### AdamW

**AdamW** (Adam with decoupled weight decay) has emerged as the de facto standard optimizer for LLM pre-training. It combines two powerful ideas:

1. **Momentum** — an exponentially weighted average of past gradients that provides inertia, helping the optimizer maintain consistent update directions rather than thrashing between conflicting gradient signals.
2. **Adaptive learning rates** — per-parameter learning rates based on the history of squared gradients. Parameters with consistently large gradients receive smaller learning rates to prevent instability; parameters with small gradients receive larger rates to accelerate learning.

The algorithm maintains two state buffers per parameter:
- **First moment** $m$ — exponentially weighted average of gradients (momentum component)
- **Second moment** $v$ — exponentially weighted average of squared gradients (adaptive rate component)

At each step, both are updated with decay coefficients $\beta_1$ and $\beta_2$, bias-corrected, and used to compute the parameter update: the first moment divided by the square root of the second moment plus a small $\epsilon$ for numerical stability.

**The "W" distinction.** Original Adam implemented weight decay as L2 regularization added to the loss — meaning weight decay was scaled by the adaptive learning rate. This caused unintended interactions: parameters with small gradients had their weight decay amplified, while parameters with large gradients had it dampened. AdamW applies weight decay **directly to parameters** as a separate operation, making regularization behave consistently and predictably.

**Standard hyperparameters** (robust across models from millions to hundreds of billions of parameters):

| Hyperparameter | Typical value |
|---|---|
| $\beta_1$ | $0.9$ |
| $\beta_2$ | $0.95$ |
| $\epsilon$ | $1 \times 10^{-8}$ |
| Weight decay | $0.1$ |
| Peak learning rate | $1 \times 10^{-4}$ to $1 \times 10^{-3}$ |

**Limitations:**
- **Memory footprint** — storing two momentum buffers requires 3× the memory of the parameters themselves. For a 175B parameter model, this exceeds 1 TB just for optimizer states, motivating ZeRO sharding.
- **Gradient spikes** — when a mini-batch produces unusually large gradients, AdamW's second moment estimate can be corrupted, taking many steps to recover. Recent variants like **SPAM** address this by detecting spikes and resetting momentum when they occur.


### Lion

**Lion** (Evolved Sign Momentum) emerged from an automated algorithm discovery process. Its update rule is remarkably simple — simpler even than SGD with momentum — yet competitive with AdamW across many benchmarks while using **significantly less memory**.

Rather than maintaining two momentum buffers, Lion operates on a **single buffer**. At each step it:
1. Computes the element-wise sign of an interpolation between the current gradient and the momentum buffer.
2. Uses this sign as the update direction (scaled by the learning rate and weight decay).
3. Updates the momentum buffer as an exponentially weighted average of gradient and previous momentum.

**Primary advantage — memory efficiency.** With only one momentum buffer instead of two, Lion requires **half the optimizer state memory** of AdamW. For a 175B parameter model, this saves approximately 350 GB — substantial when operating near GPU memory limits.

**Key trade-offs vs. AdamW:**

| | AdamW | Lion |
|---|---|---|
| Optimizer states | 2 buffers | 1 buffer |
| Learning rate sensitivity | Forgiving | Requires careful tuning (~3–10× lower LR) |
| Weight decay | ~0.1 | ~0.01 (more aggressive by default) |
| Adoption at scale | Dominant | Mixed results; architecture-dependent |

In production settings where AdamW's robustness is well-established and memory is not the primary constraint, most organizations continue using AdamW as the conservative default.


### Sophia

**Sophia** introduces second-order curvature information into optimization while maintaining computational efficiency. Traditional second-order methods like Newton's method require computing and inverting the full Hessian — prohibitive for billions of parameters. Sophia achieves second-order benefits through a **diagonal approximation** that estimates curvature parameter-wise.

The algorithm maintains two state buffers:
- **Gradient momentum** — same as Adam's first moment.
- **Hessian diagonal estimate** — the expected squared gradient under the current data distribution, updated periodically (typically every few hundred steps) by sampling mini-batches and aggregating element-wise squared gradients.

Updates scale gradients by the **inverse of estimated curvature**: parameters with high curvature receive smaller effective learning rates; parameters with low curvature receive larger rates. This provides more accurate adaptation than Adam's gradient-history-based scaling, especially in regions with complex loss landscape geometry.

**Theoretical appeal vs. practical adoption:**

> While Sophia demonstrates competitive or superior final loss in some experiments — with training speedups on moderate-scale models — its adoption at production scale remains limited as of 2025. AdamW's well-established training recipes, the added complexity of tuning Hessian estimation frequency, and the conservative nature of frontier model training all favor sticking with proven approaches.

Sophia remains a promising research direction rather than a production-ready replacement for AdamW.


## Learning Rate Schedules

### Warmup

Learning rate warmup addresses a critical instability at the start of training. With randomly initialized parameters, the model makes wildly inaccurate predictions, producing large losses and consequently **large, noisy gradients**. Applying the full target learning rate immediately to these chaotic gradients can cause parameter updates to become enormous and destabilizing — in extreme cases, the loss explodes to infinity or collapses to NaN values that cannot be recovered.

Warmup solves this by **gradually increasing the learning rate** from near zero to its peak value over the first few thousand steps. The most common approach is **linear warmup**:

$$\text{lr}(t) = \text{lr}_\text{peak} \times \frac{t}{T_\text{warmup}}, \quad t < T_\text{warmup}$$

This gradual ramp gives the model time to move from random initialization toward a region where gradients are more stable and reliable.

**Warmup also serves a second purpose with Adam-family optimizers.** During the first steps, the second moment estimates — tracking squared gradient history — are still accumulating from a very limited history and are unreliable. Warmup reduces the impact of these unreliable estimates by keeping the learning rate small while they stabilize.

**Warmup duration guidelines:**

| Factor | Effect on warmup length |
|---|---|
| Larger model size | Longer warmup needed |
| Higher peak learning rate | Longer warmup needed |
| Larger batch size | Shorter warmup acceptable |

Typical durations range from **2,000 to 10,000 steps**. The computational cost is negligible compared to months-long training runs, making warmup nearly cost-free insurance against early instability.



### Cosine Decay

After warmup, most LLM training employs **cosine decay** to gradually reduce the learning rate over the remaining training duration. The schedule follows a cosine curve from the peak learning rate down to some minimum — typically 10% of the peak:

$$\text{lr}(t) = \text{lr}_\text{min} + \frac{1}{2}(\text{lr}_\text{peak} - \text{lr}_\text{min})\left(1 + \cos\left(\pi \cdot \frac{t - T_\text{warmup}}{T_\text{total} - T_\text{warmup}}\right)\right)$$

This smooth, monotonic decay has become the de facto standard, used by GPT, LLaMA, PaLM, and most other major models.

**Shape properties of the cosine curve:**
- **Early in decay** — learning rate decreases very slowly, allowing aggressive learning while gradients are large and the model is far from convergence.
- **Middle of training** — rapid decay accelerates convergence as the model enters the final approach to a good solution.
- **Late in training** — decay slows again, allowing careful refinement in a narrow learning rate range.

**Critical limitation — sensitivity to total duration.** Cosine decay must be tuned to the planned training length. If training stops early, the learning rate is still high and performance is left on the table. If training continues past the planned endpoint, the learning rate is stuck at its minimum where learning barely progresses. This makes cosine decay suboptimal for training runs whose duration is uncertain or may change.


### Warmup-Stable-Decay (WSD)

**WSD** addresses cosine decay's inflexibility by introducing a **stable plateau phase** between warmup and final decay:

1. **Warmup** — learning rate increases linearly to peak.
2. **Stable** — learning rate remains constant at peak for the bulk of training.
3. **Decay** — learning rate decreases to minimum in the final phase.

**Typical phase allocations:**

| Phase | Fraction of total steps |
|---|---|
| Warmup | 5–10% |
| Stable | 60–80% |
| Decay | 10–30% |

**Key advantages over cosine decay:**
- Training can be **extended into the stable phase** without degrading optimization quality.
- **Multi-stage training** (e.g., web data → curated data → domain data) becomes cleaner — each stage uses its own stable+decay schedule.
- The peak learning rate during the stable phase can be **tuned independently of total duration**, simplifying hyperparameter transfer between different-length runs.

Empirically, cosine decay often achieves slightly better final performance on well-planned runs where total duration is known precisely. However, for training requiring flexibility — uncertain compute availability, multi-stage pipelines, or potential extension — WSD provides substantial practical advantages with only modest performance trade-offs.


### Curriculum-Aware Learning Rate Design

Recent research has revealed a critical interaction between **data curriculum** (the order training examples are presented) and learning rate schedules that is often overlooked.

Standard learning rate schedules assume **constant data quality** throughout training. Cosine decay is highest when data quality is lowest (early training on web-scraped data) and lowest when data quality is highest (late training on curated data) — **exactly backwards** from what we want.

**Empirical findings:**

> With curriculum-based data ordering and aggressive learning rate decay (standard for uniform shuffling), the curriculum provides minimal improvement over random shuffling — sometimes only tenths of a percentage point. The same curriculum with **moderate decay** (ending 1–2 orders of magnitude higher) yields improvements of **1–2 percentage points** in downstream task accuracy.

**Recommendations for curriculum-aware training:**
- End learning rates should be substantially higher than the traditional near-zero target — perhaps $1 \times 10^{-3}$ instead of $1 \times 10^{-5}$.
- Shorten or delay the decay phase to maintain peak learning rate through more of the curriculum.
- Consider **checkpoint averaging** (CDMA) over the final high-quality phase rather than relying solely on the final checkpoint.

The core principle: optimal learning rate schedules should be **co-designed with data curricula** rather than selected independently.


## Gradient Management

### Gradient Clipping

Gradient clipping is one of the most critical stability techniques in LLM training, providing insurance against a catastrophic failure mode: **gradient explosion**, where gradients suddenly become enormous and permanently destroy the model's parameters.

**Why gradients explode.** During backpropagation through a deep network, gradients flow backward through each layer's Jacobian matrix. With dozens of layers, the total gradient involves a product of dozens of Jacobians. If these matrices have eigenvalues larger than one, the gradient grows **exponentially**. Even a value slightly above one, multiplied through forty layers, can become enormous. Occasionally unlucky data batches — containing long sequences with rare tokens the model handles poorly — can trigger this.

**Global norm clipping** is the standard solution. After computing all parameter gradients, we calculate the global gradient norm:

$$\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}$$

If this norm exceeds a threshold $\tau$ (typically 1.0), all gradients are scaled down uniformly:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\tau}{\|\mathbf{g}\|}$$

This preserves gradient **directions** while bounding their magnitude, and intervenes only when necessary — for the vast majority of steps where gradients are reasonable, clipping does nothing.

**Monitoring gradient norms** during training is highly informative:
- Norms **frequently hitting the threshold** → learning rate may be too high, or instability is present.
- Norms **consistently far below the threshold** → the threshold may be unnecessarily restrictive.

> **Mixed precision interaction:** In FP16 training with gradient scaling, gradients must be **unscaled before clipping** and re-scaled after. Modern frameworks handle this automatically, but understanding the interaction prevents subtle bugs.

### Gradient Accumulation

Gradient accumulation enables training with **effective batch sizes larger than GPU memory allows** by splitting a large batch into multiple micro-batches, processing each independently, and accumulating their gradients before applying a single optimizer update.

**Mechanics.** To achieve an effective batch size of 1,024 with only 64 examples fitting in memory: process 16 micro-batches of 64, accumulate gradients from each, then apply one averaged optimizer update. From the optimizer's perspective, this is mathematically equivalent to processing all 1,024 examples simultaneously.

**Benefits:**
- Decouples effective batch size from GPU memory capacity.
- In distributed training, **reduces communication frequency** — synchronize gradients once per accumulation cycle rather than after every micro-batch.
- Dramatically reduces **peak activation memory** versus a true large batch (storing activations for 64 examples instead of 1,024).

**Key trade-off — throughput.** Processing 16 sequential micro-batches requires 16 forward and 16 backward passes, which cannot be fully parallelized. A single true batch of the same size requires only one forward/backward pass with better GPU utilization.

> Gradient accumulation should be viewed as **enabling capabilities that would otherwise be impossible**, not as a throughput optimization.

**Learning rate scaling.** Larger effective batch sizes allow higher learning rates since gradients are more accurate. A common heuristic is **linear scaling**: doubling batch size allows doubling the learning rate — though this breaks down beyond the critical batch size where gradient noise becomes negligible.

### Gradient Checkpointing

Gradient checkpointing (also called activation recomputation) **trades computation for memory** by selectively discarding activations during the forward pass and recomputing them on-demand during backpropagation.

**The problem.** For each layer in the forward pass, we store activations needed to compute gradients via the chain rule. For a transformer layer processing 64 sequences × 4,000 tokens × 16,000 hidden dim in BF16, attention outputs alone require ~32 GB. Multiply by 50 layers and activation memory exceeds 1.5 TB — the dominant memory bottleneck.

**The solution.** Store only a subset of activations (the "checkpoints") and discard the rest. During backpropagation, recompute discarded activations from the nearest checkpoint when needed.

**Memory vs. compute trade-off:**

| Strategy | Memory reduction | Compute overhead |
|---|---|---|
| No checkpointing | None | None |
| Every N layers | ~75% (N=4) | ~33% extra |
| Optimal (every $\sqrt{L}$ layers) | Reduces to $O(\sqrt{L})$ | ~50% extra |
| Full recomputation | Maximum | ~100% extra |

**Selective checkpointing** improves on uniform strategies by considering each layer individually:
- **Attention layers** — high memory cost (large attention score matrices), low recomputation cost → good candidates for recomputation.
- **Feed-forward layers** — smaller activation memory, higher recomputation cost → better candidates for checkpointing outputs.

**FlashAttention** exemplifies this at the hardware level. Rather than materializing the full attention score matrix in HBM (memory quadratic in sequence length), FlashAttention fuses the entire attention computation into a single kernel that works in fast SRAM, recomputing scores during the backward pass. This achieves both memory savings and improved speed simultaneously.

## Mixed Precision Training

### FP16 and BF16

Mixed precision training enables models to train **faster and with less memory** by performing most computations in 16-bit rather than 32-bit floating point — achieving roughly 2× speedup and 2× memory reduction. The choice between the two dominant 16-bit formats has significant implications for numerical stability.

**Format comparison:**

| Property | FP32 | FP16 | BF16 |
|---|---|---|---|
| Total bits | 32 | 16 | 16 |
| Exponent bits | 8 | 5 | 8 |
| Mantissa bits | 23 | 10 | 7 |
| Max value | ~$3 \times 10^{38}$ | ~65,504 | ~$3 \times 10^{38}$ |
| Min positive | ~$1 \times 10^{-38}$ | ~$6 \times 10^{-5}$ | ~$1 \times 10^{-38}$ |

**The key trade-off:** FP16 offers better precision within a limited range; BF16 offers worse precision but matches FP32's range. For LLM training, **range generally matters more than precision** — very small gradients and moderately large activations can underflow or overflow in FP16 but remain representable in BF16.

This is why **BF16 has largely displaced FP16** as the preferred training format, particularly for large models where numerical stability is paramount. NVIDIA Ampere and newer architectures (A100, H100) provide native BF16 Tensor Core support with throughput matching FP16.

**Selective precision strategies** apply different precisions based on numerical sensitivity:
- **FP32**: LayerNorm, softmax, attention score computation.
- **BF16**: Matrix multiplications (dominant compute cost, safe in reduced precision).

### Loss Scaling

When training in FP16, many gradient values fall below the format's minimum (~$6 \times 10^{-5}$), causing them to **underflow to zero** and erasing optimization signal. Loss scaling addresses this by multiplying the loss by a large constant before backpropagation, proportionally scaling all gradients to remain representable.

**Mechanics:**
1. Multiply loss by scale factor $S$ (typically a power of 2, e.g., 512–65,536).
2. Compute backpropagation with scaled gradients.
3. **Unscale** gradients (divide by $S$) before optimizer update.
4. Check for overflow (inf or NaN) — if detected, **skip the update** and reduce $S$.

**Dynamic loss scaling** adaptively adjusts $S$ throughout training:
- **Overflow detected** → skip update, reduce $S$ by factor of 2.
- **N consecutive successful steps** → increase $S$ by a constant factor, recovering range utilization.

This automatically accommodates changing gradient magnitudes: early training with large gradients uses a smaller $S$; late training with small gradients uses a larger $S$.

> **BF16 eliminates the need for loss scaling** — its larger dynamic range matches FP32, removing one of the primary operational complexities of mixed precision training. This is a significant practical advantage of BF16 over FP16.

**Ordering requirement:** Gradients must be unscaled **before gradient clipping** — otherwise the clipping threshold would need to account for the current scale factor, creating a dependency on a dynamically changing value.


## Training Stability and Best Practices

Training stability encompasses all techniques ensuring a training run completes successfully without divergence. Understanding how individual techniques interact — and recognizing failure modes before they become catastrophic — is essential for production training.

### Failure Modes

**Loss divergence** is the most catastrophic failure: training loss jumps to infinity or NaN and cannot be recovered. Once parameters contain NaN values, all subsequent computations are corrupted irreversibly. Prevention requires:
- Gradient clipping to prevent explosion.
- Mixed precision with appropriate settings to avoid overflow.
- Warmup to prevent early instability.

**Loss spikes** — occasional substantial increases in loss before recovery — are less severe but still concerning:
- Small, infrequent spikes are generally harmless (noisy batches or transient numerical issues).
- Frequent or large spikes suggest underlying problems: learning rate too high, inadequate clipping, or data quality issues.


### Architectural Stability Factors

**Weight initialization** significantly affects early stability. Modern transformers use **variance-scaling initialization** where weight variance scales inversely with layer fan-in, ensuring activation magnitudes remain consistent across layers. Recent theoretical work suggests that **smaller-than-traditional initialization scales** can prevent loss spikes in very large models.

**Normalization placement** matters critically. **Pre-normalization** architectures (LayerNorm before each sub-layer) are generally more stable than post-normalization (after sub-layers), as they prevent activation magnitudes from growing unboundedly across layers.

**Attention logit capping** clips softmax inputs to a maximum magnitude (typically 20–30), preventing extreme probability distributions that can cause numerical issues in low precision. This has negligible impact on model quality.

### Monitoring and Diagnostics

Tracking key metrics throughout training provides early warning of problems before they become catastrophic:

| Metric | What it indicates |
|---|---|
| Gradient norm | Values frequently at clip threshold → potential instability |
| Training loss | Sudden increases → gradient spikes or data issues |
| Parameter norm | Unexpected growth → optimization diverging |
| Learning rate | Confirm schedule is applying correctly |

Automated systems can detect anomalies and automatically reduce learning rates, skip problematic batches, or revert to previous checkpoints when instability emerges.

### Defense-in-Depth

Production training pipelines implement **multiple stability layers** — even with gradient clipping, mixed precision, warmup, and careful hyperparameter tuning, rare edge cases can cause failures:

- **Robust checkpointing** ensures that training divergence loses at most a few hours rather than the entire run.
- **Alerting systems** notify engineers when metrics indicate problems, enabling manual intervention.
- **Distributed fault tolerance** excludes failed nodes automatically and continues with remaining healthy hardware.

> Conservative hyperparameters that sacrifice a few percent of training speed are worthwhile if they reduce catastrophic failure probability from 10% to 1%. The goal is not to maximize instantaneous training speed, but to **maximize the probability of successfully training a good model** given finite resources and time. Stability is the foundation on which efficient training is built.