# Monitoring for LLM Pre-Training

#### Table of Contents

1. [The Monitoring Challenge](#the-monitoring-challenge)
2. [Loss-Based Monitoring](#loss-based-monitoring)
3. [Gradient Monitoring](#gradient-monitoring)
4. [Internal State Monitoring](#internal-state-monitoring)
5. [Checkpoint Evaluation](#checkpoint-evaluation)
6. [Practical Monitoring Infrastructure](#practical-monitoring-infrastructure)
7. [Putting It All Together](#putting-it-all-together)


## The Monitoring Challenge

Training a large language model is expensive, time-consuming, and fraught with potential failure modes. When Meta trains Llama 3 405B for 54 days on 16,000 H100 GPUs, the stakes are enormous. A training run that diverges halfway through wastes weeks of compute time and millions of dollars. A run that completes successfully but produces a model with degraded capabilities represents a more subtle but equally costly failure.

Unlike supervised learning where you can continuously check accuracy on a held-out test set, LLM pre-training operates in a fundamentally different regime. The model is trained with a simple next-token prediction objective on a massive corpus of text, but its ultimate value lies in how well it performs on downstream tasks that won't be evaluated until after training completes - or at best, at expensive intermediate checkpoints.

This creates a profound monitoring challenge. The training objective - cross-entropy loss on next-token prediction - can continue decreasing smoothly while the model's actual capabilities on downstream tasks plateau, regress, or exhibit wild instability. A model can achieve low perplexity on its training data while failing to generalize. Training can appear stable by conventional metrics while internal degradation is occurring that will only become apparent much later.

Traditional machine learning monitoring focuses on tracking training and validation loss, watching for overfitting when the two curves diverge. This approach breaks down for LLMs in several ways:
- Pre-training typically involves a **single pass** through trillions of tokens, so there's no repeated replay that causes classic overfitting.
- The validation set comes from the **same distribution** as training data - it's just held-out web text - so validation loss tracks training loss closely and doesn't directly measure downstream performance.
- The **sheer scale** means that comprehensive evaluation on downstream benchmarks is expensive, involving instruction fine-tuning at each checkpoint.

> The fundamental tension in LLM monitoring is between **cost and signal quality**. Training loss and perplexity can be computed essentially for free as byproducts of training, updated every single step. Downstream task evaluation provides direct measurement of capabilities you care about but can only be justified every few thousand or tens of thousands of steps.

Recent research has revealed that the relationship between training loss and downstream performance is far more complex than previously assumed. The phenomenon of **grokking** - where generalization on downstream tasks improves long after training loss has plateaued - demonstrates that the training objective is a lagging indicator of actual capability acquisition. Moreover, this memorization-to-generalization transition doesn't happen uniformly:

- Different domains (math, code, commonsense reasoning, specialized knowledge) can undergo their grokking transitions at different times.
- A model's **average performance might be improving** while specific critical capabilities are degrading.
- Aggregate metrics can be deeply misleading.

This creates a hierarchical monitoring strategy: **continuous tracking of inexpensive metrics** (loss, gradients, simple statistics) with **occasional deeper dives** (checkpoint evaluation, downstream benchmarks) to ensure the cheaper metrics remain meaningful.


## Loss-Based Monitoring

The most fundamental monitoring signal in any neural network training is the loss function itself. For LLM pre-training, this is the **cross-entropy loss on next-token prediction**. Given a sequence of tokens $x_1, x_2, \ldots, x_n$, the model predicts the probability distribution over possible next tokens at each position, and the loss measures how much probability mass the model assigned to the actual next token:

$$\ell = -\sum_i \log P(x_i \mid x_1, \ldots, x_{i-1})$$

summed over all tokens in the sequence and averaged across the batch.

In a typical training run, loss begins high (around 10–11 for models with a 50,000-token vocabulary, corresponding to random guessing) and decreases steadily as the model learns patterns. The exact trajectory depends on learning rate schedule, model architecture, and data quality, but successful runs show a smooth, monotonic decrease following predictable **scaling laws**: larger models trained on more data achieve lower final loss values, with the relationship following power laws in model size, dataset size, and compute budget.

### Training Loss vs. Validation Loss

**Validation loss** provides a critical complement by measuring performance on held-out data. In classic machine learning, divergence between training and validation loss signals overfitting. For LLMs, this picture is more nuanced:

- Because pre-training typically involves a **single epoch**, the model sees each example once - classical overfitting doesn't apply.
- Training and validation losses tend to **track each other closely**, both decreasing together.
- Validation loss serves more to confirm that training data is **representative of the broader distribution** than to detect overfitting.

**Patterns that warrant attention:**
- *Validation loss consistently much higher than training loss* - suggests data distribution mismatch, contamination, or overfitting to training-specific patterns.
- *Validation loss lower than training loss* - may indicate the validation set is artificially easy or that there's leakage between sets.

### Perplexity

**Perplexity** transforms cross-entropy loss into a more interpretable metric:

$$\text{Perplexity} = e^{\text{loss}}$$

It represents the **effective number of tokens the model is uncertain between** at each prediction step. A perplexity of 100 means the model is as uncertain as if choosing uniformly among 100 equally likely tokens. Modern LLMs achieve perplexities in the **range of 10–30** on held-out web text, depending on data quality and model size.

> Perplexity amplifies differences - a change in loss from 3.0 to 2.9 corresponds to perplexity dropping from 20.1 to 18.2, making relative improvements more visible.

Lower perplexity generally correlates with better language modeling capability, but the relationship to downstream task performance is **imperfect**:
- A model can achieve very low perplexity by memorizing frequent patterns while still struggling with complex reasoning.
- Training interventions that improve downstream accuracy (curriculum learning, data filtering) might slightly *increase* perplexity if they remove highly predictable but low-value data.

### Interpreting Loss Curve Patterns

| Pattern | Likely cause |
|---|---|
| Sudden spike (loss jumps from 2.5 → 5.0) | Gradient explosion or numerical overflow - requires immediate intervention |
| Loss completely flat for thousands of steps | Stuck training: vanishing gradients or learning rate near zero |
| Large oscillations (±1.0 or more) | Learning rate too high, poor batch sampling, or data quality issues |
| Small oscillations (±0.1–0.2) | Acceptable noise |
| Loss plateau despite continued training | Model may be approaching capacity limits - but grokking research warns this doesn't mean learning has stopped |

> **Critical limitation:** Loss only measures the model's ability to predict the next token on its training distribution. It says nothing about the model's ability to follow instructions, write code, or reason - the tasks that determine its practical value.


## Gradient Monitoring

While loss tracks the model's predictions, gradients track how the model is trying to improve those predictions. Monitoring gradient statistics provides **early warning of training instabilities** that might not yet be visible in the loss curve - gradients can spike suddenly before the loss reacts, vanish in ways that compromise learning, or exhibit patterns that reveal optimization difficulties.

### Global Gradient Norm

The most commonly monitored gradient statistic is the **global gradient norm** - the L2 norm across all parameters:

$$\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}$$

**Typical ranges:**
- Early training: norms in the range of 10–100 as the model's predictions are poor.
- Converged training: norms often settle in the range of 0.1–10 for well-tuned runs.

The gradient norm is crucial for detecting **exploding gradients** - one of the most common causes of training failure. In a 40-layer transformer, backpropagation involves 40 matrix multiplications. If the spectral norm of any Jacobian matrix exceeds 1, gradients can grow **exponentially**. A sudden spike (say, from 2.0 to 200.0 in a single step) signals extreme gradients that will corrupt parameters if applied directly.

**Gradient clipping** addresses this: if the computed norm exceeds a threshold $\tau$ (typically 1.0), all gradients are scaled down proportionally:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\tau}{\|\mathbf{g}\|}$$

This preserves gradient **directions** while bounding their magnitude, intervening only when necessary.

**Monitoring clipping frequency:**

| Fraction of steps clipped | Interpretation |
|---|---|
| >50% | Learning rate too high, or persistent instability |
| 5–20% | Healthy - threshold is catching occasional spikes |
| <1% | Threshold may be unnecessarily restrictive |

The inverse problem - **vanishing gradients** - is less common in modern transformers but still worth monitoring. If norms fall below $10^{-6}$, gradients are effectively disappearing and learning will slow or stop entirely. Layer-wise monitoring can reveal where vanishing is occurring.

### Gradient Noise Scale (GNS)

**GNS** measures the ratio of gradient signal to gradient noise:

$$\text{GNS} = \frac{\text{batch\_size} \times \|\nabla\ell\|^2}{\text{Var}(\nabla\ell)}$$

| GNS value | Interpretation |
|---|---|
| < 1 | Very noisy gradient estimates - poor optimization |
| 1–100 | Healthy range |
| > 100 | Batch size may be larger than necessary - diminishing returns |

GNS is particularly valuable for understanding **curriculum learning effects**. When training on easy data, gradients are similar across examples, yielding high GNS. Hard or diverse data produces more variable gradients, yielding low GNS. Research shows that randomly shuffled data often produces higher GNS variance than curriculum-based orderings - an instability that can degrade performance for smaller models.

### NaN and Inf Detection

Detecting **NaN** (Not a Number) or **Inf** values is critical because they corrupt all subsequent computation. Once a NaN appears, it propagates through all updated parameters, potentially destroying the model entirely.

**Diagnosing the source:**
- *NaNs in the first few hundred steps* → initialization problems (parameters too large causing overflow, or too small causing underflow).
- *NaNs appearing suddenly during stable training* → specific problematic data examples or sequences triggering numerical edge cases.
- *NaNs appearing gradually after thousands of steps* → accumulated numerical error or an excessively large learning rate.

Modern training frameworks typically include automatic NaN detection that halts training immediately and reverts to the last good checkpoint.


## Internal State Monitoring

Loss and gradients measure what the model predicts and how it's trying to improve, but they don't reveal what's happening **inside** the model - how information flows through layers, which components are active, how representations evolve. Internal state monitoring tracks activations, attention patterns, and other intermediate quantities to provide insight invisible to surface-level metrics.

### Activation Statistics

Monitoring the **mean and standard deviation of activations** in each layer provides a basic health check. Healthy transformers with proper layer normalization maintain activation magnitudes in a relatively stable range across layers. Two failure modes to watch for:

- **Exploding activations** - magnitudes growing too large, indicating problems with gradient flow.
- **Vanishing activations** - magnitudes collapsing toward zero, indicating that information is not propagating through the network.

**Effective rank** of activation matrices (via singular value decomposition) measures whether representations are utilizing the full capacity of the model's hidden dimensions or collapsing onto a lower-dimensional subspace. **Rank collapse** - where activations become increasingly aligned along a small number of principal directions - signals a form of internal degradation where the model is losing representational diversity.

### Attention Pattern Monitoring

Self-attention computes an $N \times N$ attention matrix for a sequence of length $N$. Monitoring these patterns reveals whether the model is learning reasonable linguistic relationships or developing degenerate patterns.

**Attention entropy** measures how concentrated or diffuse attention distributions are:

| Pattern | Concern |
|---|---|
| Very low entropy | Model over-relying on simple heuristics rather than full context |
| Very high entropy | Attention not discriminating between relevant and irrelevant tokens |
| Concentrated on special tokens | Model routing computation through positions rather than distributing it meaningfully |
| Unchanged across layers | Model not refining understanding as information flows deeper |

### MoE Routing Pattern Analysis

For **Mixture-of-Experts (MoE)** models, monitoring expert routing patterns provides unique insight into computational strategy. Early in training, routing is typically random - the router hasn't learned meaningful specialization, experts are selected roughly uniformly. As training progresses, routing becomes more structured, with different experts specializing in different content types (code, dialogue, scientific text).

Recent grokking research on 7B-parameter MoE models introduced two novel routing metrics:

**1. Pathway Similarity** - measures how similar routing patterns are between different inputs. A "pathway" is the sequence of experts selected across all MoE layers for a given input, with similarity measured via edit distance between expert sequences.
- *Early training*: pathways are highly individualized - high edit distance, high diversity.
- *After loss convergence*: pathways become more similar across related inputs - the model is discovering **shared computational structures** that work across multiple examples rather than memorizing input-specific patterns.

**2. Pathway Consistency** - measures how smoothly expert selections transition from layer to layer within a single input's pathway, computed via cosine similarity between consecutive layers' router-weighted expert embeddings.
- *Low consistency*: abrupt changes in active experts across layers, suggesting disorganized computation.
- *High consistency*: expert selections change smoothly, suggesting coherent multi-layer pathways.

> Across four domains (math, code, commonsense reasoning, domain-specific knowledge), correlation coefficients between pathway metrics and benchmark accuracy exceeded **0.92**, often reaching **0.97–0.99** - while training and validation loss showed weak and inconsistent correlations with the same downstream metrics.

This reveals that internal routing organization in MoE models is a strong predictor of generalization, providing a cheap monitoring signal that tracks the expensive signal you actually care about.

### Asynchronous Grokking

Rather than a single global event where all capabilities emerge simultaneously, LLMs exhibit **local grokking** - different types of data undergo their own memorization-to-generalization transition at different steps:

- Math and coding tasks require memorizing **many more examples** before the generalization leap occurs.
- Commonsense and domain-specific QA grok earlier and more smoothly.
- Aggregate accuracy can be steadily improving while specific capabilities are still in the pre-grokking memorization phase.

This asynchronous pattern makes aggregate performance metrics **potentially misleading** and underscores the value of domain-specific downstream evaluation rather than combined averages alone.


## Checkpoint Evaluation

While continuous monitoring provides real-time feedback, checkpoint evaluation provides the **ground truth**: how well does the model actually perform on tasks that matter? Checkpoints are parameter snapshots saved at regular intervals, typically every 1,000–5,000 steps. Evaluating them on downstream tasks reveals whether the model is acquiring needed capabilities - but this evaluation is expensive, typically requiring instruction fine-tuning followed by benchmark runs that can take hours of GPU time.

### Evaluation Frequency Strategy

The fundamental trade-off is **frequency versus cost**:

| Training stage | Recommended evaluation frequency | Rationale |
|---|---|---|
| Early (0–20%) | Every 10,000 steps | Model changes rapidly; capabilities not yet meaningful |
| Mid (20–80%) | Every 5,000 steps | Critical capabilities emerging |
| Late (80–100%) | Every 1,000–2,000 steps | Catch performance plateaus or regressions |

### Benchmark Selection

**General-purpose benchmarks:**
- **MMLU** - multi-domain multiple choice, broad coverage.
- **HellaSwag** - commonsense reasoning.
- **HumanEval** - code generation.

**Domain-specific benchmarks:**
- **GSM8K** - mathematical reasoning.
- **ARC** - science questions.
- **PIQA** - physical commonsense.

The evaluation suite should be chosen to **match the training objectives and data distribution** - if training is primarily on code and mathematics, those benchmarks should be heavily weighted.

### Benchmark Contamination

If examples from the evaluation benchmark appeared in the training data, the model might memorize them rather than learning the underlying capability - **inflating performance estimates**. The **Min-K% method** flags contaminated examples by computing token-level probabilities and identifying examples where the model assigns unusually high probability to most tokens, suggesting memorization. Removing the top 10% of highest-scoring examples provides a more conservative estimate of genuine generalization.

### Checkpoint Integration

Recent research has revealed that downstream task performance during pre-training is often **unstable**, exhibiting frequent fluctuations even while loss decreases smoothly. Individual examples can flip between correct and incorrect across consecutive checkpoints, and aggregate accuracy can oscillate significantly.

**Checkpoint averaging** addresses this by merging parameters from multiple nearby checkpoints. Merging checkpoints at steps 45,000, 46,000, and 47,000 typically produces **more stable performance** than any individual checkpoint, reducing the impact of stochastic fluctuations.

**Checkpoint merging** goes further - linearly combining parameters from adjacent checkpoints often produces models that **outperform either source checkpoint individually**. For example, merging a checkpoint from 1,800 billion tokens with one from 2,000 billion tokens using Bayesian-optimized weights can exceed either source checkpoint's downstream performance. This provides a near-free performance boost at the cost of only checkpoint storage and search.

### Knowledge Acquisition and Forgetting

Analysis of publicly released checkpoint sequences (Pythia, LLM360 Amber) reveals important dynamics:

- Factual knowledge is **acquired progressively** - memorization of specific facts appears after the fact has been encountered multiple times in training data.
- Knowledge is **gradually forgotten** following a power-law decay: the probability a model assigns to a fact it once learned decreases logarithmically with training steps since it last saw that fact.
- Forgetting is **mitigated** by data deduplication and larger batch sizes, both of which improve retention of acquired knowledge.


## Practical Monitoring Infrastructure

Effective monitoring requires infrastructure that can collect, store, analyze, and visualize metrics at scale. A typical training run generates **millions of metric data points** over days or weeks - this data must be logged reliably, stored efficiently, and made accessible for both real-time monitoring and post-hoc analysis.

### Logging Frameworks

**TensorBoard**, **Weights & Biases (W&B)**, and **MLflow** provide standard infrastructure for metric collection, handling:
- Aggregation across multiple processes in distributed training.
- Time-series storage of potentially millions of data points.
- Automatic visualization of loss curves, gradient norms, and other metrics.

**Metric collection overhead by type:**

| Metric | Frequency | Overhead |
|---|---|---|
| Training loss, learning rate | Every step | Negligible - computed anyway |
| Gradient norms | Every step | Modest - reduction over all parameters |
| Activation statistics | Every 100–1,000 steps | Moderate |
| Checkpoint evaluation | Every few thousand steps | High - runs asynchronously on separate compute |

### Alerting Systems

Alerts notify engineers when metrics indicate problems. Effective alerting balances **sensitivity** (catching real problems early) against **specificity** (avoiding alert fatigue from false positives).

**Common alert triggers:**

| Condition | Severity |
|---|---|
| NaN values appear anywhere | Critical - halt immediately |
| Loss has not decreased in 1,000 steps | High - investigate immediately |
| Gradient norm exceeds 100 | High - potential instability |
| Validation loss diverging from training loss | Medium - monitor closely |
| Specific benchmark regressing while others improve | Medium - investigate root cause |

### Distributed Training System Metrics

When training on hundreds of GPUs, model metrics must be complemented by system-level monitoring:

- **GPU utilization** - are all GPUs doing useful work?
- **Network bandwidth** - is communication saturating the interconnect?
- **Memory usage** - are any processes approaching OOM failure?
- **Stragglers** - are some processes much slower than others, stalling the cluster?

Per-GPU monitoring can reveal **hardware failures** (a failing GPU producing incorrect results) or **load imbalances** (some processes assigned disproportionate work). Most distributed training frameworks include built-in system metric collection.

### Data Pipeline Monitoring

Monitoring the data pipeline ensures training receives well-formed, diverse data at the expected rate:

| Signal | What it catches |
|---|---|
| Average sequence length | Sudden drop → pipeline malfunction producing truncated sequences |
| Token distribution | Unexpected frequency spikes → data stuck repeating a small subset |
| Batch loading time vs. compute time | Loading >> compute → data pipeline is the bottleneck |
| Perplexity of training examples under a reference model | Distributional drift in incoming data |

### Cost and Efficiency Monitoring

Tracking **cost alongside performance** helps optimize training configurations and justify budgets:
- Total GPU-hours consumed.
- Cost per 1B tokens processed.
- Cost per benchmark point of improvement.

If increasing batch size reduces training time by 20% but increases cost by 30%, it may not be worthwhile. If a different learning rate schedule achieves the same final performance 10% faster, that's a clear optimization win.

### Qualitative Monitoring

Pure metrics cannot catch everything. **Actually reading model outputs** at regular intervals provides a reality check:
- Is the model producing coherent, relevant completions?
- Are there subtle coherence problems or unexpected tone shifts?
- Does it exhibit harmful biases or factual errors?

Qualitative monitoring is typically done semi-manually every few thousand steps, generating completions for a small set of fixed test prompts and reading through them to confirm the numeric metrics reflect genuine progress.



## Putting It All Together

Effective monitoring requires orchestrating multiple complementary signals into a coherent picture of training health. No single metric tells the whole story:

- Training loss might look perfect while downstream performance degrades.
- Gradients might seem stable while internal representations collapse.
- Downstream benchmarks might improve while the model is learning problematic biases.

### Hierarchical Monitoring Strategy

| Layer | Frequency | Metrics | Purpose |
|---|---|---|---|
| Continuous lightweight | Every step | Loss, gradient norms, learning rate, throughput | Catch immediate failures (divergence, NaN, stalled training) |
| Periodic medium | Every 100–1,000 steps | Activation statistics, GNS, MoE routing patterns | Reveal internal training dynamics, catch subtler issues |
| Deep evaluation | Every few thousand steps | Downstream benchmarks (with instruction fine-tuning) | Validate that cheap metrics correlate with real capability gains |

### Responding to Problems

| Severity | Example | Response |
|---|---|---|
| Catastrophic | NaN loss, complete divergence | Halt immediately, revert to last good checkpoint, diagnose, adjust, resume |
| Serious | Persistent gradient clipping, benchmark regression | Analyze recent checkpoints, examine data from problematic period, potentially adjust LR and resume from pre-issue checkpoint |
| Minor | Small loss oscillations, brief gradient spikes | Monitor and wait - watch for escalation before acting |

### Post-Training Analysis

Logged monitoring data informs future training runs by enabling comparative analysis:
- Which configurations correlate with smooth gradient norms and stable routing patterns?
- Does a different data ordering produce better GNS stability?
- What's the typical lag between loss convergence and grokking for each capability domain in this model family?

This comparative analysis builds **institutional knowledge** - turning LLM training from an expensive black box into a process that can be understood, predicted, and optimized.

> The monitoring landscape continues to evolve. Recent advances include using MoE pathway dynamics as cheap proxies for expensive downstream evaluation, checkpoint merging to extract more value from intermediate checkpoints, and GNS tracking to characterize optimization stability. As models continue to scale and training costs rise, the value of effective monitoring increases proportionally - the difference between successful and failed training is often visible in the monitoring data **thousands of steps before** the outcome becomes apparent in final model quality.
