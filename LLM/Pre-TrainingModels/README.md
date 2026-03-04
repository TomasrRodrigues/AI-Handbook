# LLM Pre-Training

Pre-training is the process of teaching a neural network to predict the next token in a sequence of text. This deceptively simple objective — repeated across trillions of examples — enables models to acquire broad knowledge, reasoning patterns, and language understanding that can later be adapted to countless downstream tasks.

> The quality of any language model is largely determined during pre-training. No amount of fine-tuning compensates for poor data, unstable optimization, or a training run that diverges halfway through.

Successful pre-training requires orchestrating four interconnected pillars. Each links to a detailed guide below.


## The Four Pillars

### 1. [Data Preparation](DataPreparation.md)

Before a single training step runs, you must assemble and clean a corpus often measuring in the trillions of tokens. This sets a hard upper bound on what the model can ever learn.

**Key stages:**
- **Collection** — web crawls, books, code, scientific papers, conversational data.
- **Quality filtering** — removing spam, gibberish, and low-value content via heuristics and model-based classifiers.
- **Deduplication** — exact and fuzzy matching to eliminate repeated content that wastes compute and encourages memorization.
- **PII & toxicity removal** — scrubbing private information and harmful content.
- **Data mixing** — deciding how to weight different sources in the final corpus.

→ *[Read the full Data Preparation guide](DataPreparation.md)*


### 2. [Distributed Training](DistributedTraining.md)

A 175B parameter model requires over **2 TB** of memory for parameters, optimizer states, and gradients alone — far beyond any single GPU. Training must be distributed across hundreds or thousands of devices.

**Key strategies:**
- **Data parallelism / FSDP** — replicate or shard the model across GPUs; average gradients after each step.
- **Tensor parallelism** — split individual matrix operations across devices (within nodes, over fast NVSwitch links).
- **Pipeline parallelism** — assign consecutive layers to different devices; use micro-batching to keep all stages busy.
- **Sequence parallelism** — distribute long sequences across GPUs for extreme context lengths.

Real deployments combine all four, mapped to hardware topology. Meta's Llama 3 used 8× tensor × 4× pipeline × 64× data parallelism across 16,384 H100s.

→ *[Read the full Distributed Training guide](DistributedTraining.md)*


### 3. [Training Optimization](TrainingOptimization.md)

With data and infrastructure in place, the optimization process determines whether the model actually learns — and whether training survives to completion.

**Key components:**
- **Optimizer** — AdamW is the de facto standard; combines momentum with per-parameter adaptive learning rates.
- **Learning rate schedule** — linear warmup to prevent early instability, followed by cosine decay or WSD (Warmup-Stable-Decay) for flexibility.
- **Gradient clipping** — bounds the global gradient norm to prevent explosion; 5–20% of steps clipped is healthy.
- **Mixed precision** — BF16 for most compute; FP32 for numerically sensitive operations like LayerNorm and softmax.
- **Gradient accumulation** — enables large effective batch sizes without proportionally large memory.

→ *[Read the full Training Optimization guide](TrainingOptimization.md)*


### 4. [Monitoring](Monitoring.md)

Training loss can decrease smoothly while downstream capabilities plateau, regress, or degrade invisibly. Effective monitoring detects problems before they become catastrophic.

**Key signals:**

| Layer | Frequency | What it catches |
|---|---|---|
| Loss & perplexity | Every step | Divergence, spikes, stalled training |
| Gradient norms & GNS | Every step | Explosion, vanishing, instability |
| Activation statistics | Every 100–1,000 steps | Internal representation collapse |
| Downstream benchmarks | Every few thousand steps | Whether cheap metrics reflect real capabilities |

→ *[Read the full Monitoring guide](Monitoring.md)*


## How the Pillars Interact

These areas are not independent:
- **Data quality affects optimization stability** — bad examples cause gradient spikes.
- **Parallelism strategy constrains hyperparameter choices** — pipeline depth affects micro-batch count and gradient noise.
- **Monitoring reveals data issues** — loss spikes often trace to specific problematic batches.
- **Infrastructure reliability shapes optimization** — frequent failures demand more conservative hyperparameters and checkpointing.

The goal is not to optimize any single pillar in isolation, but to find balanced configurations that successfully train a high-quality model within a finite compute budget.


## Scale Reference

| Model | Parameters | Training tokens | Token : parameter ratio |
|---|---|---|---|
| Llama 3 8B | 8B | 15T | ~1,875 : 1 |
| Llama 3 405B | 405B | 15T+ | ~37 : 1 |
| Qwen 3 0.6B | 0.6B | 36T | ~60,000 : 1 |

The Chinchilla compute-optimal ratio is ~20 tokens per parameter. Modern practice deliberately exceeds this to produce smaller, cheaper-to-deploy models — a shift from *training efficiency* to *inference efficiency*.