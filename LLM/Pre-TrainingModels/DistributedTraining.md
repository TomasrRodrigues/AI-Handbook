# Distributed Training

#### Table of Contents

1. [The Scale Challenge](#the-scale-challenge)
2. [Parallelism Strategies](#parallelism-strategies)
3. [Infrastructure and Hardware](#infrastructure-and-hardware)
4. [Memory Optimization](#memory-optimization)
5. [Communication Optimization](#communication-optimization)
6. [Fault Tolerance](#fault-tolerance)
7. [Real-World Implementations and Best Practices](#real-world-implementations-and-best-practices)

## The Scale Challenge

Training large language models has become one of the most computationally demanding endeavors in modern computing. When Meta trained Llama 3, the largest variant consumed approximately fifty-four days of continuous computation across sixteen thousand NVIDIA H100 GPUs. This represents not just an engineering challenge but a fundamental shift in how we must think about machine learning infrastructure. No single GPU, regardless of how powerful, can hold even a fraction of a model with hundreds of billions of parameters along with its optimizer states, gradients, and activation tensors.

The mathematics are stark. Consider a model with 175 billion parameters like GPT-3. In mixed-precision training using 16-bit floats, storing just the parameters requires **350 GB**. But parameters are only the beginning - the Adam optimizer maintains two additional states per parameter (momentum and variance). Because these must be kept in 32-bit precision for numerical stability (4 bytes per parameter), each state requires **700 GB**, adding a massive **1.4 TB** of overhead. Then come the 16-bit gradients, adding another **350 GB**. Before the first forward pass, we've exceeded **2.1 TB** of memory. The most advanced GPU available in 2025, the H100, provides 80 GB of HBM. Even fitting the model states alone would leave no room for activations.

This memory wall forces distributed training, where model, states, and workload are partitioned across many devices. But distribution introduces its own challenges. GPUs must communicate constantly - synchronizing gradients, exchanging activations between pipeline stages, coordinating optimizer updates. Communication over networks is orders of magnitude slower than local GPU memory: while an H100 accesses HBM at ~4 TB/s, even the fastest interconnects top out at hundreds of GB/s, shared across all simultaneous traffic.

A subtler challenge is numerical equivalence. Distributed training should produce the same model as single-device training - the same gradients, the same updates, the same final quality - regardless of how work is divided. Small differences in floating-point operation order can accumulate over billions of steps, potentially causing divergence. Maintaining this equivalence while distributing work across thousands of devices requires careful engineering throughout the entire stack.


### The Three Pillars: Scalability, Efficiency, Reliability

**Scalability** asks whether we can effectively use more hardware to train larger models or reduce training time. Ideal scaling would be linear - double the GPUs, halve the time. Reality falls short. Meta's Llama 3 training achieved **38–41% Model FLOPs Utilization** on 16,384 GPUs, meaning less than half of theoretical peak compute translated into useful work. The gap between ideal and actual grows as we add more devices, as communication patterns become more complex and synchronization overhead multiplies.

There are two distinct scaling regimes. *Strong scaling* keeps model size fixed and divides work across more devices to reduce time - this hits communication limits quickly. *Weak scaling* keeps work-per-device constant while growing both model and device count proportionally - this scales better but requires model architectures that grow smoothly. The most practically valuable form is simply being able to train models that would be impossible on fewer devices, even imperfectly.

**Efficiency** measures how well we exploit available hardware. It manifests across several dimensions: computational efficiency (how much of each GPU's FLOPs go toward training vs. waiting), memory efficiency (how much memory stores useful data vs. redundant copies), and communication efficiency (whether bandwidth is used for large useful transfers or wasted on small frequent messages). These interact in complex ways - larger batch sizes improve GPU utilization but increase activation memory; gradient accumulation improves communication efficiency but occasionally stalls compute. Every decision in distributed training involves trading efficiency in one dimension for another.

**Reliability** becomes paramount when training spans weeks. Even if individual GPUs fail at just 0.1% per day, a 10,000-GPU cluster expects ten failures daily. Without robust fault tolerance, each failure requires restarting from the last checkpoint, potentially losing hours of expensive GPU time. Beyond catastrophic failures, transient network issues create stragglers that stall entire clusters, and silent memory corruption can introduce errors that only surface as divergent training curves hours later.

The three pillars interlock. Improving scalability often means accepting lower efficiency - using more devices than strictly necessary to handle load imbalance, or maintaining redundancy for reliability. Reliability mechanisms impose efficiency costs through monitoring and redundant computation. The best distributed training systems don't optimize any single pillar in isolation, but find balanced configurations that support training at the required scale within acceptable time and budget.


### Understanding Training at 16000 + GPUs

To ground these abstractions, consider a single training iteration on Llama 3 405B across 16,384 H100 GPUs.

The cluster is organized hierarchically. Eight H100s within a single node connect through NVSwitch at **900 GB/s** bidirectional bandwidth - fast enough for frequent fine-grained communication. Multiple nodes within a rack connect via InfiniBand or RoCEv2 at **200–400 GB/s**. Racks connect through spine switches with progressively lower bandwidth at each layer.

Meta's configuration employs **four-dimensional parallelism**:
- **Tensor parallelism ×8** - within nodes, exploiting NVSwitch bandwidth
- **Pipeline parallelism ×4** - across nearby nodes, splitting the model into four stages
- **Data parallelism ×64** - across the full cluster, with 64 independent replicas
- **Context parallelism ×2** - handling 8,000-token sequences

Multiplying these: 8 × 4 × 64 × 2 = 16,384 GPUs total.

During a forward pass, the input sequence flows through the first pipeline stage across its eight tensor-parallel GPUs. Each attention layer requires an All-Gather to reconstruct full activations, then computation, then the pattern repeats. When a stage finishes its micro-batch, it sends activations to the next stage over the inter-node network. Meanwhile, the second stage processes a different micro-batch it received earlier - this pipelining keeps all stages busy simultaneously.

Backpropagation reverses the flow. After gradients propagate through all four stages, a Reduce-Scatter averages gradients across the 64 data-parallel replicas. Each GPU receives averaged gradients only for its parameter shard - one sixty-fourth of the total. The optimizer updates those shards locally, then an All-Gather reconstructs complete parameters before the next forward pass.

Throughout this orchestration, communication is constant. Each GPU participates in dozens of collective operations per training step, moving gigabytes of data. These must be carefully scheduled to overlap with computation rather than blocking GPU progress. The complexity explains why achieving even 40% hardware utilization at this scale is a significant accomplishment - and why organizations invest heavily in co-designing model architecture, parallelism strategy, network topology, and software stack together.


## Parallelism Strategies

Data parallelism is the most intuitive approach: replicate the entire model across multiple GPUs, feed each GPU different batches, and synchronize gradients after each backward pass. The workflow has four steps - each GPU processes its mini-batch independently, computes gradients, participates in an All-Reduce that averages gradients across all replicas, and applies the averaged update. The mathematical equivalence to single-device training holds exactly: averaging 100 GPUs' gradients over 20 examples each gives the same result as one GPU processing all 2,000 examples.

The elegance of data parallelism is real, but it faces a fundamental memory barrier. Each GPU must store complete parameters, optimizer states, and gradients. For a 175B parameter model, this exceeds 1 TB per GPU before considering activations - impossible on any current hardware.

This drove development of **sharded data parallelism**, embodied in ZeRO (Zero Redundancy Optimizer) and PyTorch FSDP. Rather than each GPU storing everything, these partition parameters, optimizer states, and gradients across devices. Each GPU owns 1/N of the parameters, stores optimizer state only for those parameters, and temporarily gathers complete parameters via All-Gather when needed for computation - immediately discarding borrowed parameters afterward.

ZeRO divides its optimization into three stages:
- **ZeRO-1** - partitions optimizer states only; parameters and gradients remain replicated
- **ZeRO-2** - additionally partitions gradients
- **ZeRO-3 / FSDP** - partitions parameters as well, achieving maximum memory reduction

The communication overhead of ZeRO-3 initially seems daunting - every layer requires two All-Gathers and one Reduce-Scatter instead of a single end-of-step All-Reduce. However, the total communication volume is similar to standard data parallelism. The difference is that communication is distributed throughout the step rather than concentrated at the end, which actually enables better overlap with computation. ZeRO-3 often achieves better throughput than standard data parallelism because improved memory efficiency enables larger batch sizes.

### Tensor Parallelism: Splitting the Math

Tensor parallelism splits individual tensor operations across multiple devices. The key insight comes from matrix multiplication: computing Y = X × W can be split by dividing W column-wise across GPUs, computing partial results independently, then concatenating. The two GPUs perform their matrix multiplications completely independently, synchronizing only to combine results.

Megatron-LM pioneered efficient tensor parallelism for transformers by carefully analyzing which operations to split and how. In multi-head attention, each head can be assigned to a different GPU and computed entirely independently until outputs must be concatenated. The feed-forward networks use alternating column-wise and row-wise splits to minimize communication while keeping memory footprints small.

Each transformer layer requires two synchronization points - one All-Reduce after attention and one after feed-forward. For a batch of 16 sequences × 4,000 tokens × 16,000 hidden dim in BF16, each All-Reduce transfers ~2 GB. With 40 attention blocks, that's roughly 80 GB of communication per step.

This communication volume explains why tensor parallelism is confined to high-bandwidth domains. Meta's Llama 3 used tensor parallelism degree 8 strictly within nodes where NVSwitch provides 900 GB/s. Attempting it across nodes, where bandwidth drops to 200–400 GB/s with higher latency, would cause communication to dominate training time.


### Pipeline Parallelism: Layer-wise Distribution

Pipeline parallelism assigns consecutive layers to different devices. A 100-layer model with pipeline degree 4 assigns layers 1–25 to device one, 26–50 to device two, and so on. The challenge is utilization - naively processing one batch at a time leaves three of four devices idle at any moment.

The solution is **micro-batching**: divide the batch into multiple micro-batches that flow through the pipeline in overlapping fashion. While stage one processes micro-batch two, stage two processes micro-batch one. Once the pipeline fills, every stage computes simultaneously on different micro-batches.

"Bubble time" remains an inherent cost - periods of idleness during pipeline warm-up and cool-down. The efficiency is mathematically expressed as:

$$\text{Efficiency} \approx \frac{M}{M + P - 1}$$

where $M$ is the number of micro-batches and $P$ is the number of pipeline stages. With 64 micro-batches and 4 stages, efficiency reaches $64/67 \approx 95.5\%$, which is why pipeline parallelism works best with many micro-batches.

Different scheduling strategies trade memory against efficiency:
- **GPipe** completes all forward passes before backward passes - easy to implement but maximizes bubble time
- **1F1B (one-forward-one-backward)** interleaves forward and backward passes, reducing bubbles but increasing activation memory
- **Interleaved schedules** assign multiple non-consecutive stage chunks per GPU, further reducing bubble time at additional implementation complexity


### Sequence Parallelism: Conquering Long Contexts

As context lengths push toward hundreds of thousands of tokens, activation memory becomes the limiting factor. Attention over 128,000 tokens with hidden dim 16,000 in BF16 requires ~32 GB just for attention scores - before key-value caches or anything else.

Sequence parallelism distributes the sequence dimension across GPUs. The challenge is that attention creates dependencies between all positions, so each GPU needs information about tokens on other GPUs.

**Ring attention** solves this elegantly. Each GPU holds a quarter of the sequence. GPUs compute attention between their local queries and their local keys, then pass key-value pairs around a ring - GPU 1 → GPU 2 → GPU 3 → GPU 4 → GPU 1. After P complete rotations, each GPU has computed its queries against all keys. Partial outputs accumulate across iterations.

**DeepSpeed-Ulysses** takes a different approach, splitting across the head dimension rather than sequence. Each GPU computes all attention for a subset of heads across the full sequence. This requires All-to-All communication before and after attention to transpose data, but achieves perfect load balance - unlike ring attention which has imbalance under causal masking, as later positions have fewer past tokens to attend to.

### Expert Parallelism: MoE Models

Mixture-of-Experts models route each token to a subset of expert networks - a typical MoE layer might have 8 experts but each token uses only the top 2. Expert parallelism distributes these experts across GPUs: in an 8-expert, 8-GPU setup, each GPU hosts one expert.

Forward propagation requires All-to-All communication to send tokens from their current GPU to the GPUs hosting their assigned experts. After expert processing, another All-to-All returns tokens to their original GPUs. This pattern repeats during backpropagation.

The challenge is that routing is dynamic - if some experts are more popular, some GPU pairs exchange many tokens while others exchange few. This imbalance wastes bandwidth. Load balancing techniques address this through auxiliary losses that encourage uniform routing, or capacity factors that limit tokens per expert and force overflow to less popular ones - a deliberate trade of model quality for training efficiency.


### Hybrid Approaches: Combining Strategies

Real LLM training combines multiple strategies mapped to hardware topology. Meta's Llama 3 configuration illustrates the principle:

- **Tensor parallelism ×8** maps to intra-node NVSwitch (900 GB/s) - for high-frequency All-Reduces
- **Pipeline parallelism ×4** maps to inter-node links within racks (200–400 GB/s) - for less frequent activation transfers
- **Data parallelism ×64** maps across the full cluster - tolerant of modest bandwidth and higher latency

This hierarchical mapping is not accidental. Research on topology-aware parallelism shows that communication patterns must be matched to network layers based on their bandwidth requirements and frequency. Violations - like tensor parallelism across nodes - severely degrade performance by saturating slow links with high-frequency traffic.


## Infrastructure and Hardware

Modern GPUs like the H100 contain hierarchical memory tiers. At the top sits **HBM** (~80 GB, ~3 TB/s), storing parameters, optimizer states, activations, and gradients. Below sits **L2 cache** (~50 MB) for frequently reused data. At the lowest level, each streaming multiprocessor has local shared memory and registers providing extremely fast but tiny capacity.

This hierarchy creates a fundamental tension. Operations are either *compute-bound* (arithmetic dominates; memory access is infrequent - matrix multiplications fall here) or *memory-bound* (loading and storing data consumes more time than computation - activation functions, layer norm, dropout). Compute-bound operations achieve high GPU utilization; memory-bound operations do not.

**FlashAttention** exemplifies optimizing for memory hierarchy. Standard attention materializes the full attention score matrix in HBM, requiring memory quadratic in sequence length. FlashAttention redesigns the computation to work in blocks that fit in fast SRAM. It loads small blocks of Q, K, V from HBM, computes attention entirely in SRAM, and writes only the final output back. This IO-aware algorithm reduces memory from quadratic to linear in sequence length while simultaneously improving speed by minimizing slow HBM accesses.

The H100's **Transformer Engine** provides hardware acceleration for FP8 computation, halving memory bandwidth requirements and doubling effective throughput for supported operations - though it requires careful numerical handling to maintain training stability.

### Network Topologies and Interconnects

Within a node, NVLink and NVSwitch provide direct GPU-to-GPU connectivity. NVSwitch aggregates connections to create a fully-connected topology within nodes at 900 GB/s - making tensor parallelism efficient. Between nodes, **InfiniBand** (400 Gb/s per link with RDMA) has long dominated HPC clusters. **RoCEv2** (RDMA over Converged Ethernet) has emerged as a cost-effective alternative; Meta's Llama 3 training used RoCE clusters, demonstrating that carefully tuned Ethernet can match InfiniBand for large AI workloads.

Cluster-scale topology typically follows leaf-spine architecture: GPUs connect to top-of-rack switches, which connect to spine switches for inter-rack connectivity. This hierarchy creates bandwidth oversubscription - 1:1 within racks, 2:1 or worse between racks. Communication crossing spine layers experiences lower bandwidth and higher latency.

**Rail-optimized topologies** specialize this design for ML. GPUs in the same position across different servers connect to dedicated switches - GPU 0 in every server connects to rail-0 switches, GPU 1 to rail-1, and so on. This aligns with data-parallel communication patterns: gradient synchronization stays within single switches rather than crossing multiple hops. Meta reports this topology significantly improves training throughput by reducing congestion.

### Storage Systems for Checkpoints

A 400B parameter model with optimizer states produces ~3 TB checkpoint files. Writing these hourly to limit work loss from failures demands aggregate storage bandwidth in the TB/s range as hundreds of GPUs simultaneously write sharded checkpoint parts.

Distributed filesystems like Lustre serve this role in HPC environments. Object stores like S3 or Ceph are increasingly popular for cloud deployments - they sacrifice POSIX compatibility for better scalability and simpler deployment. Training frameworks increasingly support direct object store integration.

Key techniques for checkpoint efficiency:
- **Asynchronous checkpointing** - copies state to CPU memory in seconds, allowing GPU computation to resume; background process transfers to storage while training continues
- **Checkpoint sharding** - divides across many files written in parallel, maximizing bandwidth utilization
- **In-memory checkpointing** - stores copies in other nodes' memory using erasure coding (similar to RAID5), enabling near-zero-overhead checkpointing every few iterations

### Cluster Scheduling and Management

**Gang scheduling** ensures all resources a distributed job requires are available simultaneously before launching. For a job spanning 1,000 GPUs, attempting to start with only 900 available would cause launched processes to hang indefinitely waiting for the rest. Gang scheduling prevents this by atomically allocating all required resources - at the cost of reduced cluster utilization, since large jobs must wait for sufficient contiguous resources.

**Elastic training** partially addresses this by allowing jobs to run on varying numbers of GPUs. As resources become available or are reclaimed, training adapts by adjusting parallelism degree. This is most natural for data parallelism where adding or removing replicas is straightforward; pipeline and tensor parallelism can also support elasticity but require more complex resharding.

**Priority scheduling and preemption** allow high-priority jobs to displace lower-priority ones. Production training runs with deadlines receive higher priority than exploratory jobs. When they need resources, the scheduler suspends low-priority jobs, checkpoints their state, and reclaims their GPUs. This maximizes throughput for high-priority workloads but requires robust checkpointing infrastructure.

## Memory Optimization

Activation tensors consume substantial memory. For each layer in the forward pass, we store intermediate activations to compute gradients during backpropagation. A batch of 64 sequences × 4,000 tokens × 16,000 hidden dim requires ~32 GB to store attention outputs from a single layer in BF16. Multiply by 50 layers and activation memory alone approaches 1.6 TB.

**Activation checkpointing** trades computation for memory by selectively discarding activations during the forward pass and recomputing them on-demand during backpropagation. The simplest strategy checkpoints every Nth layer, discarding intermediate activations and recomputing them from the nearest checkpoint when needed. This reduces memory from linear in depth to approximately square root of depth, at the cost of roughly 33% extra computation.

**Selective activation checkpointing** optimizes which layers to checkpoint. Attention layers are memory-intensive but relatively cheap to recompute - good candidates for not checkpointing. FlashAttention improves this further by recomputing attention during backpropagation using only the output and softmax statistics, avoiding the need to store massive attention score matrices. Feed-forward networks are computationally heavier, making them better candidates for checkpointing their outputs.


### Sharded Optimizers: ZeRO and FSDP

Optimizer state memory often exceeds parameter memory. Adam maintains two momentum terms per parameter - both in FP32 for numerical stability even when parameters use BF16. For a 175B parameter model using BF16, parameters consume **350 GB**, but each of Adam's two FP32 states requires **700 GB**, totalling **1.4 TB** for optimizer states alone - four times the parameter memory.

ZeRO eliminates this redundancy by partitioning across data-parallel ranks. Each rank owns 1/N of parameters and stores optimizer states only for those parameters. The three stages (previously described) progressively partition optimizer states, gradients, and parameters, with ZeRO-3/FSDP achieving maximum memory reduction.

The communication pattern of FSDP - All-Gather before each layer's forward pass, Reduce-Scatter after backward - distributes communication throughout the step rather than concentrating it at the end. This distribution enables better overlap with computation. Combined with the fact that improved memory efficiency enables larger batch sizes, FSDP often achieves better throughput than standard data parallelism despite increased communication volume.


### CPU and SSD Offloading

When models exceed even sharded GPU memory, offloading transfers data to slower but more capacious storage tiers.

**ZeRO-Offload** keeps optimizer states in CPU memory while parameters and activations remain on GPU. During each step, gradients transfer to CPU where optimizer computations occur, then updated parameters transfer back. This works because optimizer computations are lightweight element-wise operations easily handled by CPUs. The bandwidth cost is manageable - only gradients and parameter updates cross the PCIe bus, not the much larger optimizer states.

**ZeRO-Infinity** extends offloading to NVMe SSDs, enabling models with trillions of parameters. Parameters and optimizer states reside on SSD, loaded to CPU and GPU as needed. Careful prefetching and pipelining hide latency by loading upcoming layers' data while current layers compute. SSD offloading trades throughput for capacity - training slows considerably, but otherwise impossible models become feasible.

**Activation offloading** temporarily moves activations to CPU after each forward layer, prefetching them back just before they're needed in backpropagation. This dramatically reduces GPU memory pressure at the cost of increased PCIe traffic; careful scheduling can maintain high GPU utilization despite the additional movement.


## Communication Optimization

Communication overhead limits distributed training scalability. As we add more GPUs, communication volume often grows faster than computation.

Distributed training relies on a small set of collective operations:
- **All-Reduce** - averages data across all ranks; essential for gradient synchronization in data parallelism
- **All-Gather** - collects data from all ranks and replicates it everywhere; used in ZeRO-3 to reconstruct parameters
- **Reduce-Scatter** - reduces data and distributes results; used in FSDP for gradient reduction

The **ring algorithm** implements these collectives efficiently. Devices form a logical ring, passing chunks sequentially. All-Reduce proceeds in two phases: reduce-scatter (each device accumulates partial sums from neighbors until holding fully reduced data for its chunk) then all-gather (reduced chunks circulate until everyone has complete data). Ring algorithm bandwidth scales linearly with data size but independently of device count - excellent strong scaling.

**Tree algorithms** offer different trade-offs: O(log N) latency rather than O(N), making them better for small messages where latency dominates. However, they don't fully utilize network bandwidth since only tree edges carry data. Modern libraries like NCCL auto-select ring, tree, or hybrid algorithms based on message size, topology, and device count.


### Topology-Aware Optimization

Effective communication requires matching patterns to network topology. The principle is consistent: high-frequency, high-volume communication belongs on fast links; low-frequency operations can tolerate slower paths.

Recursive halving and doubling algorithms exploit hierarchy explicitly. For All-Reduce across 64 GPUs organized as 8 nodes × 8 GPUs, the algorithm first reduces within each node using fast intra-node links, then reduces across the 8 nodes using inter-node links, then broadcasts back within nodes. This minimizes the amount of data that must cross slow inter-node links.

### Overlapping Communication and Computation

**Gradient bucketing** groups multiple small gradient tensors into buckets. Rather than communicating each layer's gradients individually - creating many small communications with high overhead - gradients accumulate in buckets until reaching a size threshold, then a single communication transmits the entire bucket. This amortizes communication overhead while still enabling early communication of recently computed gradients.

**Chunked streaming** breaks large tensors into pieces. As soon as a matrix multiplication finishes its first output chunk, communication for that chunk begins while remaining chunks compute. By the time computation finishes, much of the communication has already completed - particularly effective for large FSDP All-Gathers.

**Double buffering** extends this further: while one layer computes using already-gathered parameters, the next layer's parameters are prefetched. During backpropagation, while computing gradients for one layer, activations for the previous layer are prefetched from CPU if offloaded. These producer-consumer patterns ensure slow operations always have work queued, minimizing idle time.


## Fault Tolerance

Training jobs spanning weeks on thousands of GPUs face near-certain failures. The question is not whether failures will occur but how quickly training can recover.


### Failure Analysis in Large-Scale Training

Empirical data paints a sobering picture. Meta's Llama 3 pre-training experienced **466 job interruptions** over 54 days on 16,384 GPUs. ByteDance's MegaScale project observed over 100 failures in several weeks on ~12,000 GPUs. Hardware failures dominate - approximately 78% of interruptions - manifesting as CUDA errors, ECC memory errors, or NVLink failures. Network failures range from complete node isolation to subtle link degradation. The remaining failures stem from software bugs, numerical instabilities, race conditions, or user code mistakes.

Distributed training amplifies software bugs: an issue on any single rank propagates instantly through collective operations. A single rank hanging in an All-Reduce call blocks all other ranks indefinitely.


### Checkpointing Strategies

The fundamental trade-off is frequency versus overhead. Checkpointing every 5 minutes limits potential work loss to 5 minutes, but writing a 3 TB checkpoint takes minutes during which GPUs sit idle.

**Asynchronous checkpointing** addresses this by copying state to CPU memory in seconds, allowing GPU computation to resume immediately. A background process then transfers to persistent storage while training continues. The challenge is consistency - if training crashes before the async write completes, that checkpoint is unusable. Systems must maintain at least one fully written previous checkpoint as a fallback.

**In-memory checkpointing** stores snapshots in other nodes' RAM using erasure coding. If any single node fails, its data can be reconstructed from other nodes. This enables checkpointing every few iterations with negligible overhead since memory-to-memory transfers over the network are far faster than storage writes. Simultaneous multi-node failures can still cause checkpoint loss, so periodic writes to persistent storage remain necessary as an ultimate backup.


### Recovery Machines

Health monitoring tracks GPU utilization, memory errors, network packet drops, and training throughput. Deviations trigger alerts: a GPU's compute utilization dropping to zero while others remain high indicates a problem; NaN loss or unexpected loss increases suggests numerical instability.

Once a failure is detected, the system isolates problematic nodes and restarts training on remaining healthy nodes. **Automatic exclusion lists** blacklist failing ranks from subsequent attempts. If the failure was transient, training resumes successfully on fewer nodes. If persistent, the blacklist grows until stable training resumes or human intervention is needed.

**Elastic training** provides another recovery path: rather than requiring a fixed cluster configuration, training detects available workers at startup and configures parallelism accordingly. If failures reduce worker count, training reshards across fewer devices - straightforward for data parallelism, more complex for tensor and pipeline parallelism.



## Real-World Implementations and Best Practices

**Megatron-LM** (NVIDIA) is heavily optimized for NVIDIA GPUs and network fabric. It implements efficient tensor parallelism with carefully designed layer partitioning, pipeline parallelism with interleaved schedules, and ZeRO-style parameter sharding. Performance comes from low-level optimizations: fused kernels combining multiple operations, custom collective implementations exploiting NVLink topology, and memory-efficient attention. For maximum performance with experienced engineers on NVIDIA hardware, it's hard to beat.

**PyTorch FSDP** is built into PyTorch as a first-party feature, providing ZeRO-3 style sharding with a relatively simple API. Researchers can enable it with minimal code changes, making advanced memory optimization accessible without deep systems expertise. FSDP achieves competitive performance through careful communication overlap, bucketing strategies, and mixed-precision training. Its integration with the broader PyTorch ecosystem makes it a natural default for most research use cases.

**DeepSpeed** (Microsoft) offers a comprehensive optimization library including all three ZeRO stages, pipeline parallelism, efficient optimizers, and compression techniques. ZeRO-Infinity's SSD offloading enables training models that would be completely infeasible otherwise. Its modular design allows enabling just the optimizations needed - from simple ZeRO-1 to complex combinations of ZeRO-3, pipeline parallelism, and CPU offloading.

Choosing among these requires understanding your constraints. For maximum performance on NVIDIA hardware, Megatron-LM. For ecosystem integration and accessibility, FSDP. For extreme memory optimization beyond available GPU memory, DeepSpeed. Many organizations use multiple frameworks for different use cases.

### Avoiding Common Pitfalls

**Mismatching parallelism to topology** is the most costly mistake. Tensor parallelism across nodes, or data parallelism confined to intra-node links, creates severe bottlenecks that no amount of tuning can fix. Always map high-frequency communication to fast links.

**Under-checkpointing** risks losing hours of work to a single hardware failure. The math usually favors more frequent checkpoints - asynchronous approaches make this cheap enough that there's little excuse for checkpointing less than every 15–30 minutes at scale.

**Ignoring stragglers** lets one slow node degrade the entire cluster. Proactive monitoring that detects degraded nodes before they cause full failures is worth the engineering investment.

**Skipping profiling** makes optimization guesswork. Tools like NVIDIA Nsight Systems and PyTorch Profiler reveal whether bottlenecks are compute, communication, or memory - answers that are often surprising. A training job that seems slow due to communication might actually be memory-bound, requiring a different solution entirely.

> The knowledge in this document provides a foundation, but distributed training remains as much art as science. Small inefficiencies multiply across thousands of GPUs and billions of steps - a 1% improvement in efficiency translates to hours saved in total training time and substantial cost reduction. Co-designing model architecture, parallelism strategy, network topology, and software stack together, rather than optimizing each in isolation, is what separates efficient training from simply throwing hardware at the problem.


---

<div style="display: flex; justify-content: space-between;">
  <a href="DataPreparation.md">Back</a>
  <a href="TrainingOptimization.md">Next</a>
</div>