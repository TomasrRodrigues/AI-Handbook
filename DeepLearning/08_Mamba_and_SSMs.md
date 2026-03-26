# Chapter 8: State Space Models & Mamba - The Sequence Revolution

> *"The best architecture is the one that wastes no computation on information that is not relevant"*
> - the animating principle behind selective state spaces



## 8.1 The Aha! Moment: The Quadratic Wall

By 2023, Transformers had become the unchallenged rulers of sequence modeling. Their self-attention mechanism - allowing any position to attend directly to any other - had produced GPT-4, Claude, Gemini, and an entire generation of models with remarkable capabilities.

But Transformers carry a fundamental mathematical constraint that limits what they can become: **self-attention scales quadratically with sequence length**.

Let us unpack this precisely. Given a sequence of $T$ tokens, the attention matrix $A \in \mathbb{R}^{T \times T}$ is computed as:

$$A = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

The product $QK^T$ has $T^2$ entries - every token's query vector dot-producted with every token's key vector. Computing this matrix requires $O(T^2 d_k)$ operations and storing it requires $O(T^2)$ memory.

**Why this is a wall:**
- $T = 1{,}000$ tokens: $10^6$ attention weights - manageable
- $T = 100{,}000$ tokens (a short book): $10^{10}$ attention weights - barely feasible with specialized engineering
- $T = 10^6$ tokens (a full novel, or 60 seconds of raw audio at 16kHz): $10^{12}$ attention weights - computationally infeasible on any current hardware

Meanwhile, RNNs process sequences in **linear time**: each step costs $O(d_h^2)$ regardless of total length $T$. But RNNs suffer vanishing gradients (Chapter 6) and cannot be parallelized during training (each step requires the previous step's hidden state). They train slowly and fail on long-range dependencies.

**The challenge:** Can we build a sequence model with all three properties simultaneously?

1. **Linear time and memory in $T$** (like RNNs - needed for very long sequences)
2. **Long-range memory** (like Transformers - needed for complex reasoning)
3. **Parallelizable during training** (like Transformers - needed for efficient use of modern hardware)

This is the problem that **State Space Models (SSMs)** and specifically **Mamba** (Gu & Dao, 2023) set out to solve. The answer comes from an unexpected source: control theory and signal processing from the 1960s.



## 8.2 The Continuous State Space Model: A Physics-Inspired Foundation

State Space Models did not originate in deep learning. They describe **dynamical systems** - the mathematical framework physicists and engineers use to model anything that evolves over time (electrical circuits, mechanical systems, population dynamics).

### 8.2.1 The Core Equations

A linear time-invariant (LTI) state space model describes how a system's internal state $\mathbf{h}(t) \in \mathbb{R}^N$ evolves in response to an input signal $u(t) \in \mathbb{R}$ (a scalar for simplicity):

$$\dot{\mathbf{h}}(t) = A\mathbf{h}(t) + B\, u(t)$$

$$y(t) = C\mathbf{h}(t) + D\, u(t)$$

Reading every symbol completely:

**Equation 1 (state transition):**
- $\dot{\mathbf{h}}(t)$: the time derivative of the state vector - "how fast is the state changing right now?" The dot above $\mathbf{h}$ is standard physics notation for time derivative
- $A \in \mathbb{R}^{N \times N}$: the **state matrix** - governs how the state naturally evolves in the absence of input. Like the "internal dynamics" of the system: a pendulum swings due to its own physics even without being pushed
- $A\mathbf{h}(t)$: the state matrix multiplied by the current state - how the current state drives future state changes
- $B \in \mathbb{R}^{N \times 1}$: the **input matrix** - how the scalar input $u(t)$ drives the state
- $B\, u(t)$: the input's influence on state change
- Together: the state's rate of change equals its natural evolution plus the input's influence

**Equation 2 (output):**
- $y(t)$: the scalar output at time $t$
- $C \in \mathbb{R}^{1 \times N}$: the **output matrix** - which linear combination of state dimensions is the output?
- $C\mathbf{h}(t)$: read the relevant parts of the state
- $D \in \mathbb{R}$: direct skip connection (often set to 0 or 1 in practice)

**Analogy to understand the components:**

Think of the state $\mathbf{h}(t)$ as the water level in a system of $N$ interconnected tanks. The state matrix $A$ describes how water flows between tanks naturally (tank 1 drains into tank 2, etc.). The input matrix $B$ describes how new water poured in (the input signal $u(t)$) fills the tanks. The output matrix $C$ describes which measurement you read off the system.

**Why this is the right abstraction:** Every linear sequential system - every filter, every recurrent process - can be written in this form. SSMs are not a new computational primitive; they are the universal language for linear dynamical systems.

### 8.2.2 The Convolution Perspective

One of the most powerful properties of linear SSMs: the output is the **convolution** of the input with a kernel derived from the system matrices.

For $\mathbf{h}(0) = \mathbf{0}$ (starting at rest), the output at time $t$ is:

$$y(t) = \int_0^t C e^{A(t-\tau)} B\, u(\tau)\, d\tau = (K * u)(t)$$

Reading:
- $e^{A(t-\tau)}$: the **matrix exponential** of $A(t-\tau)$ - the solution to the state ODE, telling us how the state evolves for a time duration $(t - \tau)$
- $\int_0^t \cdots d\tau$: integrate over all past times $\tau$ from 0 to $t$
- $K(t) = Ce^{At}B$: the **impulse response** or **convolutional kernel** - how the system responds to an impulse input at time 0, as a function of elapsed time $t$
- $(K * u)(t)$: the convolution of kernel $K$ with input $u$ - the output is the input, filtered through the system's impulse response

**Why convolution matters for computation:** Convolutions can be computed very efficiently using Fast Fourier Transforms (FFT). This means that for a fixed kernel $K$, we can process a sequence of length $T$ in $O(T \log T)$ time - much better than the $O(T^2)$ of attention.



## 8.3 Discretization: From Continuous Physics to Discrete Sequences

Neural networks process discrete sequences of tokens, not continuous signals. To apply an SSM to text or audio, we must **discretize** the continuous-time equations - convert them into a step-by-step recurrence.

### 8.3.1 The Zero-Order Hold Method

The standard discretization uses **zero-order hold (ZOH)** - assume the input $u(t)$ is constant within each time step of width $\Delta$ (a learnable "timescale" parameter):

$$\bar{A} = e^{\Delta A}$$

$$\bar{B} = (e^{\Delta A} - I)\, A^{-1} B$$

Reading:
- $\Delta$ (Delta): the step size - how long each discrete time step represents in continuous time. This is a learnable parameter that controls the "temporal resolution" of the model
- $e^{\Delta A}$: the matrix exponential - the exact evolution of the state over one time step $\Delta$. For small $\Delta$: $e^{\Delta A} \approx I + \Delta A$ (first-order approximation)
- $\bar{A}$: the **discrete state matrix** - maps the state from one step to the next
- $\bar{B}$: the **discrete input matrix** - derived from the continuous $A$ and $B$; for small $\Delta$: $\bar{B} \approx \Delta B$
- $I$: the identity matrix (the "do nothing" transformation)
- $A^{-1}$: the inverse of $A$ (exists when $A$ is invertible)

**The resulting discrete recurrence:**

$$\mathbf{h}_k = \bar{A}\, \mathbf{h}_{k-1} + \bar{B}\, u_k$$

$$y_k = C\, \mathbf{h}_k$$

Reading:
- $\mathbf{h}_k$: the hidden state at discrete time step $k$ (after processing $k$ tokens)
- $\bar{A}\, \mathbf{h}_{k-1}$: the previous state, evolved forward by one step through $\bar{A}$
- $\bar{B}\, u_k$: the current input $u_k$, integrated into the state
- $y_k = C\, \mathbf{h}_k$: read the output from the current state

**This is a linear RNN!** The discrete SSM is structurally identical to a vanilla RNN with weight matrices $\bar{A}$ (recurrent) and $\bar{B}$ (input). But with a crucial difference: the matrices are *structured* - derived from the physics-inspired continuous-time equations, not unconstrained random matrices. This structure is what enables efficient computation.

<!-- DIAGRAM: [Two-level diagram. Top: continuous-time SSM - a curved arrow labeled "$\dot{\mathbf{h}} = A\mathbf{h} + Bu$" flowing from left to right, with a continuous input signal $u(t)$ shown as a wavy curve below, and continuous output $y(t)$ above. Bottom: discrete-time SSM - a chain of boxes labeled "$\mathbf{h}_0$, $\mathbf{h}_1$, $\mathbf{h}_2$, …$\mathbf{h}_T$" connected by arrows labeled "$\times\bar{A}$"; discrete inputs $u_1, u_2, \ldots$ entering from below via "$\times\bar{B}$"; outputs $y_1, y_2, \ldots$ exiting above via "$C$". A vertical arrow between the two levels labeled "Discretize (ZOH, step $\Delta$)" Caption: "The same system, expressed in continuous time (for theoretical analysis) and discrete time (for computation). Discretization converts the differential equation into a recurrence - a linear RNN".] -->



## 8.4 The Dual Computation Modes: The Key Efficiency Innovation

The discrete SSM has two mathematically equivalent ways to compute its outputs. The choice of which to use depends on the setting:

### 8.4.1 Recurrent Mode: Linear-Time Inference

For generating outputs one token at a time (autoregressive generation), use the recurrence directly:

$$\mathbf{h}_k = \bar{A}\, \mathbf{h}_{k-1} + \bar{B}\, u_k, \qquad y_k = C\, \mathbf{h}_k$$

**Cost per step:** $O(N^2)$ for the matrix-vector product $\bar{A}\mathbf{h}_{k-1}$ (matrix size $N \times N$, vector size $N$).
**Total cost for $T$ steps:** $O(TN^2)$ - **linear in $T$**.
**Memory:** Just the current state $\mathbf{h}_k \in \mathbb{R}^N$ - **constant**, independent of sequence length.

Compare to a Transformer generating $T$ tokens: it must store and re-attend to all $T$ previous tokens' keys and values, requiring $O(T \cdot d)$ memory and $O(T \cdot d)$ computation per new token.

### 8.4.2 Convolutional Mode: Parallel Training

For processing a full sequence at once during training, unroll the recurrence starting from $\mathbf{h}_0 = \mathbf{0}$:

$$y_k = C\bar{A}^k \bar{B}\, u_0 + C\bar{A}^{k-1}\bar{B}\, u_1 + \cdots + C\bar{B}\, u_k = \sum_{j=0}^{k} \underbrace{C\bar{A}^{k-j}\bar{B}}_{\bar{K}_{k-j}}\, u_j$$

Reading:
- $\bar{A}^k$: the matrix $\bar{A}$ multiplied by itself $k$ times - the state evolution over $k$ steps
- The sum: the output at step $k$ is a weighted sum of all past inputs $u_0, u_1, \ldots, u_k$
- $\bar{K}_j = C\bar{A}^j\bar{B}$: the weight given to an input $j$ steps in the past - this is the **convolutional kernel**

Define the full kernel: $\bar{K} = (\bar{K}_0, \bar{K}_1, \ldots, \bar{K}_{T-1})$. Then the output sequence:

$$\mathbf{y} = \bar{K} * \mathbf{u}$$

This is a **causal 1D convolution** - the output at position $k$ is a weighted sum of inputs $u_0, \ldots, u_k$ with kernel weights $\bar{K}_{k}, \bar{K}_{k-1}, \ldots, \bar{K}_0$.

**Computing the kernel:** To use this formulation, we need $\bar{K}$. For a diagonal $\bar{A}$ with eigenvalues $\lambda_1, \ldots, \lambda_N$:

$$\bar{K}_j = C\bar{A}^j\bar{B} = \sum_{n=1}^{N} C_n \lambda_n^j B_n$$

Reading: a sum of geometric sequences in $j$ - one per eigenvalue. Computing all $T$ kernel values $\bar{K}_0, \ldots, \bar{K}_{T-1}$ requires evaluating $N$ geometric sequences at $T$ points - achievable in $O(N + T)$ operations.

Once the kernel is computed, applying it to the input sequence uses an FFT-based convolution in $O(T \log T)$ time - **all $T$ outputs computed in parallel**, no sequential dependency.

| Mode | When to use | Time complexity | Memory |
|------|-------------|-----------------|--------|
| **Recurrent** | Inference (one token at a time) | $O(TN^2)$ total, $O(N^2)$ per step | $O(N)$ - just the state |
| **Convolutional** | Training (full sequence at once) | $O(T\log T)$ with FFT | $O(T \cdot N)$ for the sequence |

**This is the breakthrough:** The same model switches between computation modes depending on the context. Training uses the parallelizable convolutional mode (full GPU utilization). Inference uses the efficient recurrent mode (constant memory, linear time).



## 8.5 The HiPPO Matrix: Optimal Memory Through Mathematics

Linear SSMs with random state matrices $A$ cannot reliably remember the distant past - the state vector $\mathbf{h}(t)$ loses information about old inputs as new ones arrive. The choice of $A$ is not arbitrary; it determines the model's memory capacity.

### 8.5.1 The Memory Problem, Formally

Suppose we want the state $\mathbf{h}(t)$ to encode a summary of all past inputs $u(0), u(1), \ldots, u(t)$ that allows us to reconstruct any function of the history as accurately as possible. What $A$ achieves this optimally?

This is a precise mathematical optimization problem. The **HiPPO** framework (High-Order Polynomial Projection Operators, Gu et al., 2020) solves it for polynomial function spaces.

### 8.5.2 The HiPPO-LegS Matrix

For the "Legendre Scaled" (LegS) measure, the optimal $A$ is:

$$A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

Reading (entry by entry):
- $A_{nk}$: the entry at row $n$, column $k$ of the matrix $A$
- When $n > k$ (below the diagonal): $-(2n+1)^{1/2}(2k+1)^{1/2}$ - a negative number whose magnitude depends on both $n$ and $k$
- When $n = k$ (the diagonal): $-(n+1)$ - a negative number proportional to the row index
- When $n < k$ (above the diagonal): $0$ - upper triangular entries are zero

This produces a **lower triangular** matrix with specific structured entries. When the SSM uses this $A$, the state $\mathbf{h}(t)$ maintains the optimal polynomial approximation of the entire input history up to time $t$.

**Intuition:** The state vector dimensions correspond to polynomial coefficients. Lower-order components ($n = 0, 1, 2$) capture broad, slowly-varying trends in the input history. Higher-order components ($n = N-1$) capture fine, rapidly-varying details. Together, the state provides a "multi-resolution" summary of the history - like a compression scheme that preserves both the broad strokes and the recent details.

**Why this matters practically:** Initializing the SSM's state matrix as HiPPO (rather than random or identity) dramatically improves performance on long-range tasks - the model starts with a mathematically optimal memory structure rather than having to learn it from scratch.



## 8.6 S4: Structured State Spaces for Sequences

**S4** (Gu et al., 2021) was the first SSM architecture to demonstrate performance competitive with Transformers on long-sequence tasks. The key innovation was enabling efficient computation of the convolutional kernel.

### 8.6.1 The Diagonalization Approach

For a general matrix $A$, computing $\bar{K}_j = C\bar{A}^j\bar{B}$ requires matrix exponentiation - expensive. S4 uses a structural property: if $A$ has a specific "diagonal plus low-rank" structure (DPLR), it can be efficiently diagonalized.

For a diagonalizable $A = P\Lambda P^{-1}$ (where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$ contains the eigenvalues):

$$\bar{A}^j = P \Lambda^j P^{-1} = P\, \text{diag}(\lambda_1^j, \ldots, \lambda_N^j)\, P^{-1}$$

Reading: $\bar{A}^j$ is just the diagonal matrix of eigenvalues each raised to the $j$-th power, sandwiched by the change-of-basis matrices $P$ and $P^{-1}$.

The kernel entry:

$$\bar{K}_j = C\bar{A}^j\bar{B} = (CP)\, \text{diag}(\lambda_1^j, \ldots, \lambda_N^j)\, (P^{-1}\bar{B}) = \sum_{n=1}^{N} \tilde{C}_n \lambda_n^j \tilde{B}_n$$

Reading: $\tilde{C}_n = (CP)_n$ and $\tilde{B}_n = (P^{-1}\bar{B})_n$ are fixed vectors (computable once from $C$, $P$, $\bar{B}$). The kernel is a sum of $N$ terms, each a geometric sequence $\lambda_n^j$ scaled by a constant $\tilde{C}_n \tilde{B}_n$.

Computing all $T$ values of the kernel: evaluate $N$ geometric sequences at powers $j = 0, 1, \ldots, T-1$. This is a Vandermonde matrix-vector product - $O(N \log T)$ with efficient algorithms, compared to the naive $O(N^2 T)$.

S4 parameterizes $A$ in terms of its eigenvalues (constrained to have negative real parts - ensures stability) and the transformation matrices, enabling efficient kernel computation while preserving the HiPPO memory structure.

### 8.6.2 S4 Performance

S4 achieved strong results on the **Long Range Arena** benchmark - tasks requiring context lengths of 1,000–16,000 tokens:

- **PathFinder** (long-range pixel dependency detection): S4 dramatically outperformed all Transformer variants, reaching 88% accuracy where Transformers hovered near chance
- **Sequential image classification** ($28 \times 28$ MNIST treated as a 784-step sequence): S4 reached 98% accuracy; Transformers struggled
- **Audio modeling** at sample level (44kHz - sequences of 160,000+ samples): only feasible for SSMs

But S4 had one significant limitation that prevented it from becoming a universal architecture: **its state transition is input-independent**.

The matrices $\bar{A}$, $\bar{B}$, $C$ do not change based on what the input is. Every token - "the", "cat", "quantum" - is processed through the same dynamics. This is appropriate for a physical filter (a guitar equalizer applies the same frequency shaping regardless of what notes are played), but severely limiting for language modeling (the word "bank" should activate different memory dynamics depending on context).



## 8.7 Mamba: Selectivity Changes Everything

**Mamba** (Gu & Dao, 2023) introduces the key innovation that makes SSMs competitive with Transformers on language modeling: **make the state space parameters depend on the input**.

### 8.7.1 The Selective Mechanism

In Mamba, the parameters $B$, $C$, and $\Delta$ are functions of the current input token $u_k$:

$$B_k = s_B(u_k), \qquad C_k = s_C(u_k), \qquad \Delta_k = \text{softplus}(s_\Delta(u_k) + \text{parameter}_\Delta)$$

Reading:
- $s_B$, $s_C$, $s_\Delta$: simple linear projections (one linear layer each)
- $s_B(u_k)$: map the current token embedding to a $B$ matrix for this step
- $s_C(u_k)$: map the current token to a $C$ (output projection) for this step
- $\text{softplus}(x) = \log(1 + e^x)$: a smooth, always-positive function - ensures $\Delta_k > 0$ (the step size must be positive)
- $\Delta_k$: the step size for this token - how much this token's input matters relative to the accumulated state

The discrete transition matrices become token-dependent:

$$\bar{A}_k = e^{\Delta_k A}, \qquad \bar{B}_k \approx \Delta_k B_k$$

Reading:
- $\bar{A}_k = e^{\Delta_k A}$: the discrete state matrix for step $k$. When $\Delta_k$ is large: $e^{\Delta_k A}$ is "far" from the identity - the state changes significantly. When $\Delta_k$ is small: $e^{\Delta_k A} \approx I + \Delta_k A \approx I$ - the state barely changes
- $\bar{B}_k \approx \Delta_k B_k$: the discrete input matrix is proportional to $\Delta_k$ - when $\Delta_k$ is small, the input barely enters the state

The recurrence is now **time-varying**:

$$\mathbf{h}_k = \bar{A}_k\, \mathbf{h}_{k-1} + \bar{B}_k\, u_k, \qquad y_k = C_k\, \mathbf{h}_k$$

### 8.7.2 What Selectivity Enables

With input-dependent parameters, Mamba can:

**Selectively ignore tokens:** When $\Delta_k \to 0$: $\bar{A}_k \to I$ (identity) and $\bar{B}_k \to 0$. The state update becomes $\mathbf{h}_k = I \cdot \mathbf{h}_{k-1} + 0 \cdot u_k = \mathbf{h}_{k-1}$ - the state is unchanged, the token is effectively skipped. This is the model saying "this token carries no new relevant information".

**Selectively memorize tokens:** When $\Delta_k$ is large: $\bar{A}_k \approx 0$ and $\bar{B}_k$ is large. The update becomes $\mathbf{h}_k \approx \bar{B}_k u_k$ - the previous state is erased and replaced by the new token's information. This is the model saying "this token is so important it overwrites the existing memory".

**Selectively read from memory:** The output $y_k = C_k \mathbf{h}_k$ uses a different $C_k$ at every step. The model can ask different "questions" of its memory at different positions - "what subject noun have we seen?" at a verb, "what object noun?" at a pronoun, etc.

**Comparison to LSTM gates:** LSTM gates (Chapter 6) also perform selective memory - the forget gate decides what to retain, the input gate what to write. Mamba's selectivity is derived from SSM principles rather than heuristic gate design, and it operates at a different level: the entire state transition dynamics change, not just the gating of existing dimensions.

### 8.7.3 The Parallel Scan Algorithm: Making Selectivity Trainable

The input-dependence of $\bar{A}_k$ and $\bar{B}_k$ destroys the convolutional formulation - the kernel $\bar{K}_j = C\bar{A}^j\bar{B}$ was fixed for time-invariant SSMs, but now it changes at every step. How do we train efficiently?

The key: the recurrence $\mathbf{h}_k = \bar{A}_k \mathbf{h}_{k-1} + \bar{B}_k u_k$ is a **linear associative recurrence**. Despite the varying coefficients, it satisfies the associativity property that enables parallel computation.

**Define a "combination operator" $\otimes$ on pairs:**

$$(A_1, b_1) \otimes (A_2, b_2) = (A_1 \cdot A_2, \; A_1 b_2 + b_1)$$

Reading: the $\otimes$ operator takes two (matrix, vector) pairs and combines them. The combined matrix is the matrix product; the combined vector is the first matrix applied to the second vector plus the first vector.

**Why this matters:** The hidden state sequence satisfies:

$$\mathbf{h}_k = (\bar{A}_k, \bar{B}_k u_k) \otimes (\bar{A}_{k-1}, \bar{B}_{k-1} u_{k-1}) \otimes \cdots \otimes (\bar{A}_1, \bar{B}_1 u_1) \otimes (\mathbf{h}_0)$$

The entire hidden state sequence is a sequence of $\otimes$ operations applied to the local transitions. Because $\otimes$ is **associative** - $(a \otimes b) \otimes c = a \otimes (b \otimes c)$ - we can compute all prefix products in parallel using a tree structure.

**The parallel prefix scan (Blelloch, 1990):**

For a sequence of $T$ elements with an associative operation $\otimes$, compute all prefix "products" (prefixes $A_1$, $A_1 \otimes A_2$, $A_1 \otimes A_2 \otimes A_3$, ...) in $O(\log T)$ parallel steps:

**Up-sweep (reduce) phase:**
- Level 0: $T$ leaf nodes, each holding $(\bar{A}_k, \bar{B}_k u_k)$ for $k = 1, \ldots, T$
- Level 1: combine adjacent pairs: $(A_1, b_1) \otimes (A_2, b_2)$, $(A_3, b_3) \otimes (A_4, b_4)$, etc. → $T/2$ nodes
- Level 2: combine adjacent pairs of level-1 results → $T/4$ nodes
- ... continue for $\log_2 T$ levels → 1 root node containing $\otimes$-product of all elements

**Down-sweep phase:**
- Distribute prefix products downward through the tree
- Each node receives the $\otimes$-product of all elements to its left

After the down-sweep: node $k$ holds the $\otimes$-product $(\bar{A}_k \cdots \bar{A}_1, \text{combined}\ b)$, which directly gives $\mathbf{h}_k$.

**Total computation:** $O(T)$ operations in $O(\log T)$ parallel steps - compared to the $O(T)$ sequential steps of the vanilla recurrence. On modern GPUs with thousands of parallel cores, this $O(\log T)$ depth is highly efficient.

<!-- DIAGRAM: [Binary tree structure. Bottom row: 8 leaf nodes labeled "$(\bar{A}_1, \bar{B}_1 u_1)$, $(\bar{A}_2, \bar{B}_2 u_2)$, ..., $(\bar{A}_8, \bar{B}_8 u_8)$". Level above: 4 nodes, each labeled as the $\otimes$-combination of two leaves below. Level above that: 2 nodes, each combining two of the previous level. Root: 1 node combining the two above it. Right side of diagram: arrows showing the "down-sweep" distributing prefix products. Caption: "The parallel prefix scan computes all T hidden states simultaneously in $O(\log T)$ parallel steps. This is why Mamba can be trained as efficiently as a Transformer (parallel) while being as efficient as an RNN at inference (recurrent)".] -->

### 8.7.4 Hardware-Aware Implementation: The Engineering That Makes It Work

Beyond the algorithmic efficiency, Mamba's practical performance depends critically on hardware-aware implementation. Modern GPUs have a memory hierarchy:

- **HBM (High Bandwidth Memory)**: Large (40–80 GB) but slow (seconds-level latency for frequent small accesses)
- **SRAM (Shared Memory)**: Small (hundreds of KB per streaming multiprocessor) but extremely fast

Standard implementations repeatedly load/store intermediate computations to HBM - expensive. Mamba's implementation fuses the entire selective scan into a single CUDA kernel that keeps all intermediate state computations in SRAM, only reading inputs and writing outputs to HBM.

This "kernel fusion" reduces memory bandwidth requirements dramatically - not by doing less computation, but by doing the same computation more cleverly with respect to memory access patterns. The result: Mamba achieves higher throughput than a naive implementation by a factor of 3–5×, and is competitive with highly optimized Transformer implementations.



## 8.8 The Complete Mamba Block

A full Mamba layer takes a sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$ and processes it through:

**Step 1 - Linear expansion:**
$$\mathbf{U}, \mathbf{V} = \text{Linear}(\mathbf{X}) \qquad \mathbf{U}, \mathbf{V} \in \mathbb{R}^{T \times E}$$
Expand to dimension $E = 2d$ (two branches with the same expanded dimension).

**Step 2 - Short convolution on $\mathbf{U}$:**
$$\mathbf{U}' = \text{DepthwiseConv1D}(\mathbf{U})$$
A short (width 4) causal depthwise 1D convolution - adds local context, improves performance slightly.

**Step 3 - SiLU activation:**
$$\mathbf{U}'' = \text{SiLU}(\mathbf{U}')$$
Where $\text{SiLU}(x) = x \cdot \sigma(x)$ - a smooth, non-monotone activation.

**Step 4 - Compute selective SSM parameters:**
$$B_k = s_B(u_k''), \quad C_k = s_C(u_k''), \quad \Delta_k = \text{softplus}(s_\Delta(u_k'') + \text{param}_\Delta)$$
Simple linear projections, different for each token.

**Step 5 - Selective scan:**
$$\mathbf{Y}_{\text{SSM}} = \text{SelectiveScan}(\mathbf{U}'', \bar{A}_k, \bar{B}_k, C_k)$$
Apply the parallel scan algorithm using the token-dependent parameters.

**Step 6 - Gating with the second branch:**
$$\mathbf{Z} = \mathbf{Y}_{\text{SSM}} \odot \text{SiLU}(\mathbf{V})$$
Element-wise multiply the SSM output with SiLU-activated $\mathbf{V}$. This creates a "gated" output - $\mathbf{V}$ acts as a soft switch that can zero out SSM outputs that are not relevant.

**Step 7 - Linear projection back:**
$$\text{Output} = \text{Linear}(\mathbf{Z}) \in \mathbb{R}^{T \times d}$$
Project back to the original dimension $d$.

Multiple such blocks are stacked with residual connections to form the full Mamba model.



## 8.9 Why Selectivity = Soft Attention (The Theoretical Connection)

A remarkable theoretical result from Mamba-2 (Dao & Gu, 2024) reveals that selective SSMs and linear attention are mathematically related.

**Linear attention** replaces softmax in the attention formula with a kernel function $\phi$:

$$O_t = \frac{\sum_{s=1}^{t} \phi(Q_t)^T \phi(K_s) V_s^T}{\sum_{s=1}^{t} \phi(Q_t)^T \phi(K_s)}$$

Define the running state $S_t = \sum_{s=1}^{t} \phi(K_s) V_s^T$ (updated as new tokens arrive):

$$S_t = S_{t-1} + \phi(K_t) V_t^T$$

Reading: this is a **linear recurrence** - exactly an SSM with state $S_t$! The "state matrix" is the identity ($S_t$ accumulates without decay), and the update adds a rank-1 outer product at each step.

**Where Mamba fits:** Mamba's selective SSM corresponds to linear attention with a *decaying* accumulation (the $\bar{A}_k$ term allows old information to be forgotten) and *input-dependent* queries, keys, and values (the $C_k$ and $B_k$ terms). It is a more general form of linear attention with content-dependent forgetting.

**Significance:**
- SSMs and attention are not competing paradigms - they are points in a unified design space
- SSMs with decay ($\bar{A}_k$ near 1) ≈ attention with recency bias
- SSMs without decay ($\bar{A}_k = I$) = cumulative (linear) attention
- Standard softmax attention = SSMs with infinite state size and no approximation
- Hybrid architectures can smoothly interpolate between these extremes



## 8.10 Empirical Performance and the Limits of Each Approach

### 8.10.1 Where Mamba Excels

**Language modeling:** Mamba-3B matches or exceeds Transformer language models of similar size on standard benchmarks (Pile, LAMBADA) at sequences up to 4,096 tokens. At longer contexts, Mamba's advantage grows - Transformers degrade or become infeasible.

**Long-sequence tasks:** DNA sequence modeling (bacterial genomes: millions of nucleotides), long audio (raw waveform modeling at 16kHz), and long document understanding - regimes where Transformers cannot fit in memory.

**Inference throughput:** For autoregressive generation, Mamba's recurrent mode uses constant memory and linear time. A Transformer's KV cache grows linearly with context length, eventually becoming a memory bottleneck. For context lengths of 100K+, Mamba can be 5–10× more memory-efficient at inference.

### 8.10.2 Where Transformers Still Win

**Exact retrieval:** Tasks requiring the model to copy or look up specific tokens from a long context. Transformers attend to all positions directly; Mamba compresses everything into a fixed-size state, potentially losing specific details.

**In-context learning:** The few-shot ability of large language models relies on attention's ability to directly compare the few-shot examples with the query. Mamba's compressed state may not preserve the fine-grained structure needed for reliable in-context learning at the same level.

**Established ecosystem:** Years of optimization, fine-tuning recipes, tooling, and practical knowledge for Transformers represent an enormous practical advantage.

### 8.10.3 The Hybrid Consensus

By 2024–2025, the research consensus has converged on **hybrid architectures** that interleave SSM and attention layers:

```
[Mamba] → [Mamba] → [Attention] → [Mamba] → [Mamba] → [Attention] → ...
```

The pattern:
- SSM layers handle local patterns and build a compressed state efficiently (most of the layers)
- Occasional attention layers provide global reasoning and exact retrieval (fewer layers)

Models like Jamba achieve near-Transformer quality on all standard benchmarks while scaling to contexts of 256K+ tokens - a capability impossible for pure Transformers without massive engineering effort.



## 8.11 The Broader SSM Family

Mamba is one architecture in a rapidly evolving family:

| Architecture | Year | Key Innovation |
|---|---|---|
| **S4** | 2021 | HiPPO initialization, efficient kernel computation |
| **S4D** | 2022 | Simplified diagonal SSM |
| **Mamba / S6** | 2023 | Selective (input-dependent) parameters |
| **Mamba-2** | 2024 | Reformulation as structured matrix multiplication; unified with linear attention |
| **RWKV** | 2023 | Attention-free language model with recurrent inference |
| **RetNet** | 2023 | Retention mechanism with explicit training/inference duality |

The common thread: all seek efficient sequence processing with long-range memory. The specific mechanisms differ, but the design space they explore - between full attention and pure recurrence - is the same.



## 8.12 Summary: The Arc of Sequence Modeling

The story of sequence modeling has traced a clear arc:

**Vanilla RNNs (1980s–1990s):** Memory through recurrence. Linear time. Fails on long-range dependencies due to vanishing gradients.

**LSTMs and GRUs (1997–2014):** Gated memory. Largely solved gradient flow. Still sequential (cannot parallelize training).

**Attention mechanisms (2015):** Direct access to all past tokens. Solved long-range dependencies. $O(T^2)$ cost.

**Transformers (2017):** Attention only, no recurrence. Fully parallelizable training. Dominant for NLP and beyond. Limited by quadratic scaling.

**SSMs / S4 (2021):** Return to recurrence with structured matrices and HiPPO initialization. Linear time. Parallelizable via convolution. Limited by input-independence.

**Mamba (2023):** Selective SSMs. Input-dependent memory management. Combines linear inference efficiency with content-aware memory. Competitive with Transformers on language modeling.

**Hybrids (2024–):** Combine SSM efficiency with attention precision. The current state-of-the-art.

The deeper lesson: **each architecture is a different answer to the same fundamental question** - "how do we build a model that efficiently maintains and retrieves relevant context over long sequences?" The constraints of memory, computation, and expressivity drive the trade-offs, and the best answer changes as hardware, datasets, and applications evolve.



## 8.13 Conclusion: The Principles Behind the Architectures

This final chapter has brought us to the research frontier. But looking back across all eight chapters, a single principle runs through every architectural innovation:

**The structure of the model should match the structure of the data.**

Images are spatially local and translation-equivariant → **Convolutions**. Sequences are temporal and causal → **Recurrence**. Long-range dependencies require direct access → **Attention**. Data lies on manifolds → **Autoencoders**. Long sequences require compressed memory → **State Space Models**.

Every "Aha! Moment" at the start of each chapter is an instance of this same principle: the previous architecture failed because its structure did not match some property of the data or task. The new architecture succeeded by encoding that property directly.

Understanding *why* an architecture works - tracing it to the mathematical structure of the problem it solves - is what distinguishes a practitioner who uses these tools from a researcher who builds new ones.

The field will continue to evolve. The architectures described in this chapter will be refined, replaced, and combined in ways we cannot yet anticipate. But the principles - expressivity, efficiency, inductive bias, gradient flow, information theory - are enduring. These are the foundations from which the next generation of architectures will grow.

The open questions remain: Why do overparameterized networks generalize? What determines which representations form? Can we find architectures that are simultaneously as expressive as attention, as efficient as RNNs, and as interpretable as sparse autoencoders? These questions will occupy the field for decades, and you are now equipped to pursue them.



## Appendix: Key SSM Equations at a Glance

**Continuous-time SSM:**
$$\dot{\mathbf{h}}(t) = A\mathbf{h}(t) + Bu(t), \qquad y(t) = C\mathbf{h}(t) + Du(t)$$

**ZOH Discretization:**
$$\bar{A} = e^{\Delta A}, \qquad \bar{B} \approx \Delta B$$

**Discrete recurrence (time-invariant S4):**
$$\mathbf{h}_k = \bar{A}\mathbf{h}_{k-1} + \bar{B}u_k, \qquad y_k = C\mathbf{h}_k$$

**Convolutional kernel:**
$$\bar{K} = (C\bar{B},\, C\bar{A}\bar{B},\, C\bar{A}^2\bar{B},\, \ldots,\, C\bar{A}^{T-1}\bar{B})$$

**Selective SSM recurrence (Mamba):**
$$\mathbf{h}_k = \bar{A}_k\mathbf{h}_{k-1} + \bar{B}_k u_k, \qquad y_k = C_k\mathbf{h}_k$$
where $\bar{A}_k, \bar{B}_k, C_k$ are functions of the input token $u_k$.

**Parallel scan associative operator:**
$$(A_1, b_1) \otimes (A_2, b_2) = (A_1 A_2,\; A_1 b_2 + b_1)$$

**HiPPO-LegS matrix (optimal polynomial memory):**
$$A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & n > k \\ n+1 & n = k \\ 0 & n < k \end{cases}$$

---


