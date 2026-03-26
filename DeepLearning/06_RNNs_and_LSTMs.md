# Chapter 6: Memory and the Flow of Time - Recurrent Networks, LSTMs, and the Architecture of Sequential Thought


<div style="text-align: center; margin: 20px 0;">
  <p style="font-size: 1.4em; margin-bottom: 8px;">
    <i>"Memory is the treasury and guardian of all things"</i>
  </p>
  <p style="font-size: 0.9em; color: #777;">
    Marcus Tullius Cicero
  </p>
</div>




## 6.1 When Space Becomes Time

The world does not present itself as still photographs. Language unfolds word by word; music unfolds note by note; stock prices unfold tick by tick; a patient's health unfolds heartbeat by heartbeat. In each of these domains, the *order* of events is as meaningful as the events themselves. "The dog bit the man" and "The man bit the dog" use identical words; their meanings are opposite. The sequence is the message.

For a standard feedforward network - or even a CNN - this presents a fundamental barrier. These networks receive a fixed-size input and produce a fixed-size output. They have no **memory**: each forward pass is stateless, treating every input as if the world began at that moment. They are anterograde amnesics, incapable of connecting what is happening now to what happened a second ago.

**Recurrent Neural Networks (RNNs)** solve this by introducing a **hidden state** $h^{(t)}$ - a vector that persists across time steps, encoding the network's current understanding of the sequence's history. At each step, the network reads a new input $x^{(t)}$ and updates its hidden state based on both the new input and the previous state. The hidden state is the network's working memory, and its content at any moment is a compressed summary of everything the network has read so far.

This chapter traces the full arc: from the deceptively simple recurrence equation that defines basic RNNs, through the mathematical crisis of vanishing gradients that nearly killed the field, to the elegant gating mechanisms of LSTMs and GRUs that resolved it, and finally to the modern landscape of state space models that are rewriting the relationship between recurrence and parallelism.



## 6.2 The Basic Recurrence: A Dynamical System Perspective

The defining equation of a vanilla (Elman) RNN is:

$$h^{(t)} = \sigma\!\left(W_{hh} h^{(t-1)} + W_{xh} x^{(t)} + b_h\right)$$

$$y^{(t)} = W_{hy} h^{(t)} + b_y$$

where $h^{(t)} \in \mathbb{R}^{d_h}$ is the hidden state at time $t$, $x^{(t)} \in \mathbb{R}^{d_x}$ is the input, and $y^{(t)} \in \mathbb{R}^{d_y}$ is the output. The matrices $W_{hh}$ (recurrent), $W_{xh}$ (input-to-hidden), and $W_{hy}$ (hidden-to-output) are shared across all time steps - the same "rules" apply at each moment, regardless of position in the sequence.

The **unrolled view** reveals the architecture's implicit depth. Substituting repeatedly:

$$h^{(T)} = \sigma\!\left(W_{hh} \sigma\!\left(W_{hh} \cdots \sigma\!\left(W_{hh} h^{(0)} + W_{xh} x^{(1)} + b_h\right) \cdots + W_{xh} x^{(T-1)} + b_h\right) + W_{xh} x^{(T)} + b_h\right)$$

The hidden state $h^{(T)}$ is a highly nonlinear function of every input $x^{(1)}, \ldots, x^{(T)}$. The network is not shallow - it is a $T$-layer deep network in time, where every layer shares the same weight matrices. Training on a sequence of 100 time steps is equivalent to training a 100-layer network with weight tying.

This unrolled view also reveals the computational complexity. Forward pass: $\mathcal{O}(T \cdot (d_h^2 + d_h d_x + d_h d_y))$, dominated by the recurrent matrix multiply $W_{hh} h^{(t-1)}$. Unlike CNNs, which can parallelize across spatial positions, RNNs are **inherently sequential**: $h^{(t)}$ cannot be computed until $h^{(t-1)}$ is available. This sequential bottleneck is the central limitation that Transformers ultimately resolved.

### The Dynamical Systems Interpretation

Beyond the computational view, RNNs are best understood as **discrete-time dynamical systems**. The hidden state $h^{(t)}$ is a point in a $d_h$-dimensional state space, and the recurrence equation defines its trajectory. Understanding RNN behavior requires understanding the geometry of this trajectory.

**Fixed points** $h^*$ satisfy $h^* = \sigma(W_{hh}h^* + b)$ for zero input. They are the "resting positions" of the system. The stability of a fixed point is determined by the Jacobian at that point: if all eigenvalues of $J = \text{diag}(\sigma'(z^*)) W_{hh}$ have magnitude less than 1, the fixed point is a stable attractor; if any eigenvalue exceeds 1 in magnitude, it is unstable.

Multiple fixed points correspond to multiple distinct "memories" the RNN can store. The process of reading a sequence is the process of navigating among these attractors - each word or event perturbs the hidden state, and the network's "understanding" at each moment is which attractor it is currently near or approaching.

**Chaos** is possible: for certain weight configurations, the RNN's trajectory is sensitive to initial conditions and inputs, making it computationally expressive but hard to control during training. The "edge of chaos" - the transition between stable and chaotic dynamics - is empirically the regime where RNNs achieve their best information processing capacity.

<DIAGRAM: A state-space portrait of a simple 2D RNN. Multiple fixed points are shown as colored dots (stable attractors as filled circles, unstable points as open circles). Trajectories starting from different initial conditions are shown as curved arrows converging to different attractors. A sequence of inputs is shown perturbing the trajectory between attractors. Annotations explain the correspondence between attractors and memories.>



## 6.3 Backpropagation Through Time: Training in Sequence

Training an RNN requires gradients of the loss with respect to all parameters. The algorithm is **Backpropagation Through Time (BPTT)**: mentally unroll the RNN into a $T$-layer network and apply standard backpropagation.

The total loss $\mathcal{L} = \sum_{t=1}^T \mathcal{L}^{(t)}$ sums contributions from every time step (for sequence labeling) or uses only the final step (for sequence classification). The gradient with respect to $W_{hh}$ accumulates contributions from all time steps:

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}^{(t)}}{\partial W_{hh}} = \sum_{t=1}^T \sum_{k=1}^t \delta^{(k)} (h^{(k-1)})^T$$

where $\delta^{(k)} = \frac{\partial \mathcal{L}^{(t)}}{\partial z^{(k)}}$ is the error signal at step $k$ due to the loss at step $t$. Propagating this signal from step $t$ backward to step $k$ requires the product:

$$\delta^{(k)} = \frac{\partial \mathcal{L}^{(t)}}{\partial h^{(t)}} \cdot \prod_{j=k}^{t-1} \frac{\partial h^{(j+1)}}{\partial h^{(j)}} = \frac{\partial \mathcal{L}^{(t)}}{\partial h^{(t)}} \cdot \prod_{j=k}^{t-1} \text{diag}(\sigma'(z^{(j+1)})) W_{hh}$$

This product of $t - k$ copies of $\text{diag}(\sigma') W_{hh}$ is where the crisis lives.

### Teacher Forcing

At training time, we have access to the true output at each step. **Teacher forcing** uses the ground truth $y^{(t-1)}$ as the next input rather than the model's own prediction $\hat{y}^{(t-1)}$. This stabilizes early training when the model's predictions are poor - a bad prediction at step 5 would corrupt the hidden state for all subsequent steps.

The cost is **exposure bias**: at inference time, the model sees its own (potentially wrong) predictions. If the training distribution (teacher-forced) differs from the inference distribution (self-predicted), performance degrades. **Scheduled sampling** anneals the proportion of teacher forcing, gradually transitioning from 100% teacher forcing early in training to 0% later.



## 6.4 The Vanishing Gradient Crisis: The Mathematical Event Horizon

The central failure of vanilla RNNs is their inability to learn long-range dependencies. The mathematical explanation, given rigorously by Hochreiter (1991) and Bengio et al. (1994), is the exponential attenuation of gradients over time.

The product of Jacobians $\prod_{j=k}^{t-1} J_j$ where $J_j = \text{diag}(\sigma'(z^{(j+1)})) W_{hh}$ determines gradient flow. Let $\lambda_{\max}$ be the largest singular value of $W_{hh}$ and $\gamma = \max_z |\sigma'(z)|$. Then:

$$\left\|\prod_{j=k}^{t-1} J_j\right\| \leq (\lambda_{\max} \cdot \gamma)^{t-k}$$

For $\text{tanh}$, $\gamma \leq 1$, so $\lambda_{\max} \cdot \gamma < 1$ when $\lambda_{\max} < 1$. Over $t - k = 100$ steps, the gradient decays by $(\lambda_{\max} \gamma)^{100} \approx 0$ for any product less than 1. Early time steps receive zero gradient and the network cannot learn that any event 100 steps ago was relevant.

The fundamental dilemma (Bengio et al., 1994): **storing information robustly requires the dynamics to be stable** (attractors with eigenvalues $\approx 1$), but **learning requires gradient flow** that is also $\approx 1$ in magnitude. Near an attractor, $J \approx I$ and gradients neither vanish nor explode - but this is an unstable configuration that requires fine-tuning weights to a very precise value.

This is not merely a practical problem with better initialization or optimization - it is a fundamental mathematical tension built into the architecture itself.



## 6.5 Long Short-Term Memory: Engineering a Gradient Highway

The solution was architectural. In 1997, Hochreiter and Schmidhuber published **Long Short-Term Memory (LSTM)**, introducing a dedicated memory cell $c^{(t)}$ that is protected from the multiplicative gradient decay that plagues the hidden state.

The LSTM's key insight is that the gradient of $c^{(t)}$ with respect to $c^{(t-1)}$ should be close to the **identity** - not zero, not a large eigenvalue matrix, but the identity. This is achieved through **additive** cell state updates controlled by learned gates:

$$c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}$$

The gradient is:

$$\frac{\partial c^{(t)}}{\partial c^{(t-1)}} = \text{diag}(f^{(t)})$$

If the forget gate $f^{(t)} \approx \mathbf{1}$ (all ones), this Jacobian is the identity. Gradients flow backward through the cell state without attenuation. This is the **Constant Error Carousel** (Hochreiter's term) - a protected highway for gradients that bypasses the multiplicative collapse of the hidden state path.

### The Four Gates in Detail

The LSTM combines four learned functions, all computed from the same inputs $[h^{(t-1)}, x^{(t)}]$:

**Forget gate** $f^{(t)} = \sigma(W_f [h^{(t-1)}, x^{(t)}] + b_f)$: Values near 0 erase the corresponding dimension of the cell state; values near 1 preserve it. The network learns to forget irrelevant past information when new information supersedes it - forget the subject of the previous sentence when a new sentence begins.

**Input gate** $i^{(t)} = \sigma(W_i [h^{(t-1)}, x^{(t)}] + b_i)$: Controls which dimensions of the candidate state to write. Values near 0 prevent new information from entering; values near 1 allow it.

**Candidate state** $\tilde{c}^{(t)} = \tanh(W_c [h^{(t-1)}, x^{(t)}] + b_c)$: The proposed new content, using tanh to keep values bounded in $(-1, 1)$.

**Cell state update**: $c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}$. The forget gate decides what to erase from memory; the input gate decides what new information to write. Together, they implement a learned read-write operation on the cell state.

**Output gate** $o^{(t)} = \sigma(W_o [h^{(t-1)}, x^{(t)}] + b_o)$: Controls which parts of the cell state to expose as the hidden state.

**Hidden state**: $h^{(t)} = o^{(t)} \odot \tanh(c^{(t)})$. The output gate filters the cell state through tanh (bounding it in $(-1, 1)$) and selects which dimensions to expose.

<DIAGRAM: A detailed LSTM cell diagram. Shows the four gate computations on the left (each as a sigmoid or tanh operation). The cell state c^(t) is shown as a horizontal "conveyor belt" through the top, with the forget gate as a multiplicative gate (f · c^(t-1)) and the input gate as an additive gate (i · c_tilde). The output gate extracts h^(t) from c^(t) at the right. All connections from [h^(t-1), x^(t)] are shown entering from the bottom.>

### The Conveyor Belt Analogy

Think of the cell state as a conveyor belt running through a factory. Information - factory products - sits on the belt and is carried forward automatically. The **forget gate** is a worker who can push products off the belt (set to zero) when they're no longer needed. The **input gate** determines which new products can be placed on the belt. The **output gate** determines which products on the belt are visible to the next department.

With the conveyor belt running smoothly (forget gate $\approx 1$), information from 500 steps ago can still be on the belt when needed. This is why LSTMs can bridge dependencies that vanilla RNNs cannot: the highway for information is protected by gates that the network learns to use appropriately.

### Expressiveness of LSTMs

LSTMs are strictly more computationally powerful than vanilla RNNs for discrete tasks. The cell state's additive update allows LSTMs to implement **counters** exactly - incrementing $c^{(t)}$ by 1 each time a certain input appears, and comparing to a threshold at the end. Vanilla RNNs with tanh activation cannot count exactly, because the tanh saturates as the counter value grows.

This counting ability places LSTMs in a higher position in the Chomsky hierarchy. Vanilla RNNs can recognize **regular languages** (finite automata). LSTMs can recognize **counter languages** - including many context-free languages like balanced parentheses - making them strictly more expressive for formal language tasks.



## 6.6 Gated Recurrent Units: Streamlined Memory

In 2014, Cho et al. introduced the **Gated Recurrent Unit (GRU)** as a simplified alternative to the LSTM. The GRU merges the cell state and hidden state into a single state $h^{(t)}$, and reduces the gate count from three to two:

**Reset gate** $r^{(t)} = \sigma(W_r [h^{(t-1)}, x^{(t)}] + b_r)$: Controls how much of the previous state enters the candidate. Setting $r = 0$ makes the candidate ignore the past entirely, computing a fresh representation from only $x^{(t)}$.

**Update gate** $z^{(t)} = \sigma(W_z [h^{(t-1)}, x^{(t)}] + b_z)$: Simultaneously controls forgetting and updating. Values near 0 make the GRU ignore the candidate and retain the old state; values near 1 replace the old state with the candidate.

**Candidate state**: $\tilde{h}^{(t)} = \tanh(W [r^{(t)} \odot h^{(t-1)}, x^{(t)}] + b)$

**State update**: $h^{(t)} = (1 - z^{(t)}) \odot h^{(t-1)} + z^{(t)} \odot \tilde{h}^{(t)}$

The gradient of $h^{(t)}$ with respect to $h^{(t-1)}$:

$$\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \text{diag}(1 - z^{(t)}) + \text{diag}(z^{(t)}) \frac{\partial \tilde{h}^{(t)}}{\partial h^{(t-1)}}$$

When $z^{(t)} \approx 0$, this Jacobian is approximately $I$ - the same constant-error-carousel property as the LSTM's cell state.

The GRU has 3 weight matrices per cell (versus 4 for the LSTM), making it approximately 25% faster. Empirically, GRU and LSTM perform comparably across most tasks, with LSTM occasionally outperforming on tasks requiring very long dependencies. The practitioner's rule: start with GRU for speed; upgrade to LSTM if necessary.



## 6.7 Bidirectionality and Depth: Scaling Recurrent Networks

### Bidirectional RNNs

Standard RNNs are causal - $h^{(t)}$ depends on $x^{(1)}, \ldots, x^{(t)}$ but not on future inputs. For offline tasks (processing fixed sequences like translation or sentiment classification), this is an unnecessary constraint.

**Bidirectional RNNs (BiRNNs)** run two separate recurrent networks in opposite directions:

$$\overrightarrow{h}^{(t)} = \text{RNN}_\text{fwd}(x^{(1:t)}), \qquad \overleftarrow{h}^{(t)} = \text{RNN}_\text{bkw}(x^{(T:t)})$$

The final representation at position $t$ is the concatenation $[{\overrightarrow{h}^{(t)}}, {\overleftarrow{h}^{(t)}}]$, capturing context from both past and future. BiRNNs are strictly more powerful than unidirectional RNNs for any task where the label depends on both past and future context - which includes virtually all NLP tasks except real-time generation.

BiRNNs are the foundation of **BERT's** encoding architecture: the "bidirectional" in BERT refers to bidirectional context in the attention mechanism, which generalizes the idea of BiRNNs to allow any position to attend to any other position.

### Deep RNNs

Just as stacking CNN layers enables hierarchical spatial feature learning, stacking RNN layers enables hierarchical temporal feature learning:

$$h_\ell^{(t)} = \text{RNN}_\ell(h_{\ell-1}^{(t)}, h_\ell^{(t-1)})$$

where $h_0^{(t)} = x^{(t)}$. The first layer extracts low-level temporal patterns; deeper layers build increasingly abstract temporal structure. A 3-layer BiLSTM is typical for competitive sequence-to-sequence tasks.

**Residual connections** are also effective for deep RNNs: $h_\ell^{(t)} = \text{RNN}_\ell(h_{\ell-1}^{(t)}) + h_{\ell-1}^{(t)}$, enabling stable training with many stacked layers.



## 6.8 Sequence-to-Sequence: The Architecture of Translation

Many of the most valuable sequence tasks involve mapping one sequence to another of different length - translation, summarization, question answering. The standard architecture is the **Encoder-Decoder** (Sutskever et al., 2014; Cho et al., 2014):

**Encoder**: An RNN reads the source sequence $x^{(1:T_x)}$ and produces a **context vector** $c = h_{\text{enc}}^{(T_x)}$ - the final hidden state encoding the entire source.

**Decoder**: A separate RNN is initialized with $c$ and generates the target sequence $y^{(1:T_y)}$ autoregressively - each output token becomes the input for the next step.

$$h_{\text{dec}}^{(t)} = \text{RNN}_\text{dec}(h_{\text{dec}}^{(t-1)}, y^{(t-1)}, c), \qquad y^{(t)} \sim \text{softmax}(W_{hy} h_{\text{dec}}^{(t)})$$

The bottleneck is the context vector $c$: a fixed-size vector encoding an entire variable-length sequence. For long sequences, this bottleneck causes information loss - the final hidden state cannot reliably encode all relevant information from the beginning of the source.



## 6.9 Attention: Breaking the Bottleneck

The **attention mechanism** (Bahdanau et al., 2015) resolves the context bottleneck by allowing the decoder to "look back" at all encoder hidden states, weighting them by relevance to the current decoding step.

At each decoder step $t$, the attention mechanism computes:

**Attention scores**: $e_i^{(t)} = \text{score}(h_{\text{dec}}^{(t)}, h_{\text{enc}}^{(i)})$ for each encoder position $i$.

**Attention weights**: $\alpha_i^{(t)} = \frac{\exp(e_i^{(t)})}{\sum_{j} \exp(e_j^{(t)})}$

**Context vector**: $c^{(t)} = \sum_i \alpha_i^{(t)} h_{\text{enc}}^{(i)}$

The context vector is now dynamic - it is a different weighted average of encoder states at each decoder step. When generating the word "apple" in French, the attention can focus on the encoder state corresponding to the English "apple", regardless of its position in the source sentence. The bottleneck is eliminated; the gradient path from decoder to encoder is now just one step, resolving the long-range dependency problem.

Three scoring functions became standard:

$$e_i^{(t)} = \begin{cases} v^T \tanh(W_1 h_{\text{dec}}^{(t)} + W_2 h_{\text{enc}}^{(i)}) & \text{(Additive / Bahdanau)} \\ h_{\text{dec}}^{(t)T} W h_{\text{enc}}^{(i)} & \text{(Multiplicative / Luong)} \\ h_{\text{dec}}^{(t)T} h_{\text{enc}}^{(i)} / \sqrt{d_k} & \text{(Scaled Dot-Product / Transformer)} \end{cases}$$

The scaled dot-product attention, used in Transformers, is the most computationally efficient - the scaling by $\sqrt{d_k}$ prevents dot products from growing large in high dimensions, where the gradient of softmax becomes small.

**Self-attention** is the final evolution: apply attention between all positions within the same sequence. Instead of an encoder attending to decoder, every position attends to every other position:

$$\text{SelfAttention}(X) = \text{softmax}\!\left(\frac{XW^Q(XW^K)^T}{\sqrt{d_k}}\right) XW^V$$

Self-attention connects any two positions in a sequence in a single step, regardless of their distance - the fundamental advantage that enabled Transformers to succeed where RNNs struggled. The "Attention Is All You Need" paper (Vaswani et al., 2017) showed that stacking self-attention with feedforward layers, without any recurrence, outperforms RNNs on translation benchmarks. This was the beginning of the Transformer era.



## 6.10 Computational Complexity and the Case for Recurrence

The transition from RNNs to Transformers came at a cost: computational complexity. For a sequence of length $T$ and hidden dimension $d$:

| Model | Training complexity | Inference per step |
|-------|--------------------|--------------------|
| Vanilla RNN | $\mathcal{O}(T \cdot d^2)$ (sequential) | $\mathcal{O}(d^2)$ |
| LSTM | $\mathcal{O}(T \cdot 4d^2)$ (sequential) | $\mathcal{O}(4d^2)$ |
| Transformer | $\mathcal{O}(T^2 \cdot d)$ (parallelizable) | $\mathcal{O}(T \cdot d)$ |

The Transformer's $T^2$ complexity means that doubling the sequence length quadruples the compute. For short sequences, this is manageable. For sequences of length $T = 10{,}000$ (a short book, a long audio file, a genomic sequence), the quadratic cost becomes prohibitive.

RNNs scale linearly in $T$ and process each step with constant memory - no growing KV cache. For applications requiring **streaming inference** (processing real-time data as it arrives), or **very long sequences** (up to millions of tokens), the recurrent architecture's linear scaling makes it the only practical choice.

This observation motivated the modern return to recurrent architectures, but with a critical upgrade: making them parallelizable during training.



## 6.11 State Space Models: Recurrence Made Parallel

**State Space Models (SSMs)** bridge the gap between RNNs (efficient inference) and Transformers (efficient training). They begin from the continuous-time perspective:

$$\frac{dh}{dt} = Ah(t) + Bx(t), \qquad y(t) = Ch(t) + Dx(t)$$

where $h(t) \in \mathbb{R}^N$ is the state, $x(t) \in \mathbb{R}$ the input, $y(t) \in \mathbb{R}$ the output, and $A, B, C, D$ are learned matrices. Discretizing with step size $\Delta$ (via zero-order hold):

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = Ch_t$$

$$\bar{A} = e^{\Delta A} \approx (I - \Delta A/2)^{-1}(I + \Delta A/2), \qquad \bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I)B \approx \Delta B$$

This looks exactly like a linear RNN. Its remarkable property: the linear recurrence $y_t = \sum_{k=0}^{t} \bar{C}\bar{A}^k \bar{B} \cdot x_{t-k}$ can be computed as a **convolution** $y = K * x$ where $K_k = \bar{C}\bar{A}^k\bar{B}$. Convolutions are parallelizable via FFT, enabling training in $\mathcal{O}(T \log T)$ rather than $\mathcal{O}(T)$ sequential steps.

The **S4** architecture (Gu et al., 2021) exploits specific matrix structure - HiPPO (High-Order Polynomial Projection Operators) initialization of $A$ - to make this computation efficient and stable. HiPPO matrices are designed to retain information about the history at all timescales equally, giving the SSM a kind of idealized long-term memory.

### Mamba: Selective State Spaces

The limitation of fixed SSMs is that the matrices $A, B, C$ are input-independent - they apply the same update rule regardless of what the current token is. This prevents the model from "selecting" which information to keep and which to discard based on content.

**Mamba** (Gu & Dao, 2023) makes $B$, $C$, and $\Delta$ functions of the input:

$$B_t = B(x_t), \qquad C_t = C(x_t), \qquad \Delta_t = \Delta(x_t)$$

This **selective** mechanism allows the model to decide, at each step, how much of the current input to incorporate into the state and how to read out the state. The selectivity is analogous to LSTM gating - it allows the model to filter irrelevant information and focus on task-relevant content.

The challenge: input-dependent matrices break the convolution trick. Mamba overcomes this with a parallel scan algorithm (**prefix sum**) that computes the full state sequence in $\mathcal{O}(T)$ parallel steps:

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t \implies h_t = \sum_{k=0}^t \left(\prod_{j=k+1}^t \bar{A}_j\right) \bar{B}_k x_k$$

The prefix scan computes all $h_t$ simultaneously using binary tree reduction, maintaining $\mathcal{O}(T)$ total work with $\mathcal{O}(\log T)$ parallelism.

Mamba achieves **linear scaling** in sequence length during both training and inference, with performance competitive with Transformers on language modeling benchmarks and superior on very long sequences. At sequence lengths beyond $10^4$, Mamba consistently outperforms Transformer-based models of equivalent parameter count.

<DIAGRAM: A comparison diagram showing four architectures side-by-side: Vanilla RNN, LSTM, Transformer, and Mamba. For each: (1) A schematic of the computation unit, (2) Training complexity, (3) Inference complexity, and (4) Maximum effective context length (empirical). Arrows show the historical development: RNN → LSTM (gating), LSTM → Transformer (attention replaces recurrence), Transformer → Mamba (selective SSM recovers efficiency).>



## 6.12 Generalization Theory for Sequential Models

For sequential models, generalization theory faces a new challenge: the Rademacher complexity of an RNN depends on the spectral properties of $W_{hh}$ applied $T$ times. Specifically:

$$\hat{R}_N(\mathcal{H}_{\text{RNN}}) \leq \frac{\|W_{hh}\|_2^T \cdot \|W_{xh}\|_F}{N^{1/2}} \cdot \sqrt{\ln d_h}$$

The factor $\|W_{hh}\|_2^T$ grows exponentially in $T$ unless $\|W_{hh}\|_2 \leq 1$. Constraining the spectral norm of the recurrent matrix to at most 1 (via spectral normalization or orthogonal initialization) keeps the complexity bound tight, explaining why such constraints improve generalization in practice.

**Weight sharing** across time steps is itself a powerful implicit regularization. An unshared $T$-layer network has $T$ times as many parameters; the shared RNN is forced to learn *universal rules* that apply at any position. This constraint directly reduces the effective hypothesis class size, tightening the PAC bound.

Dropout applied recurrently - the same mask used at all time steps (Gal & Ghahramani, 2016) - provides Bayesian approximate inference and maintains the weight-sharing regularization while adding variance reduction through ensemble effects.



## 6.13 Information Theory of Sequences: Compression as Intelligence

An RNN trained on language modeling is fundamentally a **compression engine**. The objective - maximizing log-likelihood - is equivalent to minimizing the description length of the sequence under the model. A model that achieves lower cross-entropy is a more efficient compressor.

**Perplexity** $= \exp(\text{avg. cross-entropy bits per token})$ is the standard evaluation metric. It represents the effective branching factor - at perplexity 50, the model is as confused as if it had to choose uniformly among 50 equally likely next tokens. English text has an empirical entropy rate of approximately 1-1.5 bits per character; a model that achieves this perplexity has essentially "understood" English structure.

**Mutual information** $I(h^{(t)}; x^{(\tau)})$ for $\tau \ll t$ measures how much information about the distant past is retained in the current hidden state. For vanilla RNNs, this mutual information decays exponentially with $t - \tau$. For LSTMs, the decay is much slower - close to constant for the information gated into the cell state. For Transformers, *all* pairs of positions have $\mathcal{O}(1)$ mutual information, bounded away from zero, because any position can attend directly to any other. This is the information-theoretic explanation for why Transformers outperform RNNs on tasks requiring long-range dependencies.



## Summary

Recurrent networks represent one of the deepest architectural ideas in deep learning: the explicit representation of memory as a learnable state that evolves through time. The path from vanilla RNNs through LSTMs to Transformers and state space models is the story of successive solutions to a single core problem - how to let gradient signals travel across time.

The key developments:

**Vanilla RNNs** (Elman, 1990) established the recurrence principle but were crippled by the vanishing gradient problem - an inherent tension between stable memory and learnable long-range dependencies.

**LSTMs** (Hochreiter & Schmidhuber, 1997) resolved this with the cell state and its gated additive updates, creating a constant-error carousel that lets gradients flow across hundreds of steps. The gating mechanism is both the LSTM's strength and its computational overhead.

**GRUs** (Cho et al., 2014) streamlined the LSTM into two gates and a single state, achieving comparable performance with greater computational efficiency.

**Attention and Transformers** (Bahdanau et al., 2015; Vaswani et al., 2017) replaced recurrence with direct position-to-position connections, eliminating sequential bottlenecks and enabling parallelization. The cost was quadratic complexity in sequence length.

**State space models** (Gu et al., 2021) and **Mamba** (Gu & Dao, 2023) returned to recurrent principles but with parallelizable training via convolution and prefix scans, achieving linear scaling while recovering strong performance on long-context tasks.

Chapter 7 steps back from temporal modeling to consider a different fundamental question: not how to predict sequences, but how to learn *representations* - compact, structured codes that capture the essential content of complex, high-dimensional data.

---
*Continue to **[Chapter 7: The Geometry of Representation - Autoencoders and Variational Autoencoders](/DeepLearning/07_Autoencoders_and_VAEs.md)***
