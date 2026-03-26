# Chapter 6: RNNs & LSTMs - Memory in Motion

<div style="text-align: center; margin: 20px 0;">
  <p style="font-size: 1.4em; margin-bottom: 8px;">
    <i>"Memory is the treasury and guardian of all things"</i>
  </p>
  <p style="font-size: 0.9em; color: #777;">
    Marcus Tullius Cicero
  </p>
</div>



## 6.1 The Aha! Moment: Sequence Requires Memory

Every architecture we have studied has one critical limitation: **memorylessness**. Feed a CNN the same image twice, and you get the same output both times - it has no recollection of the first time. For images, this is fine: a cat photograph does not change meaning based on what photograph came before it.

But consider language. The sentence "The bank was steep and muddy" requires remembering that we are talking about a riverbank, not a financial institution. The word "bank" alone is ambiguous; its meaning depends on everything that came before. More generally, sequential data has **temporal structure**: the meaning of element $t$ depends on elements $t-1, t-2, \ldots$, potentially going back hundreds of steps.

Four challenges distinguish sequential data from images:

**Temporal dependencies:** The current value depends on past values. Tomorrow's temperature depends on today's.

**Variable length:** Sequences do not have a fixed size. One sentence may be 5 words; another, 50.

**Causality:** In real-time processing, you cannot look into the future. Your model must maintain a running "summary" of the past.

**Long-range dependencies:** "The trophy would not fit in the suitcase because **it** was too big" - resolving "it" requires connecting a pronoun to a noun potentially many sentences earlier.

Standard feedforward networks fail all four: they process each input in isolation, require fixed-size input, and have no notion of "before" and "after". **Recurrent Neural Networks (RNNs)** solve this by introducing a **hidden state** - a fixed-size vector that is updated at every time step, carrying information from the past into the future.



## 6.2 The Vanilla RNN

### 6.2.1 The State Transition

A vanilla (Elman) RNN maintains a **hidden state** $\mathbf{h}^{(t)} \in \mathbb{R}^{d_h}$ - a vector of $d_h$ numbers that encodes the network's "memory" of the sequence up to time $t$. At each time step, it updates this state using the current input and the previous state:

$$\mathbf{h}^{(t)} = \sigma\!\left(W_{hh}\, \mathbf{h}^{(t-1)} + W_{xh}\, \mathbf{x}^{(t)} + \mathbf{b}_h\right)$$

Reading every symbol:
- $\mathbf{h}^{(t)}$: the hidden state at time step $t$ - a vector with $d_h$ entries. The superscript $(t)$ indexes the time step
- $\mathbf{h}^{(t-1)}$: the hidden state from the *previous* time step - the model's "memory" of what came before
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$: the **recurrent weight matrix** - controls how the previous state influences the current state. The subscript $hh$ reads "hidden-to-hidden". This square matrix transforms the previous hidden state
- $\mathbf{x}^{(t)} \in \mathbb{R}^{d_x}$: the current input at time $t$ (e.g., a word embedding)
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$: the **input weight matrix** - controls how the current input influences the state. The subscript $xh$ reads "input-to-hidden"
- $\mathbf{b}_h \in \mathbb{R}^{d_h}$: the bias vector
- $\sigma$: the activation function - typically $\tanh$, which squashes values to $(-1, 1)$ to prevent the state from growing unboundedly

The output at each step:

$$\mathbf{y}^{(t)} = W_{hy}\, \mathbf{h}^{(t)} + \mathbf{b}_y$$

- $W_{hy} \in \mathbb{R}^{d_y \times d_h}$: maps the hidden state to the output space
- $\mathbf{b}_y$: output bias

**The crucial property - parameter sharing across time:** The same matrices $W_{hh}$, $W_{xh}$, $W_{hy}$ are used at *every time step*. Whether processing the first word or the hundredth, the same "rules" apply. This allows the network to handle sequences of any length with a fixed number of parameters.

**Analogy:** The hidden state is a mental notepad. At each step: read the new word ($W_{xh}\mathbf{x}^{(t)}$), consult your notes ($W_{hh}\mathbf{h}^{(t-1)}$), update the notepad ($\mathbf{h}^{(t)}$). The notepad has a fixed size $d_h$ - you can remember $d_h$ "aspects" of the history simultaneously.

> TODO: <!-- DIAGRAM: [A horizontal chain of 5 identical RNN cells. Each cell: a box labeled "RNN Cell" with (1) an arrow entering from below labeled $\mathbf{x}^{(t)}$ (the input), (2) an arrow entering from the left labeled $\mathbf{h}^{(t-1)}$ (previous state), (3) an arrow exiting upward labeled $\mathbf{y}^{(t)}$ (output), and (4) an arrow exiting to the right labeled $\mathbf{h}^{(t)}$ (new state passed to next cell). A box on the far left labeled "Initial state $\mathbf{h}^{(0)} = \mathbf{0}$" starts the chain. Caption: "At each step, the RNN reads the current input and its own previous state, updates the state, and produces an output. The hidden state (the horizontal arrows) carries memory from past to future".] -->

### 6.2.2 Backpropagation Through Time (BPTT)

Training an RNN uses a variant of backpropagation called **Backpropagation Through Time (BPTT)**. The key insight: "unroll" the recurrence - treat each time step as a separate layer, all sharing the same weights.

For a sequence of $T$ steps and a loss at the final step $\mathcal{L}^{(T)}$ (or a sum of losses over all steps), the gradient of the loss with respect to the hidden state at time $t$ requires the chain rule across all subsequent steps:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(t)}} = \sum_{k=t}^{T} \frac{\partial \mathcal{L}^{(k)}}{\partial \mathbf{h}^{(k)}} \cdot \prod_{j=t+1}^{k} \frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}}$$

Reading the product term: $\frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}}$ is the Jacobian of the state update - for the vanilla RNN, this is $\text{diag}(\sigma'(\mathbf{z}^{(j)})) \cdot W_{hh}$, where $\mathbf{z}^{(j)} = W_{hh}\mathbf{h}^{(j-1)} + W_{xh}\mathbf{x}^{(j)} + \mathbf{b}_h$ is the pre-activation.

The problem: $\prod_{j=t+1}^{k}$ is a product of $k-t$ such matrices. For long sequences ($k-t$ large), this product is either vanishingly small or explosively large.



## 6.3 The Vanishing Gradient Problem in Time

### 6.3.1 Why It Happens

Recall from Chapter 2: products of many matrices with spectral radius (largest singular value) less than 1 shrink exponentially; greater than 1, they grow exponentially.

For the vanilla RNN's Jacobian $\text{diag}(\sigma'(\mathbf{z})) \cdot W_{hh}$:

**With $\tanh$ activation:** $\tanh'(z) = 1 - \tanh^2(z)$, which is at most 1 (when $z = 0$) and approaches 0 for large $|z|$. In practice, neurons often saturate (operate at large $|z|$), making the derivative close to 0.

**The consequence:** For a sequence of $T$ steps with $\tanh$ activations, the gradient at step $t$ involves a product of $T-t$ factors, each typically much less than 1. For $T - t = 50$: even if each factor is 0.9, the product is $0.9^{50} \approx 0.005$ - a 200-fold reduction. For 100 steps: $0.9^{100} \approx 0.000027$ - a 37,000-fold reduction.

The error signal is essentially zero by the time it reaches the early steps. **The model cannot learn that what happened 50 steps ago caused the error today.** Long-range dependencies - the most interesting and important patterns in language - are invisible.

### 6.3.2 Why This Is Fundamental, Not Accidental

Bengio et al. (1994) proved this is not a coincidence of specific activation functions or weight magnitudes. It is a fundamental mathematical tension:

- To reliably **store** information over long periods, the dynamics must be stable: perturbations to the state should not grow. This requires the Jacobian's spectral radius $< 1$ - **exactly the condition that causes vanishing gradients**.
- To **propagate gradients** over long periods, the Jacobian's spectral radius must be $\geq 1$ - **exactly the condition that causes instability (exploding gradients)**.

There is no setting of vanilla RNN parameters that can store information reliably over long sequences and also propagate gradients reliably. The architecture is fundamentally broken for long-range dependencies.



## 6.4 Long Short-Term Memory: Engineering Around the Problem

In 1997, Hochreiter and Schmidhuber published the solution. The key insight was architectural: rather than hoping gradient descent finds weights that propagate gradients well, *redesign the architecture* to provide a gradient-friendly pathway.

### 6.4.1 The Cell State: A Protected Memory Line

The LSTM introduces a second state variable: the **cell state** $\mathbf{c}^{(t)} \in \mathbb{R}^{d_h}$. Unlike the hidden state $\mathbf{h}^{(t)}$, the cell state evolves through primarily *additive* operations:

$$\mathbf{c}^{(t)} = \mathbf{f}^{(t)} \odot \mathbf{c}^{(t-1)} + \mathbf{i}^{(t)} \odot \tilde{\mathbf{c}}^{(t)}$$

Reading:
- $\mathbf{f}^{(t)}$: the **forget gate** - a vector of values between 0 and 1, same size as $\mathbf{c}^{(t-1)}$
- $\odot$: element-wise multiplication - multiply corresponding entries
- $\mathbf{f}^{(t)} \odot \mathbf{c}^{(t-1)}$: selectively "forget" parts of the old cell state. Where $f^{(t)}_j = 1$: keep memory unit $j$ fully. Where $f^{(t)}_j = 0$: erase memory unit $j$ completely
- $\mathbf{i}^{(t)}$: the **input gate** - also between 0 and 1; selectively admits new information
- $\tilde{\mathbf{c}}^{(t)}$: the **candidate** new content - what we *could* write into memory
- $\mathbf{i}^{(t)} \odot \tilde{\mathbf{c}}^{(t)}$: selectively write new information into the cell state

**Why is this gradient-friendly?** The gradient of the loss with respect to $\mathbf{c}^{(t-1)}$ is:

$$\frac{\partial \mathbf{c}^{(t)}}{\partial \mathbf{c}^{(t-1)}} = \text{diag}(\mathbf{f}^{(t)})$$

Reading: this Jacobian is a *diagonal matrix* of the forget gate values. If the forget gate learns to keep $f^{(t)}_j \approx 1$ for memory unit $j$, the gradient passes through with factor 1 - unchanged. Compare to the vanilla RNN, where the Jacobian involves the full weight matrix and activation derivatives multiplied together.

If the forget gate maintains $f^{(t)}_j \approx 1$ over many steps, the cell state gradient can travel back essentially unchanged across hundreds of steps. **The vanishing gradient problem is solved for the cell state.**

### 6.4.2 The Gates: Three Soft Switches

All three gates follow the same structure - they are sigmoid-activated linear functions of the current input $\mathbf{x}^{(t)}$ and previous hidden state $\mathbf{h}^{(t-1)}$:

**Forget gate:**
$$\mathbf{f}^{(t)} = \sigma\!\left(W_f \begin{bmatrix}\mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix} + \mathbf{b}_f\right)$$

Reading:
- $\begin{bmatrix}\mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix}$: the concatenation of previous hidden state and current input into one long vector
- $W_f$: weight matrix for the forget gate (shape: $d_h \times (d_h + d_x)$)
- $\mathbf{b}_f$: bias for the forget gate
- $\sigma(\cdot)$: sigmoid function - outputs values in $(0, 1)$, one per cell state dimension
- **Meaning:** "Based on what I just saw and what I remember, how much of each memory unit should I keep?" A value near 1 means "keep it"; near 0 means "erase it".

**Input gate:**
$$\mathbf{i}^{(t)} = \sigma\!\left(W_i \begin{bmatrix}\mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix} + \mathbf{b}_i\right)$$

- **Meaning:** "Of the candidate new information, how much should I actually write into memory?" A gate between "write everything" and "write nothing".

**Candidate cell content:**
$$\tilde{\mathbf{c}}^{(t)} = \tanh\!\left(W_c \begin{bmatrix}\mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix} + \mathbf{b}_c\right)$$

- $\tanh$: hyperbolic tangent - outputs values in $(-1, 1)$, the range of new information to potentially write
- **Meaning:** "Based on the current input and state, what new information could be written into memory?" This is the "proposal" that the input gate then filters.

**Cell state update:**
$$\mathbf{c}^{(t)} = \mathbf{f}^{(t)} \odot \mathbf{c}^{(t-1)} + \mathbf{i}^{(t)} \odot \tilde{\mathbf{c}}^{(t)}$$

- **Meaning:** Selectively forget old content + selectively write new content. The combination of these two terms allows the LSTM to simultaneously maintain long-term memories (keep $f_j \approx 1$ for important dimensions) and integrate new information (set $i_j$ high for relevant new content).

**Output gate:**
$$\mathbf{o}^{(t)} = \sigma\!\left(W_o \begin{bmatrix}\mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix} + \mathbf{b}_o\right)$$

$$\mathbf{h}^{(t)} = \mathbf{o}^{(t)} \odot \tanh\!\left(\mathbf{c}^{(t)}\right)$$

- **Output gate meaning:** "Of the current memory, which parts are relevant to my output right now?"
- $\tanh(\mathbf{c}^{(t)})$: squash the cell state to $(-1, 1)$ range
- $\mathbf{o}^{(t)} \odot \tanh(\mathbf{c}^{(t)})$: selectively read from memory to produce the visible hidden state $\mathbf{h}^{(t)}$

**The LSTM has four weight matrices** ($W_f$, $W_i$, $W_c$, $W_o$), each of shape $d_h \times (d_h + d_x)$ - four times the parameters of a vanilla RNN. The additional cost buys: gradient flow over hundreds of steps, selective memory control, and the ability to learn long-range dependencies.

> TODO: <!-- DIAGRAM: [LSTM cell diagram. Three horizontal arrows enter from the left: the cell state highway $\mathbf{c}^{(t-1)}$ (at the top), the hidden state $\mathbf{h}^{(t-1)}$ (at the bottom left), and a vertical input $\mathbf{x}^{(t)}$ from below. Three $\sigma$ boxes (forget, input, output gates) and one $\tanh$ box (candidate) are shown. The cell state highway flows through two operations: a pointwise multiply by $\mathbf{f}^{(t)}$ (forget), then a pointwise add of $\mathbf{i}^{(t)} \odot \tilde{\mathbf{c}}^{(t)}$ (input). The output gate reads from the updated cell state. Right-side outputs: $\mathbf{c}^{(t)}$ (cell state highway continues) and $\mathbf{h}^{(t)}$ (visible hidden state). Caption: "The cell state highway (top path) carries long-term memory with minimal transformation. Gates are soft switches controlling what to forget, write, and output. This architecture is designed so gradient can flow backward through the cell state highway with minimal attenuation".] -->



## 6.5 Gated Recurrent Units: Elegant Simplification

**GRU** (Cho et al., 2014) achieves similar gradient-flow properties with fewer parameters by merging the cell state and hidden state into one.

**Update gate:**
$$\mathbf{z}^{(t)} = \sigma\!\left(W_z \begin{bmatrix}\mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix} + \mathbf{b}_z\right)$$

**Reset gate:**
$$\mathbf{r}^{(t)} = \sigma\!\left(W_r \begin{bmatrix}\mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix} + \mathbf{b}_r\right)$$

**Candidate hidden state:**
$$\tilde{\mathbf{h}}^{(t)} = \tanh\!\left(W_h \begin{bmatrix}\mathbf{r}^{(t)} \odot \mathbf{h}^{(t-1)} \\ \mathbf{x}^{(t)}\end{bmatrix} + \mathbf{b}_h\right)$$

**Hidden state update:**
$$\mathbf{h}^{(t)} = (1 - \mathbf{z}^{(t)}) \odot \mathbf{h}^{(t-1)} + \mathbf{z}^{(t)} \odot \tilde{\mathbf{h}}^{(t)}$$

Reading the update equation:
- $(1 - \mathbf{z}^{(t)}) \odot \mathbf{h}^{(t-1)}$: keep $(1-z)$ fraction of the old state. When $z_j = 0$: keep the old state entirely; when $z_j = 1$: discard old state entirely
- $\mathbf{z}^{(t)} \odot \tilde{\mathbf{h}}^{(t)}$: take $z$ fraction of the new candidate state
- Together: the update gate interpolates between old state and new candidate - simultaneously performing the role of both the forget and input gates from the LSTM

**When $\mathbf{z}^{(t)} = 0$:** $\mathbf{h}^{(t)} = \mathbf{h}^{(t-1)}$ - the state is copied unchanged. Gradient flows through this step with factor 1 - no attenuation. This is the GRU's version of the LSTM's gradient highway.

**GRU vs. LSTM:** GRU has 3 weight matrices vs. LSTM's 4 - ~25% fewer parameters. Empirically, they perform similarly on most tasks. GRU trains faster and may generalize better on smaller datasets. LSTM sometimes performs better on very long sequences (the dedicated cell state provides a cleaner gradient path). In practice: try GRU first for efficiency, switch to LSTM if gradient flow seems insufficient.



## 6.6 Architectural Extensions

### 6.6.1 Bidirectional RNNs

A standard RNN is causal: at time $t$, it only knows what came *before*. For offline tasks where the entire sequence is available (translation, sentiment analysis, protein structure prediction), information from the *future* is also available and helpful.

**Bidirectional RNNs** use two RNN layers:

$$\overrightarrow{\mathbf{h}}^{(t)} = \text{RNN}_{\text{forward}}\!\left(\mathbf{x}^{(t)}, \overrightarrow{\mathbf{h}}^{(t-1)}\right) \quad \text{(processes left to right)}$$

$$\overleftarrow{\mathbf{h}}^{(t)} = \text{RNN}_{\text{backward}}\!\left(\mathbf{x}^{(t)}, \overleftarrow{\mathbf{h}}^{(t+1)}\right) \quad \text{(processes right to left)}$$

$$\mathbf{h}^{(t)} = \left[\overrightarrow{\mathbf{h}}^{(t)};\, \overleftarrow{\mathbf{h}}^{(t)}\right] \quad \text{(concatenate)}$$

The concatenation $[\cdot; \cdot]$ stacks the two vectors end-to-end, doubling the hidden size. At each position, the model simultaneously has access to everything before (from the forward pass) and everything after (from the backward pass).

**Application:** Resolving the "bank" ambiguity requires knowing what comes after "bank" - "was steep and muddy" (river) or "account was empty" (financial). Bidirectional models naturally capture this.

**Limitation:** Cannot be used for online prediction (where future is unknown) or autoregressive generation (where we predict the next token before it exists).

### 6.6.2 Multi-Layer (Deep) RNNs

Just as stacking CNN layers builds hierarchical visual features, stacking RNN layers builds hierarchical temporal features. In a **deep RNN** with $L$ layers:

$$\mathbf{h}_\ell^{(t)} = f_\ell\!\left(\mathbf{h}_\ell^{(t-1)},\, \mathbf{h}_{\ell-1}^{(t)}\right)$$

Reading: layer $\ell$'s hidden state at time $t$ depends on its own previous state (temporal memory) and the output of the layer below (the input from the lower layer). Layer 1 receives the actual input $\mathbf{x}^{(t)}$; each subsequent layer receives the hidden state of the layer below.

**What each layer learns:** Lower layers capture fine-grained temporal patterns (character sequences, local syntax). Higher layers capture abstract semantic content (topics, discourse structure). Neural machine translation models from 2016–2017 used 4–8-layer stacked LSTMs - before Transformers made them obsolete.

### 6.6.3 From RNNs to Attention: The Bridge

Even with LSTMs, a bottleneck problem remained for sequence-to-sequence tasks (like translation). The **encoder** processes the source sentence and compresses its meaning into a single hidden vector $\mathbf{h}^{(T)}$. The **decoder** must reconstruct the translation from this single vector - a challenging compression for long sentences.

**Attention mechanisms** (Bahdanau et al., 2015) solved this by allowing the decoder to "look back" at all encoder hidden states:

$$\mathbf{context}^{(s)} = \sum_{t=1}^{T} \alpha^{(s,t)}\, \mathbf{h}^{(t)}_{\text{enc}}$$

where:

$$\alpha^{(s,t)} = \frac{\exp(e(d^{(s)}, \mathbf{h}^{(t)}_{\text{enc}}))}{\sum_{t'=1}^{T} \exp(e(d^{(s)}, \mathbf{h}^{(t')}_{\text{enc}}))}$$

Reading:
- $\alpha^{(s,t)}$: the attention weight at decoder step $s$ for encoder step $t$ - how much the decoder "attends to" encoder position $t$ while generating output word $s$
- $e(d^{(s)}, \mathbf{h}^{(t)}_{\text{enc}})$: a scalar "alignment score" between the current decoder state $d^{(s)}$ and encoder state $\mathbf{h}^{(t)}_{\text{enc}}$ - typically a dot product or small neural network
- The $\exp(\cdot)$ and normalization: convert scores to probabilities (all $\alpha^{(s,t)} \geq 0$, sum to 1 over $t$) using the softmax formula
- $\mathbf{context}^{(s)}$: a weighted sum of all encoder states - a "dynamic context" that highlights the source positions most relevant to the current output position

This mechanism proved so powerful that in 2017, Vaswani et al. asked: what if we removed the recurrence entirely and used *only* attention? The result was the Transformer - the subject of later chapters.



## 6.7 Practical Considerations

### 6.7.1 Gradient Clipping (Revisited for RNNs)

Exploding gradients are especially common in RNNs because the same weight matrix $W_{hh}$ is multiplied many times. The standard remedy: clip the gradient norm before the parameter update.

$$\text{if } \|\mathbf{g}\|_2 > \tau: \quad \mathbf{g} \leftarrow \frac{\tau}{\|\mathbf{g}\|_2}\, \mathbf{g}$$

For RNNs, $\tau = 1.0$ or $\tau = 5.0$ is typical. This is not a substitute for a good architecture (LSTMs still vastly outperform clipped vanilla RNNs for long sequences) - it is a safety mechanism that prevents catastrophic parameter updates when they occasionally occur.

### 6.7.2 Truncated BPTT

For very long sequences (books, long audio), full BPTT is computationally infeasible: computing gradients back through 100,000 steps would require storing 100,000 copies of the hidden state.

**Truncated BPTT** processes the sequence in chunks of length $k$ (typically 20–100 steps):
- Forward pass: carry the hidden state forward across chunks (the model sees the full sequence)
- Backward pass: cut off after $k$ steps (gradients only flow back within each chunk)

Effect: the model can leverage context from the entire sequence (through the carried-forward hidden state) but can only learn dependencies up to $k$ steps back (through gradient flow). Choosing $k$ balances memory/computation against long-range learning capability.

### 6.7.3 The Remaining Role of RNNs

With Transformers dominating sequence modeling benchmarks, one might declare RNNs obsolete. This would be premature:

**Online/streaming tasks:** Processing a live audio stream or sensor feed requires generating outputs as each element arrives, without access to future elements. RNNs compute each step in $O(d_h^2)$ time regardless of total sequence length. Transformers require $O(T \cdot d)$ per step (caching all previous keys/values), growing with sequence length.

**Memory efficiency at inference:** Generating text token by token, a Transformer's memory grows as $O(T)$ (the KV cache). An LSTM's memory is $O(d_h)$ - constant regardless of sequence length. For very long generation contexts, LSTMs are dramatically more memory-efficient.

**Embedded systems:** Mobile devices and microcontrollers cannot run large Transformer models. LSTMs with small $d_h$ are deployable on hardware with kilobytes of RAM.


## Summary

Recurrent networks represent one of the deepest architectural ideas in deep learning: the explicit representation of memory as a learnable state that evolves through time. The path from vanilla RNNs through LSTMs to Transformers and state space models is the story of successive solutions to a single core problem - how to let gradient signals travel across time.

The key developments:

**Vanilla RNNs** (Elman, 1990) established the recurrence principle but were crippled by the vanishing gradient problem - an inherent tension between stable memory and learnable long-range dependencies.

**LSTMs** (Hochreiter & Schmidhuber, 1997) resolved this with the cell state and its gated additive updates, creating a constant-error carousel that lets gradients flow across hundreds of steps. The gating mechanism is both the LSTM's strength and its computational overhead.

**GRUs** (Cho et al., 2014) streamlined the LSTM into two gates and a single state, achieving comparable performance with greater computational efficiency.

**Attention and Transformers** (Bahdanau et al., 2015; Vaswani et al., 2017) replaced recurrence with direct position-to-position connections, eliminating sequential bottlenecks and enabling parallelization. The cost was quadratic complexity in sequence length.

**State space models** (Gu et al., 2021) and **Mamba** (Gu & Dao, 2023) returned to recurrent principles but with parallelizable training via convolution and prefix scans, achieving linear scaling while recovering strong performance on long-context tasks.

*Sequential data required memory (RNNs). Spatial data required local filters (CNNs). What about learning the structure of data itself - not just recognizing patterns, but generating new examples? That requires generative models: autoencoders and VAEs, the subject of Chapter 7.*


Chapter 7 steps back from temporal modeling to consider a different fundamental question: not how to predict sequences, but how to learn *representations* - compact, structured codes that capture the essential content of complex, high-dimensional data.

---
*Continue to **[Chapter 7: The Geometry of Representation - Autoencoders and Variational Autoencoders](/DeepLearning/07_Autoencoders_and_VAEs.md)***
