# Chapter 2: Backpropagation - The Art of Assigning Blame

> *"Backpropagation is the world's most successful algorithm for training neural networks - and one of the least understood, even by those who use it daily"*



## 2.1 The Problem of Attribution

Let us begin with the core challenge, stated in the most concrete possible way.

You have a neural network with five layers and, say, ten million parameters - ten million individual numbers called weights and biases. You show it a photo of a cat, and the network confidently says "dog". There is a **loss** - some number measuring how wrong that prediction was. Now you face a question that seems almost impossibly difficult:

*Which of the ten million numbers caused this error, and by how much should each one be adjusted to fix it?*

The naive solution: try each weight one by one. Add a tiny amount to weight number 1, see if the error goes up or down. Then reset it, try weight number 2. Repeat ten million times. This approach - called **numerical differentiation** - would require ten million separate forward passes per training step. At even a thousand training steps, that is ten billion forward passes. Training a modern network would take not hours or days, but geological epochs.

**Backpropagation** is the algorithm that computes the required adjustment for *every* weight simultaneously, in a single additional pass through the network. Not approximately - exactly. And the cost of this single backward pass is only about twice the cost of a single forward pass, regardless of how many parameters there are.

This is not a trick or an approximation. It is a consequence of a mathematical law - the **chain rule** of calculus - applied with great cleverness. Understanding why it works, not just how to run it, is the goal of this chapter.



## 2.2 The Intuition: A Factory Tracing a Defect

Before any mathematics, let us build the right mental picture.

Imagine a factory with five workstations in a line. Raw material enters at Station 1 and is processed at each station in turn, until a finished product exits Station 5. One day, the product comes out defective.

To fix it, the manager traces the defect *backward* through the factory: "This scratch on the surface came from Station 5's polishing wheel. The dent underneath it came from Station 3's press. The wrong material dimensions that caused the dent were introduced at Station 1". By following the chain of causality backward - station by station - the manager assigns precise blame and determines the exact adjustment for each machine's settings.

Backpropagation is this backward tracing, made exact and automated by calculus.

In our neural network:
- Each **layer** is a workstation
- The **weights** are the machine settings
- The **loss** is the measure of defectiveness
- **Backpropagation** traces which weights contributed to the error, and by exactly how much

The word "back" is literal: we start at the output (where the error is measured) and propagate the blame signal backwards, layer by layer, to the very first layer.



## 2.3 The Mathematical Foundation: Three Essential Ideas

Backpropagation rests on three mathematical pillars. We build each one from scratch before assembling them into the full algorithm.

### 2.3.1 Pillar One: The Derivative

The **derivative** of a function $f$ at a point $x$ answers: *If I nudge the input $x$ by a tiny amount, how much does the output $f(x)$ change?*

$$f'(x) = \lim_{\varepsilon \to 0} \frac{f(x + \varepsilon) - f(x)}{\varepsilon}$$

Symbol by symbol:
- $f'(x)$: the derivative of $f$ at the point $x$ (read: "f prime of x")
- $\lim_{\varepsilon \to 0}$: "as $\varepsilon$ gets infinitesimally small" - we look at the change caused by a vanishingly tiny nudge
- $f(x + \varepsilon) - f(x)$: the change in the output when we nudge the input by $\varepsilon$
- Divided by $\varepsilon$: the *ratio* of output change to input change - how much the output moves per unit of input movement

If $f'(x) = 5$ at some point: nudge the input by $0.001$, the output changes by approximately $5 \times 0.001 = 0.005$. If $f'(x) = -3$, the output *decreases* when you increase the input.

**Why this matters for learning:** If the loss $\mathcal{L}$ has derivative $+7$ with respect to weight $w$, increasing $w$ increases the loss. We should *decrease* $w$ to reduce the loss. The derivative is our compass: it tells us which direction to move each weight.

### 2.3.2 Pillar Two: The Gradient

Neural networks have millions of weights - not one. We cannot talk about "the derivative"; we need a derivative with respect to *each weight separately*. This is the **partial derivative**.

The partial derivative $\frac{\partial \mathcal{L}}{\partial w_j}$ asks: "If I nudge only weight $w_j$, holding all other weights fixed, how does the loss $\mathcal{L}$ change?" The curved $\partial$ symbol (read: "partial") signals that we are differentiating with respect to one variable while treating all others as constants.

The **gradient** $\nabla_\theta \mathcal{L}$ collects all partial derivatives into a single vector:

$$\nabla_\theta \mathcal{L} = \left[\frac{\partial \mathcal{L}}{\partial w_1},\; \frac{\partial \mathcal{L}}{\partial w_2},\; \ldots,\; \frac{\partial \mathcal{L}}{\partial w_p}\right]^T$$

Reading this:
- $\nabla$ (the nabla or "del" symbol): means "gradient of"
- $\theta$: all the parameters together (weights and biases) - the subscript says "gradient with respect to $\theta$"
- $\mathcal{L}$: the loss function
- The bracket contains $p$ partial derivatives, one for each of the $p$ parameters
- ${}^T$: transpose - we write it as a tall column vector (stacked vertically)

The gradient vector points in the direction of *steepest increase* of the loss. Moving in the *opposite* direction (subtracting the gradient) reduces the loss. The **gradient descent update** is:

$$w_j \leftarrow w_j - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_j}$$

Reading: "the new value of weight $w_j$ equals the old value minus a small step $\eta$ (eta, the learning rate) in the direction of the gradient". The $\leftarrow$ means "is updated to become". The learning rate $\eta$ is a small positive number (say $0.001$) controlling how large a step we take.

Backpropagation's entire job is to compute every entry of this gradient vector, efficiently.

### 2.3.3 Pillar Three: The Chain Rule

This is the crown jewel. A neural network is a composed function: apply Layer 1, then Layer 2 on that result, then Layer 3, and so on. The chain rule tells us how to differentiate composed functions.

Start simple. Two functions in sequence:

$$z = f(x), \qquad y = g(z) = g(f(x))$$

You nudge $x$ by a tiny amount $\Delta x$. Then $z$ changes by $f'(x) \cdot \Delta x$. Then $y$ changes by $g'(z) \cdot \Delta z = g'(z) \cdot f'(x) \cdot \Delta x$.

So the total effect of nudging $x$ on $y$ is:

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx} = g'(z) \cdot f'(x)$$

Reading: "how $y$ changes with $x$ equals how $y$ changes with $z$, times how $z$ changes with $x$". We multiply local sensitivities because changes compound through the chain.

For three functions $a = f(x)$, $b = g(a)$, $y = h(b)$:

$$\frac{dy}{dx} = \frac{dy}{db} \cdot \frac{db}{da} \cdot \frac{da}{dx}$$

Each fraction is a local sensitivity. The chain rule says: multiply them all together.

**For a 4-layer network:** The loss $\mathcal{L}$ depends on Layer 4's output, which depends on Layer 3's output, which depends on Layer 2's, which depends on Layer 1's, which depends on the weights. The gradient with respect to a weight in Layer 1 is a product of four local sensitivities - one for each layer in between. Backpropagation computes this product efficiently by accumulating from the output backward.

> TODO: <!-- DIAGRAM: [Four boxes: "Layer 1" → "Layer 2" → "Layer 3" → "Loss $\mathcal{L}$". Blue arrows flow right (forward: data). Red arrows flow left (backward: error signal). Above each red arrow: the local derivative at that step. Below the diagram: "Gradient at Layer 1 = (local sensitivity at Layer 1) × (local sensitivity at Layer 2) × (local sensitivity at Layer 3) × (sensitivity of Loss)". Caption: "The chain rule converts the daunting question 'how does this first-layer weight affect the final loss?' into a product of manageable local sensitivities, each computable from quantities already stored during the forward pass".] -->



## 2.4 Multi-Dimensional Layers: Entering Matrix Territory

Real layers are not scalar-to-scalar functions. They transform *vectors* to *vectors*: a layer might take 256 numbers as input and produce 128 numbers as output. When a function maps vectors to vectors, its "derivative" becomes a matrix.

Let $f: \mathbb{R}^n \to \mathbb{R}^m$ take an $n$-dimensional input vector and produce an $m$-dimensional output vector. The **Jacobian matrix** $J_f \in \mathbb{R}^{m \times n}$ collects all partial derivatives:

$$(J_f)_{ij} = \frac{\partial f_i}{\partial x_j}$$

Reading element by element:
- $(J_f)_{ij}$: entry at row $i$, column $j$ of the Jacobian
- $f_i$: the $i$-th output of the function (e.g., the $i$-th neuron's value)
- $x_j$: the $j$-th input to the function (e.g., the $j$-th feature)
- So: "how much does output $i$ change when we nudge input $j$?"

For a layer with 128 inputs and 64 outputs, the Jacobian is a $64 \times 128$ matrix with $8{,}192$ entries - one partial derivative per input-output pair.

**The multi-dimensional chain rule:** For composed functions $h = g \circ f$ (apply $f$ then $g$):

$$J_h = J_g \cdot J_f$$

The Jacobian of the composition is the **matrix product** of the Jacobians. Note that this is regular matrix multiplication - not element-wise. For a network with $L$ layers:

$$J_{\text{network}} = J_{f_L} \cdot J_{f_{L-1}} \cdots J_{f_2} \cdot J_{f_1}$$

Backpropagation computes this product right-to-left - from the loss backward through each layer - accumulating the result progressively rather than forming and multiplying huge matrices all at once.

> **Why right-to-left?** We have one loss (scalar output) and millions of parameters (inputs). Computing from one output backward to many inputs - "reverse mode" differentiation - computes all gradients in one pass. Computing from each input forward to the output - "forward mode" - would require one pass per parameter. Reverse mode is the winner by a factor of millions when parameters vastly outnumber outputs (always true in neural networks).



## 2.5 The Forward Pass: Setting the Stage

Define the network formally. A feedforward neural network with $L$ layers computes, for each layer $\ell = 1, 2, \ldots, L$:

$$\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$

Every symbol decoded:
- $\mathbf{z}^{(\ell)}$: the **pre-activation** vector at layer $\ell$ - the raw scores before nonlinearity. Bold because it is a vector with one entry per neuron in layer $\ell$
- The superscript ${}^{(\ell)}$ in parentheses: the layer number. Not an exponent - a label
- $W^{(\ell)}$: the **weight matrix** of layer $\ell$, shape $n_\ell \times n_{\ell-1}$ - $n_\ell$ output neurons (rows) times $n_{\ell-1}$ input neurons (columns)
- $\mathbf{a}^{(\ell-1)}$: the **activation vector** from the *previous* layer - the output of layer $\ell-1$, serving as input to layer $\ell$. For $\ell=1$: $\mathbf{a}^{(0)} = \mathbf{x}$ (the raw data)
- $\mathbf{b}^{(\ell)}$: the **bias vector** - a vector with $n_\ell$ entries, one per output neuron. It shifts each neuron's output independently of the input
- $W^{(\ell)} \mathbf{a}^{(\ell-1)}$: matrix-vector multiplication. Each row of $W^{(\ell)}$ is a weight vector for one output neuron; we dot it with $\mathbf{a}^{(\ell-1)}$ and add the bias

Then we apply the activation function:

$$\mathbf{a}^{(\ell)} = \sigma^{(\ell)}\!\left(\mathbf{z}^{(\ell)}\right)$$

- $\mathbf{a}^{(\ell)}$: the **activation vector** - the layer's output, after nonlinearity
- $\sigma^{(\ell)}$: the activation function (ReLU, sigmoid, etc.) - applied **element-wise**: each entry of $\mathbf{z}^{(\ell)}$ is passed through $\sigma$ independently
- Applied to $\mathbf{z}^{(\ell)}$: each pre-activation becomes an activation

After $L$ such steps: $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$ - the network's prediction.

**The memory requirement.** During the forward pass, every $\mathbf{z}^{(\ell)}$ and $\mathbf{a}^{(\ell)}$ must be stored in memory. Why? The backward pass needs them to compute the activation function derivatives $\sigma'^{(\ell)}(\mathbf{z}^{(\ell)})$. This storage cost - proportional to batch size × total network width × depth - is the "memory tax" of backpropagation. Training always requires far more GPU memory than inference.



## 2.6 The Error Signal: The Core Quantity of the Backward Pass

The backward pass introduces a key quantity: the **error signal** at layer $\ell$.

$$\boldsymbol{\delta}^{(\ell)} := \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$$

Reading this carefully:
- $\boldsymbol{\delta}^{(\ell)}$: bold (it is a vector), Greek delta (conventional for "error signal"), layer index $\ell$
- $:=$: "is defined as" - this is a definition
- $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$: the vector of partial derivatives of the loss with respect to each entry of the pre-activation vector at layer $\ell$. Entry $i$ is $\frac{\partial \mathcal{L}}{\partial z^{(\ell)}_i}$: how does the loss change if we directly tweak the $i$-th pre-activation?

The error signal $\boldsymbol{\delta}^{(\ell)}$ answers: "If we could directly control the raw scores at layer $\ell$, how should we adjust them to reduce the loss?"

**Once we have $\boldsymbol{\delta}^{(\ell)}$, weight and bias gradients follow immediately:**

$$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} \cdot \left(\mathbf{a}^{(\ell-1)}\right)^T$$

Reading this outer product:
- $\boldsymbol{\delta}^{(\ell)}$: a column vector of shape $n_\ell \times 1$
- $\left(\mathbf{a}^{(\ell-1)}\right)^T$: a row vector of shape $1 \times n_{\ell-1}$ (the previous layer's activations, transposed)
- Their product: a matrix of shape $n_\ell \times n_{\ell-1}$ - exactly the shape of $W^{(\ell)}$

The entry at row $i$, column $j$: $\delta^{(\ell)}_i \cdot a^{(\ell-1)}_j$ - "the error at output neuron $i$ times the activation of input neuron $j$". Weight $W^{(\ell)}_{ij}$ connects input $j$ to output $i$; its gradient is proportional to both how "wrong" the output was and how strongly the input was firing. Large error × large activation = large gradient update. This is exactly the right thing.

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}$$

The bias gradient is simply the error signal itself - since the bias adds directly to $\mathbf{z}^{(\ell)}$ without multiplying any input, its gradient equals the error signal with no additional factor.



## 2.7 The Backward Recursion: Propagating Error Signals Layer by Layer

We need $\boldsymbol{\delta}^{(\ell)}$ for every layer $\ell$, starting from the output and going backward.

### 2.7.1 Starting at the Output Layer

For the pairing of softmax with categorical cross-entropy (multi-class) or sigmoid with binary cross-entropy (binary), the output error signal simplifies to:

$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$

Symbol by symbol:
- $\hat{\mathbf{y}}$: the network's prediction (e.g., probability vector from softmax)
- $\mathbf{y}$: the true label (e.g., one-hot vector: $[0, 1, 0, 0, \ldots]$ for class 2)
- $\hat{\mathbf{y}} - \mathbf{y}$: prediction minus truth - a vector where each entry is "how much too high (or low) the predicted probability for that class is"

**Example:** True class is cat (class index 1). True label: $\mathbf{y} = [0, 1, 0]$. Network prediction: $\hat{\mathbf{y}} = [0.1, 0.7, 0.2]$. Error signal: $\boldsymbol{\delta}^{(L)} = [0.1 - 0, 0.7 - 1, 0.2 - 0] = [0.1, -0.3, 0.2]$.

Interpretation: "Push the dog score down by $0.1$, push the cat score up by $0.3$ (since it is $0.3$ too low), push the bird score down by $0.2$".

The magnitude is proportional to the error - if the network predicts $\hat{y}_{\text{cat}} = 0.01$ (very wrong), the error signal entry is $0.01 - 1 = -0.99$ (very large correction). If it predicts $\hat{y}_{\text{cat}} = 0.95$ (nearly right), the entry is $-0.05$ (small correction). The network works harder when it is more wrong.

> **Why this simplification happens:** The sigmoid/softmax activation and the cross-entropy loss are "conjugate pairs" - their derivatives cancel beautifully, leaving just prediction minus truth. Using mismatched pairs (e.g., sigmoid with MSE) would not give this simplification, and would produce gradients that vanish when the prediction is saturated.

### 2.7.2 Hidden Layers: The Core Recursion

For any hidden layer $\ell < L$:

$$\boxed{\boldsymbol{\delta}^{(\ell)} = \underbrace{\left(W^{(\ell+1)}\right)^T \boldsymbol{\delta}^{(\ell+1)}}_{\text{Part A: transport error backward}} \;\odot\; \underbrace{\sigma'^{(\ell)}\!\left(\mathbf{z}^{(\ell)}\right)}_{\text{Part B: filter by activation derivative}}}$$

We have boxed this because it is the single most important formula in all of deep learning. Let us understand every piece.

**Part A: $(W^{(\ell+1)})^T \boldsymbol{\delta}^{(\ell+1)}$**

- $W^{(\ell+1)}$: the weight matrix of the *next* layer (layer $\ell + 1$), shape $n_{\ell+1} \times n_\ell$
- $(W^{(\ell+1)})^T$: the **transpose** of that matrix, shape $n_\ell \times n_{\ell+1}$. Transposing swaps rows and columns
- $\boldsymbol{\delta}^{(\ell+1)}$: the error signal we already computed for layer $\ell+1$, shape $n_{\ell+1} \times 1$
- Product shape: $(n_\ell \times n_{\ell+1}) \times (n_{\ell+1} \times 1) = n_\ell \times 1$ - one error value per neuron in layer $\ell$

**What does Part A do?** In the forward pass, $W^{(\ell+1)}$ carries activations from layer $\ell$ to layer $\ell+1$: each neuron in layer $\ell+1$ is a weighted sum of neurons in layer $\ell$. In the backward pass, $(W^{(\ell+1)})^T$ carries error signals in the *opposite direction*: each neuron in layer $\ell$ receives a weighted sum of error signals from neurons in layer $\ell+1$. The same weights used to build the prediction are used in reverse to distribute the blame.

Specifically, if neuron $j$ in layer $\ell$ connects strongly (large weight) to neuron $i$ in layer $\ell+1$, and neuron $i$ has a large error signal, then neuron $j$ receives a large share of the blame - as it should, since it contributed strongly to the erroneous output.

**Part B: $\sigma'^{(\ell)}(\mathbf{z}^{(\ell)})$**

- $\sigma'^{(\ell)}$: the **derivative** of the activation function used at layer $\ell$
- $\mathbf{z}^{(\ell)}$: the pre-activations stored during the forward pass at layer $\ell$
- $\sigma'^{(\ell)}(\mathbf{z}^{(\ell)})$: a vector where entry $i$ is the derivative of the activation at neuron $i$'s operating point

For **ReLU** ($\sigma(z) = \max(0, z)$): $\sigma'(z) = 1$ if $z > 0$, else $0$. So Part B is a vector of ones and zeros - a mask. Neurons that were active during the forward pass (positive pre-activation) let the error signal through unchanged. Neurons that were off (zero output) block the error signal entirely. This makes sense: an inactive neuron did not influence the output, so it bears no responsibility.

For **sigmoid** ($\sigma(z) = 1/(1+e^{-z})$): $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$. The derivative is always between 0 and 0.25 - and nearly 0 at the extremes. So Part B attenuates the error signal, sometimes severely. This is the **vanishing gradient problem** in action: sigmoid gates barely let error signals through, especially in saturated neurons.

**Part A $\odot$ Part B:**

The $\odot$ symbol is the **element-wise (Hadamard) product** - multiply the corresponding entries of two vectors. Entry $j$ of $\boldsymbol{\delta}^{(\ell)}$ is: (weighted sum of errors from layer $\ell+1$ coming through weight column $j$) times (how open the activation gate is at neuron $j$).

> TODO: <!-- DIAGRAM: [Six-layer network. Forward pass (blue arrows, left to right): activations flow through weight matrices and activation functions. Backward pass (red arrows, right to left): starting from δ^(L) = ŷ - y at the rightmost layer. At each step going left: (1) multiply by the transpose of the weight matrix (scatter the error backward through connections), (2) element-wise multiply by the activation derivative (filter by how "open" each neuron was). At each layer, branch downward: "gradient for W^(ℓ) = δ^(ℓ) · (a^(ℓ-1))^T". Caption: "Part A (the transposed weight matrix) distributes blame backward through the connections. Part B (the activation derivative) decides how much of the blame each neuron passes on, based on how active it was".] -->



## 2.8 A Complete Worked Example: Two Layers, One Neuron Each

To make the abstraction concrete, let us trace backpropagation completely through a tiny network: two layers, one neuron each, sigmoid activations, binary cross-entropy loss.

**Setup:** Input $x = 2.0$. True label $y = 1$. Weights: $w^{(1)} = 0.5$, $w^{(2)} = -1.0$. Biases: $b^{(1)} = 0$, $b^{(2)} = 0$.

**Forward pass:**

Layer 1:
$$z^{(1)} = w^{(1)} x + b^{(1)} = 0.5 \times 2.0 + 0 = 1.0$$
$$a^{(1)} = \sigma(z^{(1)}) = \sigma(1.0) = \frac{1}{1+e^{-1}} \approx 0.731$$

Layer 2 (output):
$$z^{(2)} = w^{(2)} a^{(1)} + b^{(2)} = -1.0 \times 0.731 + 0 = -0.731$$
$$\hat{y} = \sigma(z^{(2)}) = \sigma(-0.731) = \frac{1}{1+e^{0.731}} \approx 0.324$$

Loss (binary cross-entropy):
$$\mathcal{L} = -y \log\hat{y} - (1-y)\log(1-\hat{y}) = -1 \times \log(0.324) - 0 \approx 1.126$$

The network predicted $0.324$ probability for the positive class, but the truth is $1$ (it *is* positive). The loss is $1.126$ - not great.

**Backward pass:**

Output error signal (layer 2):
$$\delta^{(2)} = \hat{y} - y = 0.324 - 1 = -0.676$$

Reading: the prediction was $0.676$ too low. The network needs to push the output up.

Gradient for $w^{(2)}$:
$$\frac{\partial \mathcal{L}}{\partial w^{(2)}} = \delta^{(2)} \cdot a^{(1)} = -0.676 \times 0.731 = -0.494$$

Reading: the gradient is $-0.494$. The weight $w^{(2)}$ should be *increased* (gradient is negative, so the update $w^{(2)} \leftarrow w^{(2)} - \eta \cdot (-0.494) = w^{(2)} + 0.494\eta$ moves it upward). This makes sense: $w^{(2)}$ is currently $-1$, which multiplies the positive activation to produce a negative logit, leading to a low prediction. Increasing $w^{(2)}$ would reduce the negativity.

Error signal at layer 1 (applying the backward recursion):

First, the derivative of sigmoid at $z^{(1)} = 1.0$:
$$\sigma'(z^{(1)}) = \sigma(1.0)(1 - \sigma(1.0)) = 0.731 \times (1 - 0.731) = 0.731 \times 0.269 \approx 0.197$$

Now the backward recursion. Here $W^{(2)} = w^{(2)} = -1.0$ (it is a scalar in this 1-neuron case):

$$\delta^{(1)} = (w^{(2)})^T \cdot \delta^{(2)} \cdot \sigma'(z^{(1)}) = (-1.0) \times (-0.676) \times 0.197 \approx 0.133$$

Reading step by step:
- $(-1.0) \times (-0.676) = 0.676$: transpose of $w^{(2)}$ (still $-1.0$ for a scalar) times the output error signal. The negative weight "flips" the error signal: the output needed to go up, and $w^{(2)}$ is negative, so the input $a^{(1)}$ needed to go *down* to help
- $0.676 \times 0.197 = 0.133$: filtered by the sigmoid derivative at layer 1. The sigmoid gate was $\approx 20\%$ open, attenuating the signal

Gradient for $w^{(1)}$:
$$\frac{\partial \mathcal{L}}{\partial w^{(1)}} = \delta^{(1)} \cdot x = 0.133 \times 2.0 = 0.266$$

With learning rate $\eta = 0.1$, the update for $w^{(1)}$:
$$w^{(1)} \leftarrow 0.5 - 0.1 \times 0.266 = 0.5 - 0.027 = 0.473$$

The first-layer weight is nudged slightly downward - exactly as the gradient computed.

This is backpropagation: a simple chain of multiplications, working backward. Every quantity in the backward pass was already computed in the forward pass and stored.



## 2.9 The Full Algorithm: Mini-Batch Version

In practice, we process a **mini-batch** of $B$ examples simultaneously. The only change is that $\mathbf{x}$, $\mathbf{z}^{(\ell)}$, $\mathbf{a}^{(\ell)}$ are now matrices with $B$ columns (one per example), and gradients are averaged over the batch.

**Phase 1 - Forward Pass:** *(store everything)*

Set $A^{(0)} = X$ where $X \in \mathbb{R}^{n_0 \times B}$ (input matrix, $B$ examples).

For $\ell = 1$ to $L$:
$$Z^{(\ell)} = W^{(\ell)} A^{(\ell-1)} + \mathbf{b}^{(\ell)} \quad \text{(shape: } n_\ell \times B\text{)}$$
$$A^{(\ell)} = \sigma^{(\ell)}(Z^{(\ell)}) \quad \text{(applied element-wise)}$$
Store $Z^{(\ell)}$ and $A^{(\ell)}$ for every $\ell$.

Compute loss: $\mathcal{L} = \text{loss}(A^{(L)}, Y)$, where $Y \in \mathbb{R}^{n_L \times B}$ are the true labels.

**Phase 2 - Backward Pass:**

Compute output error signal:
$$\Delta^{(L)} = \frac{1}{B}\left(A^{(L)} - Y\right) \quad \text{(for softmax/sigmoid + cross-entropy)}$$

For $\ell = L-1, L-2, \ldots, 1$:
$$\Delta^{(\ell)} = \left[\left(W^{(\ell+1)}\right)^T \Delta^{(\ell+1)}\right] \odot \sigma'^{(\ell)}\!\left(Z^{(\ell)}\right)$$

Compute weight and bias gradients for every layer $\ell$:
$$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \Delta^{(\ell)} \left(A^{(\ell-1)}\right)^T \quad \text{(shape: } n_\ell \times n_{\ell-1}\text{, same as } W^{(\ell)}\text{)}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \Delta^{(\ell)} \cdot \mathbf{1}_B \quad \text{(sum across the batch, then divide by }B\text{)}$$

**Phase 3 - Update:**

For every $\ell$:
$$W^{(\ell)} \leftarrow W^{(\ell)} - \eta\, \frac{\partial \mathcal{L}}{\partial W^{(\ell)}}$$
$$\mathbf{b}^{(\ell)} \leftarrow \mathbf{b}^{(\ell)} - \eta\, \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}}$$

Repeat Phases 1–3 for each mini-batch.



## 2.10 Computational Cost: The 2F Rule

**How expensive is this?** Let $F$ be the total number of multiply-add operations in one forward pass. For a fully connected layer transforming $n_{\ell-1}$ inputs to $n_\ell$ outputs: approximately $2 n_\ell n_{\ell-1}$ operations. Summing over $L$ layers gives $F$.

For the backward pass:
- Computing $(W^{(\ell)})^T \boldsymbol{\delta}^{(\ell)}$: a matrix-vector multiply with dimensions swapped. Cost: $2 n_{\ell-1} n_\ell$ - exactly the same as the forward layer
- Computing $\Delta^{(\ell)} (A^{(\ell-1)})^T$: cost $2 n_\ell n_{\ell-1}$ - again the same

So each layer's backward computation costs the same as its forward computation. Total backward cost: approximately $2F$. Total training step cost: forward ($F$) + backward ($2F$) = $3F$.

**The key point:** Whether you have 1 million or 100 billion parameters, the backward pass costs roughly $2\times$ the forward pass - a fixed constant multiplier, independent of parameter count. This is what makes the algorithm tractable: we do not pay a cost proportional to the number of parameters we are differentiating with respect to.



## 2.11 Vanishing and Exploding Gradients

Recall the backward recursion applied $L-1$ times from layer $L$ to layer $1$:

$$\boldsymbol{\delta}^{(1)} = \left[\prod_{\ell=2}^{L} \text{diag}(\sigma'^{(\ell-1)}(\mathbf{z}^{(\ell-1)})) \cdot (W^{(\ell)})^T \right] \boldsymbol{\delta}^{(L)}$$

The $\prod$ symbol here means a product of matrices - multiply all the factors as $\ell$ goes from $2$ to $L$. Each factor is a matrix $\text{diag}(\sigma')$ (a diagonal matrix of activation derivatives) times $(W^{(\ell)})^T$ (the transposed weight matrix of that layer).

**The norm of a product of matrices:** Let $\lambda$ denote (loosely) the factor by which each matrix scales the typical vector that passes through it. Then the product of $L-1$ such matrices scales vectors by approximately $\lambda^{L-1}$. This scales exponentially in the depth $L$.

- If $\lambda < 1$: $\lambda^{L-1}$ shrinks exponentially. For $\lambda = 0.9$ and $L = 100$: the error signal at layer 1 is $0.9^{99} \approx 0.000037$ times the error signal at layer 100 - essentially zero. **Vanishing gradients**: early layers learn nothing.
- If $\lambda > 1$: $\lambda^{L-1}$ grows exponentially. For $\lambda = 1.1$ and $L = 100$: the error signal is $1.1^{99} \approx 12{,}527$ times the original - numerical overflow. **Exploding gradients**: training crashes.
- If $\lambda = 1$: the error signal is preserved through the network. The signal flows stably.

### 2.11.1 Sigmoid Saturates and Kills Gradients

For sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z))$.

When $z = 5$ (large and positive): $\sigma(5) \approx 0.993$, so $\sigma'(5) = 0.993 \times 0.007 = 0.007$. Only 0.7% of the error passes through.
When $z = -5$ (large and negative): $\sigma'(-5) \approx 0.007$. Same.
When $z = 0$ (center): $\sigma'(0) = 0.5 \times 0.5 = 0.25$. The maximum possible - still only 25%.

So the activation derivative factor is always at most $0.25$ per sigmoid layer. For 10 sigmoid layers: $(0.25)^{10} = 9.5 \times 10^{-7}$. The gradient at layer 1 is one million times smaller than at layer 10. The first layers learn at a rate of *less than one-millionth* the last layers. In practice, they do not learn at all.

### 2.11.2 ReLU: A Cleaner Story

For ReLU: $\sigma'(z) = 1$ if $z > 0$, else $0$.

The activation derivative is exactly 1 for all active neurons. The error signal passes through these neurons unchanged - no exponential decay from the activation derivatives. The only source of potential decay or growth is the weight matrix $W^{(\ell)}$.

This is why ReLU was the single most impactful activation function change in the history of deep learning: it eliminated the activation-derivative factor from the vanishing gradient problem, enabling training of much deeper networks.

The remaining risk is from the weight matrices: if $\|W^{(\ell)}\| \ll 1$, gradients still vanish; if $\|W^{(\ell)}\| \gg 1$, they explode. This is controlled through proper weight initialization (He initialization for ReLU) and normalization layers (BatchNorm, LayerNorm).

### 2.11.3 Residual Connections: Gradient Highways

Residual blocks add a skip connection: instead of computing only $\mathcal{F}(\mathbf{x})$, compute $\mathcal{F}(\mathbf{x}) + \mathbf{x}$. The gradient through this block is:

$$\frac{\partial (\mathcal{F}(\mathbf{x}) + \mathbf{x})}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} + \frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + I$$

- $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$: the Jacobian of the residual branch - this may vanish
- $I$: the **identity matrix** - it is always there, regardless of what $\mathcal{F}$ does
- Sum: even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}} \to 0$ completely, the gradient through the block is still $I$ - the identity

This means error signals always have an "express lane" through the skip connections, bypassing the potentially vanishing residual branch. Multiplied across 100 such blocks, the identity paths compose to give a direct gradient connection from the output to any early layer, with magnitude 1. This enabled ResNets with 150+ layers to train stably when networks of depth 20+ were previously unreliable.

### 2.11.4 Gradient Clipping: Taming Explosion

For exploding gradients (most common in RNNs and Transformers training on long sequences), the practical fix is gradient clipping. Compute the gradient norm:

$$\|\mathbf{g}\| = \sqrt{\sum_j g_j^2}$$

This is the Euclidean length of the gradient vector - treating all gradients from all layers concatenated into one long vector. If this length exceeds a threshold $\tau$ (typically $1.0$):

$$\mathbf{g} \leftarrow \frac{\tau}{\|\mathbf{g}\|} \cdot \mathbf{g}$$

Reading: scale the gradient by $\tau / \|\mathbf{g}\|$. If $\|\mathbf{g}\| = 10\tau$, scale by $1/10$ - so the rescaled gradient has length exactly $\tau$. The direction is preserved; only the magnitude is clipped.

Effect: the parameter update is bounded. No matter how large the gradient becomes, the weight change is at most $\eta \tau$ per step. Training remains stable even when the loss landscape has extremely steep cliffs.



## 2.12 Optimizers: Translating Gradients Into Learning

Backpropagation provides the gradient. The **optimizer** decides how to use it. We examine three progressively more sophisticated choices.

### 2.12.1 SGD With Mini-Batches

$$\theta_{t+1} = \theta_t - \eta\, g_t$$

- $\theta_t$: all parameters at training step $t$ (a single vector concatenating all weights and biases)
- $\eta$: the learning rate - a fixed positive scalar, the "step size"
- $g_t = \frac{1}{B}\sum_{i \in \text{batch}} \nabla_\theta \mathcal{L}(\mathbf{x}^{(i)}, y^{(i)})$: the gradient averaged over a mini-batch

The mini-batch gradient is an unbiased estimator of the full-dataset gradient: in expectation, it points in the right direction. But it is noisy - a different mini-batch would give a slightly different gradient. This noise, as we discuss in Chapter 4, turns out to be a useful regularizer.

### 2.12.2 Momentum: Accumulating Direction

$$\mathbf{v}_{t+1} = \beta\, \mathbf{v}_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, \mathbf{v}_{t+1}$$

- $\mathbf{v}_t$: the velocity vector (same shape as $\theta$) - the accumulated history of gradient directions
- $\beta$: the momentum coefficient, typically $0.9$. Controls how fast old velocity "decays"
- $\beta \mathbf{v}_t$: carry $90\%$ of previous velocity forward - the "inertia"
- $g_t$: add the current gradient - the "push"
- The update $\theta_{t+1} = \theta_t - \eta \mathbf{v}_{t+1}$: move in the direction of the accumulated velocity

**Why it helps:** Consider a weight that consistently has a positive gradient across many mini-batches. Its velocity accumulates: $v_t \approx g + \beta g + \beta^2 g + \cdots = g/(1-\beta) = 10g$ for $\beta = 0.9$. The effective step for this weight is $10\times$ larger than for SGD, allowing faster progress in consistent directions. For an oscillating gradient (positive then negative), the velocity averages toward zero, damping wasteful oscillations.

### 2.12.3 Adam: Per-Parameter Adaptive Rates

Adam maintains two vectors per parameter - estimates of the first and second moments of the gradient.

**First moment - gradient mean (like momentum):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

- $m_t$: the running estimate of the mean gradient
- $\beta_1 = 0.9$: decay rate - keep 90% of previous mean, add 10% of current gradient
- After many steps: $m_t$ is a weighted average of all past gradients, with recent ones weighted more

**Second moment - gradient variance:**
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

- $v_t$: the running estimate of the mean *squared* gradient (element-wise: $g_t^2$ means squaring each entry of the gradient vector)
- $\beta_2 = 0.999$: higher decay - tracks a slower-moving average
- $g_t^2$ is always non-negative; large $v_t$ means the gradient has been large and variable; small $v_t$ means stable and small

**Bias correction - compensating for the zero initialization:**

At step $t=1$: $m_1 = (1-\beta_1)g_1 = 0.1 g_1$ - only 10% of the true gradient. And $v_1 = 0.001 g_1^2$ - only 0.1% of the true squared gradient. Both estimates are biased toward zero because we started at zero.

The correction inflates the estimates to remove this bias:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

At $t=1$: $\hat{m}_1 = m_1/(1-0.9) = m_1/0.1 = 10 m_1 = g_1$ - the bias is fully corrected. As $t$ grows: $\beta_1^t \to 0$, so $1-\beta_1^t \to 1$, and the correction factor $\to 1$ (no correction needed at large $t$).

**The adaptive update:**
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Reading the fraction $\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$:
- Numerator $\hat{m}_t$: the direction to move (the recent gradient average)
- Denominator $\sqrt{\hat{v}_t} + \varepsilon$: the typical magnitude of the gradient, plus a tiny $\varepsilon = 10^{-8}$ to prevent division by zero
- Ratio: the direction, scaled inversely by the magnitude. If a parameter has had large gradients historically ($\sqrt{\hat{v}_t}$ is large), the step is scaled down. If gradients have been small and consistent ($\sqrt{\hat{v}_t}$ is small), the step is larger

**The intuition in plain language:** Adam gives every weight its own personal learning rate, automatically calibrated to the weight's gradient history. A weight that has had huge, erratic gradients gets a small effective step (to avoid instability). A weight that has had small, consistent gradients gets a larger effective step (to make progress). No manual tuning per layer or per parameter is needed - Adam adapts automatically.

**AdamW:** Adds explicit weight decay decoupled from the gradient statistics:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} - \eta\lambda\, \theta_t$$

The new term $\eta\lambda\, \theta_t$:
- $\lambda$: weight decay coefficient (typically $0.01$ to $0.1$)
- $\theta_t$: the current parameter value
- $\eta\lambda\, \theta_t$: subtract a small fraction of the current weight at every step, regardless of the gradient

This "decays" weights toward zero independently of gradient history. Standard Adam folds weight decay into the gradient computation, which distorts the adaptive scaling. AdamW separates them cleanly, resulting in better regularization for large models. It is the near-universal choice for training Transformers and large language models.



## 2.13 Learning Rate Schedules: Changing Speed Over the Journey

The right learning rate changes during training. Early on: large $\eta$ explores the loss landscape, covers ground quickly. Late on: small $\eta$ settles precisely into the bottom of the valley.

**Warmup + Cosine Decay:** The most common schedule for large models.

$$\eta_t = \begin{cases} \eta_{\max} \cdot \dfrac{t}{T_w} & \text{if } t \leq T_w \quad \text{(warmup)} \\[6pt] \eta_{\min} + \dfrac{\eta_{\max} - \eta_{\min}}{2}\!\left(1 + \cos\!\dfrac{\pi(t-T_w)}{T - T_w}\right) & \text{if } t > T_w \quad \text{(cosine decay)} \end{cases}$$

**Warmup phase** ($t \leq T_w$): Reading $\eta_{\max} \cdot t/T_w$: at step $t=0$, learning rate is $0$. At step $t=T_w$, learning rate reaches $\eta_{\max}$. The rate increases linearly.

Why warmup? At the very start, gradient estimates are high-variance (only a few examples have been processed). A large initial step might corrupt the initialization before meaningful learning can begin. Starting small and increasing gradually lets the optimizer "warm up" on safe, small steps.

**Cosine decay phase** ($t > T_w$): The $\cos$ function at $t = T_w$ equals $\cos(0) = 1$, giving $\eta_{T_w} = \eta_{\max}$ - continuous from warmup. At $t = T$ (end of training): $\cos(\pi) = -1$, giving $\eta_T = \eta_{\min}$. In between: smooth, curved decay. The cosine shape is preferred over linear because it spends more time at higher rates (more training at full speed) before gently decelerating.


## Summary

Backpropagation is the algorithm that makes deep learning possible. By applying the chain rule systematically to a computational graph in reverse order, it computes the gradient of the loss with respect to every parameter in $\mathcal{O}(F)$ time - where $F$ is the forward pass cost - rather than the $\mathcal{O}(|\theta| \cdot F)$ cost of finite differences.

The core mathematical structure is the error signal recursion $\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$, which propagates blame backward through the network layer by layer. The transpose relationship between forward and backward passes is not a choice but a mathematical consequence of the chain rule.

The vanishing and exploding gradient problems arise from the repeated multiplication of Jacobians across layers. The modern toolkit - ReLU activations, He initialization, residual connections, batch normalization, gradient clipping - addresses these problems from multiple angles simultaneously.

*We now have the engine: backpropagation computes the gradient. But we have not yet asked: what exactly is being minimized? The loss function is the compass that gives training its direction. Chapter 3 examines where loss functions come from, what they measure, and why the choice of loss is a statement about your beliefs about the world.*

---

*Continue to **[Chapter 3: Navigating the Loss Landscape - Loss Functions and Optimization](/DeepLearning/03_Optimizers_and_Loss.md)***
