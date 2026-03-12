# Autoencoders and Variational Autoencoders (VAEs)

#### Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Properties](#mathematical-properties)
3. [Classical Autoencoders](#classical-autoencoders)
4. [Regularized Autoencoders](#regularized-autoencoders)
5. [Variational Autoencoders](#variational-autoencoders---probabilistic-foundation)
6. [VAE Variants and Extensions](#vae-variants-and-extensions)
7. [Applications](#applications)
8. [Theoretical Connections](#theoretical-connections)
9. [Conclusion](#conclusion)

## Introduction

Modern machine learning confronts data of staggering dimensionality - a single HD video frame occupies millions of dimensions, and domains like medical imaging, genomics, and sensor networks produce observations that defy human intuition.

Yet despite this, a profound insight underlies machine learning's success: **real-world data does not uniformly occupy the ambient high-dimensional space.** A random point in $\mathbb{R}^{65{,}536}$ is extraordinarily unlikely to be a natural image. Most configurations of pixel values represent pure noise - statistically independent, with no coherent structure. Natural images, by contrast, exhibit strong regularities: smooth regions, abrupt edges, repeating textures, recognizable shapes. These constraints mean they occupy a tiny, highly structured subset of pixel space.

This observation crystallizes in the **manifold hypothesis**: high-dimensional data concentrates near *low-dimensional manifolds* embedded within the ambient space. A manifold is a geometric structure that locally resembles Euclidean space but may have complex global shape - like the surface of a sphere, which is 2D yet curves through 3D. Similarly, natural images form a manifold whose *intrinsic dimensionality* (meaningful degrees of freedom) is far smaller than the pixel count.

This is why learning is possible despite the curse of dimensionality. If data lies on a $d$-dimensional manifold with $d \ll D$, sample complexity depends on $d$, not $D$ - we need only enough samples to cover the manifold.

**Autoencoders** emerge naturally from this hypothesis. They are neural network architectures designed to learn compact representations by discovering underlying manifold structure. The architecture has two coupled components:
- **Encoder** $E_\phi: \mathbb{R}^D \to \mathbb{R}^d$ - maps high-dimensional input $\mathbf{x}$ to low-dimensional code $\mathbf{z} = E_\phi(\mathbf{x})$
- **Decoder** $D_\theta: \mathbb{R}^d \to \mathbb{R}^D$ - reconstructs $\hat{\mathbf{x}} = D_\theta(\mathbf{z})$ from the code

The complete autoencoder is the composition $\hat{\mathbf{x}} = (D_\theta \circ E_\phi)(\mathbf{x})$, approximating the identity through a dimensional bottleneck. Training minimizes reconstruction error:

$$\min_{\phi,\theta}\; \mathbb{E}_{\mathbf{x} \sim p_\text{data}}\left[\ell\!\left(\mathbf{x},\, D_\theta(E_\phi(\mathbf{x}))\right)\right]$$

where $\ell$ is typically **MSE** $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ for continuous data or **binary cross-entropy** for binary data.

After training, the decoder's range $D_\theta(\mathbb{R}^d)$ approximates the data manifold $\mathcal{M}$: on-manifold points reconstruct well; off-manifold points reconstruct poorly.



## Mathematical Properties

### Calculus

Autoencoder training reduces to optimization - finding parameters minimizing reconstruction error - which requires computing gradients with respect to millions of parameters.

#### Derivatives and Rates of Change

The derivative $f'(x)$ measures the instantaneous rate of change of $f$ at $x$. It tells us how to adjust parameters to reduce loss:
- If $\ell'(\theta) > 0$: increasing $\theta$ increases loss → decrease $\theta$
- If $\ell'(\theta) < 0$: decreasing $\theta$ increases loss → increase $\theta$
- The magnitude $|\ell'(\theta)|$ measures sensitivity

The gradient descent update formalizes this:

$$\theta_{t+1} = \theta_t - \eta\, \ell'(\theta_t)$$

Common activation derivatives (essential for backpropagation):
- **Sigmoid**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ - expressible from its own output
- **Tanh**: $\tanh'(z) = 1 - \tanh^2(z)$ - same property
- **ReLU**: $\text{ReLU}'(z) = \mathbf{1}[z > 0]$ - zero for negative inputs, one otherwise

#### Multivariable Calculus and Partial Derivatives

For $f: \mathbb{R}^n \to \mathbb{R}$, the partial derivative $\partial f/\partial x_i$ measures change when varying only $x_i$. The **gradient** collects all partials:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right]^\top$$

It points in the direction of steepest ascent; $-\nabla f$ is steepest descent. For small displacement $\Delta \mathbf{x}$:

$$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \Delta\mathbf{x}$$

For vector-valued functions $f: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** $J_f \in \mathbb{R}^{m \times n}$ contains all first-order partials: $(J_f)_{ij} = \partial f_i / \partial x_j$. Each layer's derivative is a Jacobian - for $\mathbf{a} = \sigma(W\mathbf{x} + \mathbf{b})$, the Jacobian w.r.t. $\mathbf{x}$ is $\text{diag}(\sigma'(W\mathbf{x}+\mathbf{b})) \cdot W$.

#### The Chain Rule

The chain rule is the mathematical engine powering backpropagation. For compositions $h = g \circ f$:

$$J_h = J_g \cdot J_f$$

For a $L$-layer network $h = f_L \circ \cdots \circ f_1$:

$$J_h = J_{f_L} \cdots J_{f_2} J_{f_1}$$

**Backward mode** (backpropagation) computes this product right-to-left, reusing intermediate results. This is efficient when we have *many parameters but few outputs* (a scalar loss): all gradients are computed in a single backward pass, each step costing $O(n)$ for an $n$-dimensional layer.

#### Second Derivatives and the Hessian

The **Hessian** $H \in \mathbb{R}^{n \times n}$ contains all second-order partials ($H_{ij} = \partial^2 f / \partial x_i \partial x_j$) and characterizes local curvature. At a critical point where $\nabla f = 0$:
- $H$ **positive definite** → local minimum
- $H$ **negative definite** → local maximum
- $H$ **indefinite** (mixed eigenvalues) → saddle point

Large positive eigenvalues indicate steep curvature (narrow valleys); small eigenvalues indicate flat regions. Second-order methods like Newton's method exploit Hessian information: $\mathbf{x}_{t+1} = \mathbf{x}_t - H^{-1}\nabla f(\mathbf{x}_t)$. In practice, computing and inverting $H$ is $O(n^3)$ and prohibitive at scale - **quasi-Newton methods** (e.g., L-BFGS) approximate it from recent gradients.



### Linear Algebra

Neural networks implement linear transformations composed with nonlinearities, making linear algebra their natural computational language.

#### Vectors, Dot Products, and Norms

A vector $\mathbf{x} \in \mathbb{R}^n$ represents features (e.g., flattening a $28\times28$ image to $\mathbb{R}^{784}$). The **dot product** $\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^\top \mathbf{y}$ measures alignment - positive means same direction, negative means opposite, zero means orthogonal.

Key norms:
- **Euclidean (L2)**: $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ - used in MSE loss and L2 regularization
- **L1**: $\|\mathbf{x}\|_1 = \sum_i |x_i|$ - sums absolute values, induces sparsity

#### Matrices and Linear Transformations

A weight matrix $W \in \mathbb{R}^{m \times n}$ transforms $\mathbb{R}^n \to \mathbb{R}^m$. A neural layer computes $\mathbf{a} = \sigma(W\mathbf{x} + \mathbf{b})$. The transpose $W^\top$ swaps rows and columns - it appears in backpropagation: if forward propagation uses $W$, backward propagation uses $W^\top$.

Matrix multiplication is **associative but not commutative**: $(AB)C = A(BC)$ but $AB \neq BA$ in general. Associativity is critical for efficient computation of multi-layer gradient products.

#### Eigenvalues and Eigenvectors

For $A \in \mathbb{R}^{n\times n}$, eigenvector $\mathbf{v} \neq 0$ satisfies $A\mathbf{v} = \lambda\mathbf{v}$: $A$ stretches $\mathbf{v}$ by factor $\lambda$ without changing direction. The spectral theorem guarantees symmetric matrices decompose as $A = Q\Lambda Q^\top$ (orthonormal eigenvectors, real eigenvalues). Eigenvalues of the Hessian determine critical point type; eigenvalues of weight matrices affect gradient flow (large → exploding, small → vanishing).

#### Key Matrix Quantities

| Quantity | Definition | Role |
|---|---|---|
| Frobenius norm | $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$ | L2 regularization |
| Spectral norm | $\|A\|_2 = \max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\|$ | Gradient stability |
| Trace | $\text{tr}(A) = \sum_i A_{ii}$ | Sum of eigenvalues; total variance |
| Determinant | $\det(A)$ | Volume scaling; $\det(A)=0$ means singular |



### Probability and Random Variables

VAEs fundamentally rest on probability theory, treating encoding and decoding as *probabilistic inference*.

#### Distributions and Expectations

The **expected value** $\mathbb{E}[X]$ is the probability-weighted average of $X$. Expectations are linear: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ - this underlies stochastic gradient descent, where minibatch gradients are *unbiased estimators* of true gradients. **Variance** $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$ measures spread; for $n$ i.i.d. samples, $\text{Var}(\bar{X}) = \sigma^2/n$, explaining the diminishing returns of larger batches.

#### Key Distributions

**Gaussian** $\mathcal{N}(\mu, \sigma^2)$: the central limit theorem makes it ubiquitous; it maximizes entropy for fixed mean and variance; Gaussian noise assumptions yield MSE loss via MLE. The **multivariate Gaussian** in $\mathbb{R}^n$:

$$p(\mathbf{x}) = (2\pi)^{-n/2}|\Sigma|^{-1/2}\exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

The **standard normal** $\mathcal{N}(0, I)$ serves as the prior in most VAEs - simple, unimodal, rotationally symmetric. **Bernoulli** ($p$) models binary inputs (e.g., pixel intensities), with mean $p$ and variance $p(1-p)$.

#### Conditional Probability and Bayes' Theorem

Bayes' theorem inverts conditioning: $p(\mathbf{x}|\mathbf{y}) = p(\mathbf{y}|\mathbf{x})p(\mathbf{x})/p(\mathbf{y})$. In VAE terminology:
- $p_\theta(\mathbf{x}|\mathbf{z})$ - **decoder** (likelihood)
- $p(\mathbf{z})$ - **prior** over latents
- $p_\theta(\mathbf{z}|\mathbf{x})$ - **posterior** (intractable)
- $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$ - **marginal likelihood** (intractable)

VAEs approximate the intractable posterior with a learned distribution $q_\phi(\mathbf{z}|\mathbf{x})$.

#### Entropy, Cross-Entropy, and KL Divergence

**Entropy** $H(X) = -\sum_x P(x)\log P(x)$ measures uncertainty - maximum for uniform distributions, zero for deterministic variables. Shannon's source coding theorem establishes $H(X)$ as the fundamental limit for lossless compression.

**Cross-entropy** $H(P,Q) = -\sum_x P(x)\log Q(x)$ measures the cost of encoding samples from $P$ using a code optimized for $Q$. When $Q \neq P$, we pay an extra penalty: $H(P,Q) \geq H(P)$. In classification, minimizing cross-entropy encourages model predictions to match the true label distribution.

**KL divergence** $\text{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$ measures distributional discrepancy:
- Non-negative: $\text{KL}(P\|Q) \geq 0$, with equality iff $P = Q$
- *Asymmetric*: $\text{KL}(P\|Q) \neq \text{KL}(Q\|P)$ in general
- Related to cross-entropy: $\text{KL}(P\|Q) = H(P,Q) - H(P)$

In VAEs, $\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ regularizes the latent space by penalizing deviation of the approximate posterior from the prior.

#### Maximum Likelihood Estimation

Given dataset $\{x_i\}_{i=1}^N$ i.i.d. from $p_\theta$, MLE maximizes the log-likelihood:

$$\hat{\theta}_\text{MLE} = \arg\max_\theta \sum_{i=1}^N \log p_\theta(x_i)$$

- Under **Gaussian noise** ($y = f_\theta(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,\sigma^2)$), MLE reduces to minimizing MSE
- Under **Bernoulli output**, MLE reduces to cross-entropy

For VAEs, the marginal $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$ is intractable, motivating the ELBO as a tractable surrogate.



### Information Theory and Compression

Information theory (Shannon, 1948) quantifies information, compression limits, and uncertainty - illuminating why autoencoders work.

#### Entropy and Mutual Information

The **information content** (surprisal) of outcome $x$ is $I(x) = -\log P(x)$: rare events carry high information; common events carry little. Entropy $H(X) = \mathbb{E}[I(X)]$ is the average surprise, and equals the minimum average code length for lossless compression.

**Mutual information** $I(X;Y) = H(X) - H(X|Y)$ quantifies information shared between $X$ and $Y$. It satisfies:
- Non-negativity: $I(X;Y) \geq 0$, with equality iff $X \perp Y$
- Symmetry: $I(X;Y) = I(Y;X)$
- *Data processing inequality*: for Markov chain $X \to Y \to Z$, $\;I(X;Z) \leq I(X;Y)$

In VAEs, $\mathbb{E}_x[\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))]$ equals the mutual information $I(X;Z)$ under the variational distribution - showing that VAEs trade off *maximizing information* (reconstruction) against *minimizing mutual information* (regularization/compression).

#### Rate-Distortion Theory

Rate-distortion theory studies the trade-off between compression rate (bits/sample) and distortion (reconstruction error). The **rate-distortion function** $R(D)$ defines the minimum bits per sample required to achieve expected distortion $\leq D$. For Gaussian sources with squared error:

$$R(D) = \frac{1}{2}\log\frac{\sigma^2}{D}$$

Halving distortion costs one extra bit. VAEs directly instantiate this: the ELBO decomposes as *reconstruction* (distortion) minus *KL divergence* (rate). The $\beta$-VAE hyperparameter explicitly controls this trade-off.

#### The Information Bottleneck

The information bottleneck (Tishby et al., 1999) seeks a representation $Z$ that compresses $X$ (minimize $I(X;Z)$) while retaining task-relevant information $Y$ (maximize $I(Z;Y)$):

$$\max_{p(\mathbf{z}|\mathbf{x})}\; I(Z;Y) - \beta\, I(X;Z)$$

For autoencoders where $Y = X$, this becomes $(1-\beta)I(X;Z)$. Standard VAEs correspond to $\beta = 1$; $\beta$-VAEs use $\beta > 1$ for stronger compression. The bottleneck forces the representation to retain the most informative aspects of $X$ while discarding noise.



## Classical Autoencoders

### Architecture and Training

An autoencoder has:
- **Encoder** $E_\phi: \mathbb{R}^D \to \mathbb{R}^d$: $\;\mathbf{z} = E_\phi(\mathbf{x})$
- **Decoder** $D_\theta: \mathbb{R}^d \to \mathbb{R}^D$: $\;\hat{\mathbf{x}} = D_\theta(\mathbf{z})$

Training minimizes empirical reconstruction error:

$$\min_{\phi,\theta}\; \frac{1}{N}\sum_{i=1}^N \|x^{(i)} - D_\theta(E_\phi(x^{(i)}))\|^2$$

For **binary data**, binary cross-entropy replaces MSE - arising naturally from MLE under a Bernoulli observation model.

The typical architecture is symmetric ("bowtie" / "hourglass") with decreasing dimensionality in the encoder and increasing in the decoder, using ReLU activations in hidden layers. The final encoder layer is often linear; the final decoder activation depends on the output type (sigmoid for $[0,1]$, identity for real-valued).

Training dynamics follow standard supervised learning. Gradients flow end-to-end via:

$$\frac{\partial \ell}{\partial \phi} = \underbrace{\frac{\partial \ell}{\partial \hat{\mathbf{x}}}}_{\text{loss grad}}\underbrace{\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{z}}}_{\text{decoder Jacobian}}\underbrace{\frac{\partial \mathbf{z}}{\partial \phi}}_{\text{encoder Jacobian}}$$

The encoder and decoder *co-adapt*: the encoder learns codes the decoder can reconstruct; the decoder learns to reconstruct from whatever codes the encoder provides.

#### The Bottleneck

The key feature is $d \ll D$ (the **undercomplete** regime). This prevents the trivial identity solution - insufficient capacity forces the network to discover *regularities* and discard noise. The compression ratio $D/d$ quantifies the constraint (e.g., $784/32 \approx 24.5\times$ for MNIST).

Information-theoretically: $I(X;Z) \leq H(Z)$, which is bounded by latent dimensionality. The autoencoder must use this limited capacity efficiently.

#### Overcomplete Autoencoders

When $d \geq D$, the trivial identity solution is available (encoder copies input, decoder copies back). To prevent this, **regularization** is required - sparsity penalties, denoising, or contractive constraints force meaningful structure rather than memorization.

#### Depth and Hierarchical Features

Deep architectures learn hierarchical representations: early layers extract low-level features (edges, textures), middle layers combine them (corners, curves), and deep layers produce high-level semantic codes. The decoder reverses this hierarchy. Empirically, deep autoencoders outperform shallow ones with equal parameter counts, thanks to the inductive bias toward hierarchical structure.

#### Overfitting and Regularization

Signs of overfitting: decreasing training error but stagnant or growing validation error. Standard mitigations:
- **Weight decay (L2)**: Adds $\lambda\|\theta\|^2$ penalty - penalizes large weights, effective for $\lambda \in [10^{-5}, 10^{-3}]$
- **Dropout**: Randomly zeroes activations ($p \approx 0.2$–$0.5$) during training, preventing co-adaptation
- **Early stopping**: Halt when validation error stops improving
- **Data augmentation**: Generate additional examples via structure-preserving transformations



### Connection to PCA

#### PCA: Optimal Linear Subspace

PCA finds the $d$-dimensional linear subspace minimizing squared reconstruction error by keeping the top-$d$ eigenvectors of the data covariance $\Sigma$. Encoding: $\mathbf{z} = U^\top \mathbf{x}$; decoding: $\hat{\mathbf{x}} = U\mathbf{z}$. The reconstruction error equals the *sum of discarded eigenvalues*:

$$\mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|^2] = \sum_{i=d+1}^D \lambda_i$$

#### Linear Autoencoders = PCA

A landmark result (Baldi & Hornik, 1989): *linear autoencoders learn exactly the PCA solution*. The optimal decoder satisfies $V = W^\top$, and the encoder weights converge to the top-$d$ principal components regardless of initialization. This establishes that:
1. PCA is optimal among *linear* dimensionality reduction methods
2. Autoencoders generalize PCA by discovering *curved* manifolds with nonlinear activations
3. PCA provides theoretical grounding for what linear autoencoders learn

#### Beyond Linear: Nonlinear Manifolds

Real data rarely lives on linear subspaces. Consider points on a circle in $\mathbb{R}^2$ - intrinsically 1D, but PCA fails because the circle is curved. A nonlinear autoencoder can learn the angle as a 1D representation and decode back to $(x,y)$ coordinates. More generally, deep nonlinear autoencoders learn coordinate charts on curved $d$-dimensional manifolds within $\mathbb{R}^D$ - representations that linear methods cannot capture.



## Regularized Autoencoders

Beyond the dimensional bottleneck, additional regularization improves generalization and representation quality.

### Sparse Autoencoders

Sparse representations - where most latent components are near zero - are desirable for interpretability, efficiency, and alignment with biological coding principles.

**L1 sparsity**: Add penalty $\lambda\|E_\phi(\mathbf{x})\|_1$ to the reconstruction loss. The L1 norm produces truly sparse solutions (exact zeros), unlike L2 which merely shrinks all values. Gradient: $\partial\Omega/\partial\mathbf{z} = \text{sign}(\mathbf{z})$, pushing values toward zero.

**KL sparsity** (Ng, 2011): Enforce target average activation $\rho$ (typically 0.05) per latent dimension $j$:

$$\Omega = \sum_j \text{KL}\!\left(\rho \,\|\, \hat{\rho}_j\right) = \sum_j \left[\rho\log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}\right]$$

where $\hat{\rho}_j$ is the running average activation. This penalizes over- and under-active dimensions symmetrically.

Sparse autoencoders trained on natural images learn **Gabor-filter-like receptive fields** - mirroring V1 simple cells in visual cortex (Olshausen & Field, 1996). ReLU activations naturally support sparsity since negative pre-activations yield exact zeros. Implementation requires tuning $\lambda$ via grid search, monitoring reconstruction quality and fraction of near-zero activations.

### Denoising Autoencoders

Denoising autoencoders (DAE; Vincent et al., 2008) train to reconstruct *clean* inputs from *corrupted* versions $\tilde{\mathbf{x}} \sim C(\cdot|\mathbf{x})$:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}}\,\mathbb{E}_{\tilde{\mathbf{x}} \sim C(\cdot|\mathbf{x})}\left[\|\mathbf{x} - D_\theta(E_\phi(\tilde{\mathbf{x}}))\|^2\right]$$

Common corruption types: **additive Gaussian noise** $\tilde{\mathbf{x}} = \mathbf{x} + \varepsilon$; **masking noise** (randomly zero out pixels with probability $p$); **salt-and-pepper noise** (randomly set to 0 or 1). Note: *loss compares reconstruction to the clean $\mathbf{x}$, not the corrupted $\tilde{\mathbf{x}}$.*

Why denoising helps:
- **Prevents identity solution**: Since $\tilde{\mathbf{x}} \neq \mathbf{x}$, copying doesn't work even in overcomplete regimes
- **Robust features**: Only features capturing genuine signal (not noise) can denoise reliably
- **Implicit regularization**: Corruption acts like dropout on inputs, improving generalization
- **Manifold learning**: The decoder learns to project corrupted (off-manifold) points back onto the manifold

For small corruption, the optimal denoising function approximates a *vector field pointing toward the data manifold* (Vincent et al., 2008). **Stacked denoising autoencoders** (SDAE) build deep representations via layer-wise pretraining, although this practice is less common in the modern era of improved initializers.

### Contractive Autoencoders

Contractive autoencoders (CAE; Rifai et al., 2011) explicitly penalize encoder *sensitivity* to input perturbations. The Jacobian $J_E = \partial E_\phi / \partial \mathbf{x} \in \mathbb{R}^{d \times D}$ captures how changes in $\mathbf{x}$ propagate to $\mathbf{z}$. The contractive penalty is:

$$\mathcal{L} = \mathbb{E}_\mathbf{x}\left[\|\mathbf{x} - D_\theta(E_\phi(\mathbf{x}))\|^2 + \lambda\|J_E(\mathbf{x})\|_F^2\right]$$

For a one-hidden-layer encoder (with sigmoid activations), $\|J_E\|_F^2$ can be computed efficiently using hidden activations $\mathbf{h}$:

$$\|J_E\|_F^2 = \left\|W_1^\top \text{diag}(\mathbf{h} \odot (1-\mathbf{h}))\right\|_F^2$$

*Geometric interpretation*: The penalty discourages sensitivity in all input directions. Since data concentrates near the manifold, most directions are approximately orthogonal to it - so the penalty primarily enforces invariance in off-manifold directions while allowing variation *along* the manifold.

Alain & Bengio (2013) showed that for infinitesimal Gaussian noise, minimizing DAE reconstruction error is equivalent to a score-matching objective closely related to the contractive penalty - revealing a surprising theoretical connection between the two. In practice, DAEs are simpler to implement; CAEs offer more direct geometric control.



## Variational Autoencoders - Probabilistic Foundation

VAEs transform autoencoders from deterministic dimensionality reduction into **probabilistic generative models**. This shift enables principled sample generation, uncertainty quantification, and theoretical connections to Bayesian inference.

### From Deterministic to Probabilistic Encoding

Classical autoencoders map $\mathbf{x} \mapsto \mathbf{z}$ deterministically. VAEs instead treat the encoder as defining a *distribution* $q_\phi(\mathbf{z}|\mathbf{x})$ over latent codes. Advantages:
1. **Generative modeling**: Sample $\mathbf{z} \sim p(\mathbf{z})$, then $\mathbf{x} \sim p_\theta(\cdot|\mathbf{z})$
2. **Uncertainty quantification**: $q_\phi(\mathbf{z}|\mathbf{x})$ expresses latent uncertainty
3. **Principled objective**: Likelihood maximization provides theoretical foundation
4. **Regularization**: The prior $p(\mathbf{z})$ naturally regularizes the latent space

### The Generative Model

The VAE posits a generative process:
1. Sample latent: $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(0, I)$
2. Generate observation: $\mathbf{x} \sim p_\theta(\mathbf{x}|\mathbf{z})$

The decoder network parameterizes $p_\theta(\mathbf{x}|\mathbf{z})$:
- **Continuous**: $p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x};\, \mu_\theta(\mathbf{z}),\, \sigma^2 I)$
- **Binary**: $p_\theta(\mathbf{x}|\mathbf{z}) = \prod_i \text{Bernoulli}(x_i;\, \mu_\theta(\mathbf{z})_i)$

The marginal likelihood $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$ is *intractable* - this integral has no closed form over a neural network decoder. This motivates the variational approach.

### Variational Approximation

Exact posterior $p_\theta(\mathbf{z}|\mathbf{x}) = p_\theta(\mathbf{x}|\mathbf{z})\,p(\mathbf{z})\,/\,p_\theta(\mathbf{x})$ is also intractable. VAEs approximate it with a learned Gaussian:

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}\!\left(\mathbf{z};\; \mu_\phi(\mathbf{x}),\; \text{diag}(\sigma_\phi^2(\mathbf{x}))\right)$$

where $\mu_\phi$ and $\sigma_\phi$ are both outputs of the encoder network.

### The Evidence Lower Bound (ELBO)

#### Derivation

Since $p_\theta(\mathbf{x})$ is intractable, we maximize a lower bound. Starting from the log-likelihood and applying Jensen's inequality ($\log$ is concave):

$$\log p_\theta(\mathbf{x}) \geq \underbrace{\mathbb{E}_{q_\phi}\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right]}_{\text{reconstruction}} - \underbrace{\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z}))}_{\text{regularization}} \;=:\; \mathcal{L}(\theta, \phi;\, \mathbf{x})$$

The ELBO $\mathcal{L}$ has two terms:
- **Reconstruction term**: Expected log-likelihood of $\mathbf{x}$ under the decoder - encourages accurate reconstructions
- **KL term**: Divergence between approximate posterior and prior - regularizes the latent space

#### Tightness

The gap between the ELBO and the true log-likelihood equals the KL to the *true* posterior:

$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi;\, \mathbf{x}) + \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p_\theta(\mathbf{z}|\mathbf{x}))$$

Since $\text{KL} \geq 0$, the ELBO is always a lower bound. Maximizing w.r.t. $\phi$ tightens the bound; maximizing w.r.t. $\theta$ improves the model.

### Closed-Form KL for Gaussian Posteriors

For $q_\phi = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p = \mathcal{N}(0, I)$, the KL has a closed form:

$$\text{KL}(q_\phi \| p) = \frac{1}{2}\sum_i \left[\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2\right]$$

Each term has an intuitive role:
- $\sigma_i^2 - 1$: penalizes variance $\neq 1$
- $\mu_i^2$: penalizes non-zero mean
- $-\log\sigma_i^2$: entropy term preventing variance collapse to zero

Gradients: $\partial \text{KL}_i / \partial \mu_i = \mu_i$ (push toward 0), $\partial \text{KL}_i / \partial \sigma_i^2 = \frac{1}{2}(1 - 1/\sigma_i^2)$ (push toward 1).

### The Reparameterization Trick

To compute $\nabla_\phi\,\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]$, we cannot naively move the gradient inside since $\phi$ parameterizes the sampling distribution. The **reparameterization trick** resolves this:

1. Sample $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$ (independent of $\phi$)
2. Set $\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}$

Now $\mathbf{z}$ is a *deterministic function of $\phi$*, and gradients can flow through:

$$\nabla_\phi\,\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] = \mathbb{E}_{\boldsymbol{\varepsilon}}\left[\nabla_\phi \log p_\theta(\mathbf{x}|\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon})\right]$$

In practice, a single sample $\boldsymbol{\varepsilon}$ provides a low-variance, unbiased gradient estimate - enabling stable training. The backpropagation chain is:

$$\frac{\partial}{\partial \phi}: \quad \frac{\partial \log p_\theta(\mathbf{x}|\mathbf{z})}{\partial \mathbf{z}} \cdot \begin{cases} \partial \mathbf{z}/\partial \boldsymbol{\mu} = I \\ \partial \mathbf{z}/\partial \boldsymbol{\sigma} = \text{diag}(\boldsymbol{\varepsilon}) \end{cases} \cdot \frac{\partial (\boldsymbol{\mu}, \boldsymbol{\sigma})}{\partial \phi}$$

### Complete VAE Training Algorithm

**Each minibatch step**:
1. **Encode**: Compute $\boldsymbol{\mu}^{(i)} = \mu_\phi(x^{(i)})$, $\boldsymbol{\sigma}^{(i)} = \sigma_\phi(x^{(i)})$
2. **Sample**: Draw $\boldsymbol{\varepsilon}^{(i)} \sim \mathcal{N}(0,I)$; set $z^{(i)} = \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\varepsilon}^{(i)}$
3. **Decode**: Compute $\log p_\theta(x^{(i)}|z^{(i)})$
4. **Compute ELBO**: $\mathcal{L} = \frac{1}{B}\sum_i \log p_\theta(x^{(i)}|z^{(i)}) - \frac{1}{2B}\sum_i\sum_j\left[(\sigma^{(i)}_j)^2 + (\mu^{(i)}_j)^2 - 1 - \log(\sigma^{(i)}_j)^2\right]$
5. **Update**: $\theta \leftarrow \theta + \eta\nabla_\theta\mathcal{L}$; $\phi \leftarrow \phi + \eta\nabla_\phi\mathcal{L}$

**At test time**:
- *Encode*: Use $\boldsymbol{\mu}_\phi(\mathbf{x})$ deterministically
- *Generate*: Sample $\mathbf{z} \sim \mathcal{N}(0,I)$, decode $\hat{\mathbf{x}} = \mu_\theta(\mathbf{z})$

### Practical Considerations

**Variance parameterization**: Output $\log\sigma^2$ instead of $\sigma$ to guarantee positivity: `sigma = torch.exp(0.5 * log_var)`.

**Loss implementation**:
```python
# Binary data
recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')

# Continuous data (MSE ~ Gaussian likelihood with σ²=1)
recon_loss = F.mse_loss(x_hat, x, reduction='sum')

# KL term
kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

loss = recon_loss + kl  # minimize negative ELBO
```

**Posterior collapse**: When $q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z})$ for all $\mathbf{x}$, the encoder ignores the input. This occurs when the decoder is powerful enough to model $p(\mathbf{x})$ without using $\mathbf{z}$. Mitigations:
- **KL annealing**: Gradually increase KL weight $\beta_t$ from 0 to 1 during training
- **Free bits**: Only penalize KL above threshold $\lambda$ per dimension: $\max(\lambda, \text{KL}_i)$
- **Weaker decoder**: Constrain decoder capacity to force reliance on $\mathbf{z}$

**Latent dimension**: Typical ranges - MNIST: $d = 2$–$50$; CelebA: $d = 128$–$512$; ImageNet: $d = 256$–$2048$. Monitor both reconstruction loss and KL divergence; near-zero KL signals underutilized latents.



## VAE Variants and Extensions

### β-VAE: Disentangled Representations

β-VAE (Higgins et al., 2017) strengthens the KL penalty to encourage *disentanglement* - independent latent dimensions corresponding to independent factors of variation:

$$\mathcal{L}_\beta = \mathbb{E}_{q_\phi}\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right] - \beta \cdot \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**Why $\beta > 1$ promotes disentanglement**: With a factorized prior $p(\mathbf{z}) = \prod_i p(z_i)$, the KL term penalizes complex dependencies in $q_\phi(\mathbf{z}|\mathbf{x})$. Entangled factors increase KL; to minimize the $\beta$-weighted penalty, the model learns to separate independent factors into independent dimensions. This is the *information bottleneck* at work: $\mathcal{L}_\beta \approx \mathcal{R} - \beta \cdot I(X;Z)$ forces each dimension to encode maximum information about one factor rather than distributing it redundantly.

**Trade-off**: Larger $\beta$ improves disentanglement but degrades reconstruction quality (blurrier outputs). Typical $\beta \in [2, 10]$.

**Disentanglement metrics**: Mutual Information Gap (MIG), Separated Attribute Predictability (SAP), and DCI (Disentanglement-Completeness-Informativeness). All compare latent dimensions against ground-truth generative factors.

**Theoretical caveat**: Locatello et al. (2019) showed unsupervised disentanglement is fundamentally impossible without inductive biases or weak supervision - different models with equal ELBO can learn different disentangled representations, indistinguishable by unsupervised metrics. Successful disentanglement requires architectural priors, weak supervision, or structured assumptions.

### Conditional VAE (CVAE)

Both encoder and decoder condition on auxiliary variable $\mathbf{c}$ (class labels, attributes, text):
- Encoder: $q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c})$
- Decoder: $p_\theta(\mathbf{x}|\mathbf{z}, \mathbf{c})$

**Implementation**: Concatenate $\mathbf{c}$ to inputs of both networks.

**Applications**:
- *Class-conditional generation*: Fix desired $\mathbf{c}$, sample $\mathbf{z} \sim p(\mathbf{z})$, decode
- *Semi-supervised learning*: Labeled pairs use $\mathbf{c}$ directly; unlabeled $\mathbf{x}$ marginalizes over classes
- *Attribute manipulation*: Generate faces with specific attributes (age, expression, accessories)

### Hierarchical VAEs

Multiple stochastic layers model distributions at multiple abstraction levels. For two latent layers:

$$\mathbf{z}_2 \sim p(\mathbf{z}_2), \qquad \mathbf{z}_1 \sim p_\theta(\mathbf{z}_1|\mathbf{z}_2), \qquad \mathbf{x} \sim p_\theta(\mathbf{x}|\mathbf{z}_1, \mathbf{z}_2)$$

The ELBO extends with conditional KL terms at each layer. Benefits: greater capacity, hierarchical features (global semantics in $\mathbf{z}_2$, local details in $\mathbf{z}_1$), and reduced posterior collapse. **Ladder VAE** combines top-down and bottom-up paths with skip connections for improved gradient flow.

### Discrete Latent Variables and VQ-VAE

Discrete latents offer interpretability and compatibility with autoregressive priors.

**Gumbel-Softmax** (Jang et al., 2017): Differentiable approximation to categorical sampling. Sample $g_k \sim \text{Gumbel}(0,1)$; relax the argmax with a softmax at temperature $\tau$:

$$z_k = \frac{\exp((\log\pi_k + g_k)/\tau)}{\sum_j \exp((\log\pi_j + g_j)/\tau)}$$

As $\tau \to 0$, this approaches one-hot sampling. Training uses annealing from high $\tau$ (smooth) toward 0 (discrete).

**VQ-VAE** (van den Oord et al., 2017) uses a learned discrete **codebook** $\{e_1, \ldots, e_K\} \subset \mathbb{R}^d$. The encoder produces $z_e(\mathbf{x})$; quantization maps it to the nearest codebook vector:

$$z_q(\mathbf{x}) = e_k, \qquad k = \arg\min_j \|z_e(\mathbf{x}) - e_j\|^2$$

Training uses: (1) a *straight-through estimator* (copy gradients from $z_q$ to $z_e$); (2) a *commitment loss* $\|z_e - \text{sg}[z_q]\|^2$; (3) a *codebook loss* $\|\text{sg}[z_e] - z_q\|^2$. Advantages: no posterior collapse, supports autoregressive priors over discrete codes, interpretable assignments. **VQ-VAE-2** (Razavi et al., 2019) extends this hierarchically to produce high-fidelity $1024\times1024$ images.



## Applications

### Dimensionality Reduction and Visualization

Autoencoders provide *nonlinear* dimensionality reduction, complementing linear PCA. With $d = 2$ or $d = 3$, latent codes enable direct visualization. Unlike t-SNE/UMAP (which provide static embeddings), autoencoders learn explicit mappings that generalize to new data.

**Semantic interpolation** (VAEs): Interpolating between two latent codes and decoding produces semantically meaningful morphing sequences - e.g., a smooth face-to-face transition.

### Anomaly Detection

Autoencoders trained on normal data reconstruct normal examples well but fail on anomalies. Reconstruction error serves as an anomaly score:
1. Train on normal data
2. Set threshold $\tau$ (e.g., 95th percentile of validation errors)
3. Flag $\mathbf{x}_\text{test}$ as anomaly if $\|\mathbf{x}_\text{test} - f(\mathbf{x}_\text{test})\|^2 > \tau$

Applications span fraud detection, network intrusion, manufacturing defects, and medical imaging. For VAEs, the ELBO provides a richer anomaly score incorporating both reconstruction quality and KL divergence.

### Image Denoising and Inpainting

Denoising autoencoders naturally extend to practical denoising: feed noisy image, receive denoised output. Works for Gaussian noise, salt-and-pepper noise, JPEG artifacts, and sensor noise. **Inpainting** reconstructs missing regions by masking pixels during training and learning to predict them from visible context. Combining with adversarial training (VAE-GAN) produces sharper results.

### Representation Learning and Transfer

Pre-train an autoencoder on large *unlabeled* data, then fine-tune the encoder for supervised tasks on small labeled datasets. This leverages abundant unlabeled data to improve initialization, reduce labeled data requirements, and improve generalization. While contrastive methods (SimCLR, MoCo) and masked language modeling (BERT) dominate modern self-supervised learning, autoencoders remain valuable in domains with rich unlabeled data and limited labels.

### Generative Modeling

VAEs enable **latent space arithmetic** for disentangled representations:

$$\mathbf{z}_\text{glasses} = \mathbf{z}_\text{man+glasses} - \mathbf{z}_\text{man}$$

Adding this "glasses vector" to any face code adds glasses. Applications include data augmentation, creative content generation, drug discovery (molecular structures), and scientific simulation.

### Lossy Compression

Autoencoders compress data at ratio $D/d$, trading distortion against rate. End-to-end learned compression (Ballé et al., 2018) - using convolutional VAEs with quantized latents and arithmetic coding - matches or exceeds JPEG/JPEG2000 on rate-distortion curves, especially at low bitrates.



## Theoretical Connections

### Information Theory and VAEs

The VAE objective directly instantiates rate-distortion optimization:

$$\max\; \mathbb{E}[\log p_\theta(\mathbf{x}|\mathbf{z})] - I(X;Z)$$

since $\mathbb{E}_\mathbf{x}[\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))] = I(X;Z)$. This balances low distortion (high reconstruction likelihood) against low rate (small mutual information). β-VAE makes this explicit: $\beta > 1$ tightens the rate constraint; $\beta < 1$ relaxes it.

### Bayesian Inference and Amortized Variational Inference

VAEs exemplify **amortized variational inference**: rather than solving a separate optimization for each $\mathbf{x}$, a single inference network $q_\phi(\mathbf{z}|\mathbf{x})$ generalizes across all observations. The fundamental identity:

$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi;\, \mathbf{x}) + \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$

shows that maximizing the ELBO simultaneously: (1) maximizes model evidence and (2) minimizes approximation error to the true posterior.

### Connections to Other Generative Models

| Model | Likelihood | Training | Latent Space | Sample Quality |
|---|---|---|---|---|
| VAE | Explicit (approx.) | Stable | Structured | Blurry |
| GAN | Implicit | Unstable | Less structured | Sharp |
| Normalizing Flow | Exact | Stable | Deterministic map | Good |
| Autoregressive | Exact | Stable | None | Excellent (slow) |

Key hybrids: **VAE-GAN** combines VAE structure with adversarial decoding for sharper outputs; **VQ-VAE + PixelCNN** gains both latent structure and autoregressive generation quality; **Flow-based posteriors** (IAF, MAF) use normalizing flows for $q_\phi(\mathbf{z}|\mathbf{x})$, combining VAE flexibility with flow expressiveness.



## Conclusion

Autoencoders and VAEs have evolved from simple dimensionality reduction tools into sophisticated generative models with deep theoretical foundations. Key themes:

- **Manifold hypothesis**: Data concentrates near low-dimensional manifolds; autoencoders discover and parameterize these structures
- **Probabilistic framework**: VAEs unify generation, inference, and regularization through the ELBO
- **Information-theoretic grounding**: Rate-distortion theory and the information bottleneck formalize the compression-distortion trade-off
- **Architectural flexibility**: The same principles scale from linear PCA to deep hierarchical networks

**Open challenges**: Sample quality (blurriness vs. GANs/diffusion), posterior collapse, unsupervised disentanglement, evaluation metrics, and scalability to very high resolutions.

**Future directions**: Hybrid models (latent diffusion shows promise), improved priors beyond simple Gaussians, discrete VQ-VAE extensions, and theoretical characterization of when VAEs succeed or fail.