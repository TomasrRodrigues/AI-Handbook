# Chapter 7: Autoencoders & VAEs - Learning to Compress and Create

> *"All models are wrong, but some are useful. A generative model is wrong in a way that reveals the structure of reality".*
> - after George Box



## 7.1 The Aha! Moment: From Prediction to Understanding

Every architecture we have studied so far is **discriminative**: given an input, produce a label or prediction. A CNN classifies images. An LSTM predicts the next token. These models learn $p(y | \mathbf{x})$ - the probability of the output *given* the input.

But there is a deeper question: *What does the data itself look like?* What is the structure of the space of natural images? What makes one image "realistic" and another "garbage"? Answering this requires learning the data distribution $p(\mathbf{x})$ itself - not conditioned on any label, but as a model of the world.

This is the domain of **generative models**. A generative model that has truly learned $p(\mathbf{x})$ can:

1. **Generate new examples:** Sample $\mathbf{x} \sim p(\mathbf{x})$ to produce novel, realistic data
2. **Compress:** Encode $\mathbf{x}$ into a compact representation, decode when needed
3. **Detect anomalies:** If $p(\mathbf{x})$ is very low for a new example, that example is unusual
4. **Understand structure:** Discover the underlying "degrees of freedom" - the factors that govern what the data looks like

The **Autoencoder** is the architecturally simplest generative model: force the network to reconstruct its own input after passing through a bottleneck. The bottleneck forces discovery of compact, informative representations. The **Variational Autoencoder (VAE)** extends this with probabilistic rigor, enabling principled generation of new examples.



## 7.2 Why Compression Is Possible: The Manifold Hypothesis

Before diving into architectures, we need to understand *why* compression is possible at all.

A $256 \times 256$ RGB image has $256 \times 256 \times 3 = 196{,}608$ pixels. A random assignment of intensities to these pixels is almost certainly visual noise - no recognizable structure. The probability of a random point in this $196{,}608$-dimensional space being a natural image is essentially zero.

Natural images are highly constrained: smooth regions, consistent lighting, coherent shapes, recognizable objects. These constraints mean natural images occupy a tiny, structured subset of the full pixel space.

This is the **manifold hypothesis**: high-dimensional data concentrates near a low-dimensional **manifold** embedded in the ambient space.

**What is a manifold?** Intuitively, a surface that is locally flat but may curve globally. The surface of a sphere is a 2-dimensional manifold embedded in 3-dimensional space - locally flat (any small patch looks like a flat plane), but globally curved. Natural images form a manifold whose **intrinsic dimensionality** (the number of meaningful degrees of freedom - pose, identity, lighting, etc.) is far smaller than the pixel count.

**Why this enables compression:** If data lies on a $d$-dimensional manifold, we only need $d$ numbers to specify a point - not $D$ (the full ambient dimension). An autoencoder attempts to learn these $d$ numbers (a coordinate system for the manifold) and the mapping back from them to the full $D$-dimensional data.



## 7.3 Classical Autoencoders

### 7.3.1 Architecture and Objective

An autoencoder has two components:
- **Encoder** $E_\phi: \mathbb{R}^D \to \mathbb{R}^d$: maps input $\mathbf{x}$ to latent code $\mathbf{z} = E_\phi(\mathbf{x})$
- **Decoder** $D_\theta: \mathbb{R}^d \to \mathbb{R}^D$: reconstructs $\hat{\mathbf{x}} = D_\theta(\mathbf{z})$ from the code

The subscripts $\phi$ and $\theta$ denote the parameters of the encoder and decoder respectively. The input $\mathbf{x} \in \mathbb{R}^D$ is high-dimensional; the latent code $\mathbf{z} \in \mathbb{R}^d$ is low-dimensional ($d \ll D$).

Training minimizes reconstruction error:

$$\min_{\phi,\, \theta}\; \frac{1}{N}\sum_{i=1}^{N} \ell\!\left(\mathbf{x}^{(i)},\, D_\theta\!\left(E_\phi(\mathbf{x}^{(i)})\right)\right)$$

Reading:
- $\min_{\phi, \theta}$: find encoder parameters $\phi$ and decoder parameters $\theta$ that minimize
- $\frac{1}{N}\sum_{i=1}^{N}$: average over all $N$ training examples
- $\ell(\mathbf{x}^{(i)}, D_\theta(E_\phi(\mathbf{x}^{(i)})))$: the loss between original $\mathbf{x}^{(i)}$ and its reconstruction $D_\theta(E_\phi(\mathbf{x}^{(i)}))$
  - For continuous data: $\ell = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$ (MSE)
  - For binary data (pixel values in $\{0, 1\}$): $\ell = $ binary cross-entropy

**The bottleneck as information pressure:** The key constraint is $d \ll D$. With $d = 32$ and $D = 784$ (MNIST), the compression ratio is $24.5\times$. The encoder must compress all useful information about the input into 32 numbers. Information that cannot fit - noise, irrelevant variation - is discarded. The decoder must reconstruct the input from these 32 numbers alone.

This information pressure forces the autoencoder to discover the most important "axes of variation" in the data - exactly the manifold structure we were looking for.

> TODO: <!-- DIAGRAM: [A "bowtie" or "hourglass" shaped diagram. Left side: a wide block labeled "Input $\mathbf{x} \in \mathbb{R}^{784}$" (MNIST example). Tapering through several layers to a narrow "neck" labeled "Latent Code $\mathbf{z} \in \mathbb{R}^{32}$" with a note "$d \ll D$". Widening back out through mirrored layers to "Reconstruction $\hat{\mathbf{x}} \in \mathbb{R}^{784}$". Arrow below showing "Encoder $E_\phi$" on the left half and "Decoder $D_\theta$" on the right half. At the bottom: loss $\ell(\mathbf{x}, \hat{\mathbf{x}})$ comparing input to reconstruction. Caption: "The bottleneck forces information compression. The 32-dimensional latent code must contain everything needed to reconstruct the 784-dimensional input - the most efficient possible summary".] -->

### 7.3.2 Connection to PCA: The Linear Case

**Principal Component Analysis (PCA)** is the classical linear dimensionality reduction technique. It finds the $d$-dimensional linear subspace (hyperplane through the origin) minimizing expected squared reconstruction error.

A celebrated theorem (Baldi & Hornik, 1989):

**Theorem:** *A linear autoencoder - encoder and decoder both using no activation function (pure linear transformations) - converges to the PCA solution at any global optimum, regardless of initialization.*

Reading "linear autoencoder": encoder is $\mathbf{z} = W_e \mathbf{x}$ (no activation, just matrix multiply); decoder is $\hat{\mathbf{x}} = W_d \mathbf{z}$ (no activation). The composition is $\hat{\mathbf{x}} = W_d W_e \mathbf{x}$ - a rank-$d$ approximation of the identity.

**Significance:** PCA is optimal among all linear methods. Autoencoders generalize PCA to *nonlinear* methods: with ReLU or tanh activations, the encoder and decoder can represent curved coordinate systems on the data manifold, capturing structure that no linear method could represent.

Concretely: points on a circle in $\mathbb{R}^2$ are intrinsically 1-dimensional (parameterized by angle), but a linear autoencoder cannot represent this - the best rank-1 linear approximation just projects onto a line through the origin, losing the circular structure. A nonlinear autoencoder can learn to encode angle and decode back to $(x, y) = (\cos\theta, \sin\theta)$ - perfectly representing the manifold.

### 7.3.3 Regularized Autoencoders

**Sparse autoencoders** add an L1 penalty on the latent activations:

$$\mathcal{L} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{reconstruction}} + \underbrace{\lambda \|\mathbf{z}\|_1}_{\text{sparsity penalty}}$$

Reading the penalty: $\|\mathbf{z}\|_1 = \sum_j |z_j|$ - the L1 norm of the latent code. Most entries of $\mathbf{z}$ are driven to zero; only a few are active for any given input. This forces the model into the overcomplete regime ($d > D$) while preventing trivial identity solutions. Sparse overcomplete autoencoders trained on natural images learn Gabor-filter-like features resembling V1 simple cells - a surprising bridge between deep learning and neuroscience.

**Denoising autoencoders** corrupt the input before encoding, then train to reconstruct the clean original:

$$\min_{\phi, \theta}\; \mathbb{E}_{\mathbf{x}}\,\mathbb{E}_{\tilde{\mathbf{x}}|\mathbf{x}}\!\left[\|\mathbf{x} - D_\theta(E_\phi(\tilde{\mathbf{x}}))\|^2\right]$$

Reading: the outer expectation is over real data $\mathbf{x}$; the inner expectation is over corrupted versions $\tilde{\mathbf{x}}$ (e.g., add Gaussian noise, randomly mask pixels). The network encodes the corrupted version and must decode the clean original.

**Why this helps:** The model cannot memorize individual examples (corruption changes them at every step). It must learn the underlying structure that allows a corrupted observation to be restored - structure that generalizes. Denoising autoencoders have a deep theoretical connection to score matching: the learned encoder approximates $\nabla_\mathbf{x} \log p(\mathbf{x})$, the gradient of the log data density - a key quantity in modern diffusion models.



## 7.4 The Generation Problem: Why Classical Autoencoders Cannot Generate

Classical autoencoders excel at compression and reconstruction. But they have a fundamental limitation for *generative* modeling.

After training, the encoder maps each training example $\mathbf{x}^{(i)}$ to a specific code $\mathbf{z}^{(i)}$. These codes form a cloud of points in the $d$-dimensional latent space. The shape of this cloud is arbitrary - it could be any irregular cluster of points, with "holes" (regions between training examples) containing no useful codes.

**The generation problem:** To generate new examples, you sample a code $\mathbf{z}$ from the latent space and decode it. But if you sample uniformly or from a Gaussian, you will almost certainly land in a "hole" - a region where the decoder was never trained and will produce garbage.

Classical autoencoders have **no mechanism** to structure the latent space in a way that enables sampling. They are excellent compressors but cannot generate.

The Variational Autoencoder solves this with one elegant constraint: force the latent codes to be distributed like a standard Gaussian. Any point sampled from $\mathcal{N}(\mathbf{0}, I)$ is then a valid latent code, and the decoder produces something sensible.



## 7.5 Variational Autoencoders: The Probabilistic Framework

### 7.5.1 The Generative Story

The VAE posits a specific probabilistic story for how data is generated:

1. Sample a latent code from the **prior**: $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$
2. Generate data from the code: $\mathbf{x} | \mathbf{z} \sim p_\theta(\mathbf{x} | \mathbf{z})$

Reading:
- $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$: the prior over latent codes - a standard Gaussian. $\mathbf{0}$ is the zero mean vector; $I$ is the identity covariance (all dimensions independent, unit variance)
- $p_\theta(\mathbf{x} | \mathbf{z})$: the decoder - given a latent code $\mathbf{z}$, produces a distribution over $\mathbf{x}$ (e.g., Gaussian with mean given by the decoder network output)

The marginal likelihood of an observed data point:

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} | \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$$

Reading: integrate over all possible latent codes $\mathbf{z}$ - the probability of observing $\mathbf{x}$ under any code, weighted by the prior probability of that code. The $\int \cdots d\mathbf{z}$ denotes integration over the entire $d$-dimensional latent space.

**The problem:** This integral is **intractable** for any expressive decoder $p_\theta(\mathbf{x}|\mathbf{z})$. We cannot enumerate all possible $\mathbf{z}$ values. We cannot maximize the log-likelihood $\log p_\theta(\mathbf{x})$ directly.

### 7.5.2 Variational Inference: Approximating the Intractable

The solution: instead of computing the true posterior $p_\theta(\mathbf{z}|\mathbf{x})$ ("what latent codes explain this data point?"), introduce an **approximate posterior** (also called the **recognition model** or **encoder**):

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}\!\left(\boldsymbol{\mu}_\phi(\mathbf{x}),\; \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}))\right)$$

Reading:
- $q_\phi(\mathbf{z}|\mathbf{x})$: the approximate posterior - a distribution over latent codes given input $\mathbf{x}$, parameterized by $\phi$ (the encoder's weights)
- $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$: a Gaussian distribution with mean vector $\boldsymbol{\mu}$ and diagonal covariance matrix (all dimensions independent, each with its own variance $\sigma_j^2$)
- $\boldsymbol{\mu}_\phi(\mathbf{x})$: a vector of $d$ mean values - the encoder network outputs these, one per latent dimension
- $\boldsymbol{\sigma}^2_\phi(\mathbf{x})$: a vector of $d$ variance values - the encoder also outputs these
- Together: for each input $\mathbf{x}$, the encoder outputs not a single code but a *distribution* of codes - capturing the uncertainty about which latent representation best explains the input

**The key difference from a classical autoencoder:** The classical encoder outputs a single point $\mathbf{z} = E_\phi(\mathbf{x})$. The VAE encoder outputs a Gaussian distribution: "The latent code is probably around $\boldsymbol{\mu}_\phi(\mathbf{x})$, with uncertainty $\boldsymbol{\sigma}^2_\phi(\mathbf{x})$".

### 7.5.3 The ELBO: A Tractable Training Objective

We derive the training objective using a mathematical identity. Starting from the log-likelihood of a single example $\mathbf{x}$, for any distribution $q_\phi(\mathbf{z}|\mathbf{x})$:

$$\log p_\theta(\mathbf{x}) = \underbrace{\mathcal{L}_{\text{ELBO}}(\phi, \theta; \mathbf{x})}_{\text{Evidence Lower BOund}} + \underbrace{\text{KL}\!\left(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p_\theta(\mathbf{z}|\mathbf{x})\right)}_{\geq 0}$$

Reading:
- The left side: $\log p_\theta(\mathbf{x})$ is the log-likelihood we want to maximize - the log-probability of the observed data under our model
- The right side: the ELBO plus a KL divergence term
- $\text{KL}(q \| p) \geq 0$: KL divergence is always non-negative (it measures how different two distributions are; zero only when they are identical)

**Consequence:** Since the KL term $\geq 0$:

$$\log p_\theta(\mathbf{x}) \geq \mathcal{L}_{\text{ELBO}}(\phi, \theta; \mathbf{x})$$

The ELBO is a **lower bound** on the log-likelihood. Maximizing the ELBO:
- Pushes the log-likelihood up (good - we want high likelihood)
- Simultaneously minimizes the KL gap (good - makes $q_\phi$ a better approximation to the true posterior)

The ELBO is tractable - we can compute and optimize it. The log-likelihood is intractable - we cannot. The ELBO is the best tractable surrogate.

**Expanding the ELBO:** Using the rules of probability and logarithms:

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right]}_{\text{Reconstruction Term}} - \underbrace{\text{KL}\!\left(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\right)}_{\text{Regularization Term}}$$

Reading:
- $\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\cdot]$: expectation under the approximate posterior - sample $\mathbf{z}$ from the encoder's distribution, evaluate the bracketed quantity, and average
- $\log p_\theta(\mathbf{x}|\mathbf{z})$: the log-probability of the input $\mathbf{x}$ given latent code $\mathbf{z}$ - measured by the decoder. Higher is better; we want the decoder to assign high probability to the true input
- **Reconstruction term:** Maximizing $\mathbb{E}[\log p_\theta(\mathbf{x}|\mathbf{z})]$ is exactly minimizing reconstruction error - just like a classical autoencoder
- $\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$: the KL divergence between the encoder's distribution and the prior $\mathcal{N}(\mathbf{0}, I)$. Minimizing this makes the encoder's codes look like standard Gaussian draws
- **Regularization term (negative sign):** We *subtract* the KL term (equivalently, we maximize the ELBO by minimizing the KL divergence)

**The tension in the objective:**
- The reconstruction term wants the encoder to produce highly informative, specific codes ($\boldsymbol{\mu}$ far from zero, $\boldsymbol{\sigma}$ small) - codes that let the decoder perfectly reconstruct $\mathbf{x}$
- The KL term wants the codes to look like draws from $\mathcal{N}(\mathbf{0}, I)$ - pushing $\boldsymbol{\mu}$ toward $\mathbf{0}$ and $\boldsymbol{\sigma}$ toward $\mathbf{1}$
- Training finds the balance: codes that are informative enough to reconstruct well, but structured enough to look like the prior

This balance is what gives the VAE its generative power: at equilibrium, a sample from $\mathcal{N}(\mathbf{0}, I)$ is a plausible latent code, and the decoder produces something that looks like real data.

### 7.5.4 The KL Term: Closed Form Solution

For diagonal Gaussian approximate posterior $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and standard Gaussian prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$, the KL divergence has an exact, closed-form solution:

$$\text{KL}\!\left(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\right) = \frac{1}{2}\sum_{j=1}^{d}\!\left(\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2\right)$$

Reading:
- $\sum_{j=1}^{d}$: sum over all $d$ latent dimensions
- For each dimension $j$: the term is $\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2$
- When $\mu_j = 0$ and $\sigma_j = 1$ (exactly the prior): the term is $1 + 0 - 1 - 0 = 0$ - zero KL divergence, the encoder matches the prior perfectly
- When $\mu_j \neq 0$ or $\sigma_j \neq 1$: the term is positive - the encoder deviates from the prior, incurring regularization cost

**Gradient of KL with respect to encoder parameters:** $\frac{\partial \text{KL}}{\partial \mu_j} = \mu_j$ (push $\mu_j$ toward zero); $\frac{\partial \text{KL}}{\partial \sigma_j^2} = \frac{1}{2}(1 - 1/\sigma_j^2)$ (push $\sigma_j$ toward 1). These gradients are simple and always computable - no approximation needed.

### 7.5.5 The Reparameterization Trick: Making Sampling Differentiable

Training the VAE requires gradients of the ELBO with respect to encoder parameters $\phi$. The ELBO reconstruction term involves an expectation over $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$:

$$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right]$$

To compute this expectation, we must **sample** $\mathbf{z}$ from the encoder's distribution. But sampling is a stochastic operation - you cannot differentiate through a random sample with respect to the distribution's parameters. The gradient would be zero everywhere.

**The reparameterization trick** circumvents this by moving the randomness outside the gradient computation:

**Before reparameterization:** $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$

Sample directly from the encoder's Gaussian - the parameters $\phi$ are "inside" the sample.

**After reparameterization:**

$$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I), \qquad \mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}$$

Reading:
- $\boldsymbol{\varepsilon}$: "noise vector" - sampled from a *fixed* standard Gaussian, containing no learnable parameters
- $\boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}$: compute $\mathbf{z}$ as a *deterministic function* of the encoder outputs and the noise
  - $\boldsymbol{\mu}_\phi(\mathbf{x})$: the encoder's mean output (differentiable with respect to $\phi$)
  - $\boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}$: element-wise multiply the encoder's standard deviation output by the noise (differentiable with respect to $\phi$)
- Result: $\mathbf{z}$ is still a sample from $\mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$ - statistically equivalent to before
- But now: $\mathbf{z}$ is a differentiable function of $\phi$ (through $\boldsymbol{\mu}_\phi$ and $\boldsymbol{\sigma}_\phi$). Gradients can flow through $\mathbf{z}$ back to the encoder parameters.

**Why it works:** The randomness (the noise $\boldsymbol{\varepsilon}$) is in a fixed, parameter-free distribution. The parameters $\phi$ control only the *location and scale* of the Gaussian - both differentiable operations. We have "separated" the source of randomness from the source of trainable computation.

> TODO: <!-- DIAGRAM: [Two versions side by side. Left: "Before Reparameterization" - box labeled "Encoder $\phi$" outputs $\boldsymbol{\mu}$, $\boldsymbol{\sigma}$. A "Sample" node takes $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ and produces $\mathbf{z}$. A red X over the arrow from Sample to Encoder says "No gradient through sampling". Right: "After Reparameterization" - box labeled "Encoder $\phi$" outputs $\boldsymbol{\mu}$, $\boldsymbol{\sigma}$. A separate "Sample $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},I)$" node (with no parameters). A deterministic "Transform: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}$" box combines them. A green checkmark over the gradient path. Caption: "Reparameterization moves randomness outside the computation graph. Gradients can now flow through the deterministic transformation back to the encoder parameters".] -->

### 7.5.6 The Complete VAE Training Objective

For a dataset of $N$ examples, with $L$ Monte Carlo samples per example (in practice $L=1$ works fine because gradients are averaged over the minibatch):

$$\mathcal{L}(\phi, \theta) = \frac{1}{N}\sum_{i=1}^{N} \left[\frac{1}{L}\sum_{\ell=1}^{L} \log p_\theta\!\left(\mathbf{x}^{(i)} \middle| \boldsymbol{\mu}_\phi(\mathbf{x}^{(i)}) + \boldsymbol{\sigma}_\phi(\mathbf{x}^{(i)}) \odot \boldsymbol{\varepsilon}^{(\ell)}\right) - \text{KL}\!\left(q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \| p(\mathbf{z})\right)\right]$$

Reading the key structure:
- The outer sum: average over $N$ training examples
- The reconstruction term: for each sampled noise $\boldsymbol{\varepsilon}^{(\ell)}$, compute a code $\mathbf{z}^{(\ell)} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}^{(\ell)}$, decode it, measure the log-likelihood. Average over $L$ samples for a low-variance estimate
- The KL term: computed analytically (closed form from Section 7.5.4), subtracted

Both terms have gradients computable by backpropagation: the reconstruction term through the reparameterization trick; the KL term analytically.



## 7.6 Understanding the VAE Latent Space

Training produces a latent space with three useful properties:

**1. Continuity:** Nearby codes decode to similar outputs. You can smoothly interpolate between two images by linearly interpolating their latent codes - the intermediate codes decode to sensible intermediate images. The KL regularization ensures "holes" in the latent space (regions not covered by training codes) decode to something reasonable, because these regions overlap with the standard Gaussian prior.

**2. Completeness:** Sampling from $\mathcal{N}(\mathbf{0}, I)$ always produces a plausible code. The KL term forces all training examples' codes to cluster around the origin with spread roughly $\pm 1$ - covering the standard Gaussian prior.

**3. Disentanglement (approximate):** Different latent dimensions encode different factors of variation. One dimension may control pose, another identity, another lighting. Moving along a single latent dimension changes only the corresponding factor. The independence assumption in the diagonal Gaussian posterior promotes this: each dimension is pushed to vary independently.

**The $\beta$-VAE** amplifies disentanglement by weighting the KL term more heavily:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}\!\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right] - \beta\, \text{KL}\!\left(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\right), \quad \beta > 1$$

Reading: $\beta > 1$ applies a larger penalty for each unit of KL divergence, forcing more aggressive compression. This stronger pressure pushes dimensions to be more independent (it becomes costly to use multiple dimensions to encode the same information). Empirically, $\beta$-VAEs learn more interpretable representations at the cost of slightly worse reconstruction quality.



## 7.7 VAE Variants

### 7.7.1 Conditional VAE (CVAE)

The standard VAE generates samples unconditionally. To control *what* is generated - produce a specific digit, generate a molecule with certain properties - condition both encoder and decoder on auxiliary information $\mathbf{c}$:

$$q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c}), \quad p_\theta(\mathbf{x}|\mathbf{z}, \mathbf{c})$$

The conditioning variable $\mathbf{c}$ can be a class label (concatenated to the input), a text description (encoded by a separate encoder), or any structured annotation. At generation time: sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$, specify $\mathbf{c}$ (the desired attributes), and decode. The decoder has learned "given code $\mathbf{z}$ and condition $\mathbf{c}$, generate an image with attributes $\mathbf{c}$".

### 7.7.2 Vector Quantized VAE (VQ-VAE)

Standard VAEs use continuous Gaussian latent spaces. For discrete data (language tokens, symbolic structures), or when the decoder becomes too powerful (causing **posterior collapse** - the decoder ignores the latent code entirely), a discrete latent space is preferable.

**VQ-VAE** (van den Oord et al., 2017) replaces the Gaussian encoder with quantization to a learned **codebook** $\{e_k\}_{k=1}^K$:

1. The encoder produces a continuous embedding $E_\phi(\mathbf{x}) \in \mathbb{R}^d$
2. The embedding is quantized to the nearest codebook entry:
   $$\mathbf{z}_q = e_{k^*}, \quad k^* = \arg\min_k \|E_\phi(\mathbf{x}) - e_k\|_2^2$$
   Reading: find the index $k^*$ of the codebook entry $e_k$ closest to the encoder output in Euclidean distance. Use that codebook entry as the latent code.
3. The decoder receives $\mathbf{z}_q$ and reconstructs $\hat{\mathbf{x}}$

The quantization step ($\arg\min$) is not differentiable. The **straight-through estimator** handles this: in the backward pass, copy gradients from the decoder input to the encoder output, bypassing the argmin. The codebook entries $e_k$ are updated by moving them toward the encoder outputs that were assigned to them (exponentially moving average).

VQ-VAE eliminates posterior collapse because the discrete codebook has finite capacity - the decoder cannot ignore the code (there are only $K$ possible codes, and each carries distinct information). Combined with autoregressive priors trained over the codebook indices, VQ-VAE-2 generates images at $1024 \times 1024$ resolution with remarkable quality.



## 7.8 The Connection to Diffusion Models

Denoising diffusion probabilistic models - which power DALL-E 2, Stable Diffusion, and Midjourney - are deeply connected to the autoencoder framework.

A diffusion model defines a **forward process** that gradually adds noise over $T$ steps:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{1-\beta_t}\, \mathbf{x}_{t-1},\; \beta_t I\right)$$

Reading:
- $\mathbf{x}_t$: the data at noise level $t$ (at $t=0$: clean data; at $t=T$: pure noise)
- $\mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}, \Sigma)$: a Gaussian with mean $\boldsymbol{\mu}$ and covariance $\Sigma$, evaluated at point $\mathbf{x}_t$
- $\sqrt{1-\beta_t}\, \mathbf{x}_{t-1}$: a slightly scaled-down version of the previous step (preserves signal)
- $\beta_t I$: added Gaussian noise with variance $\beta_t$ in each dimension
- Together: each step slightly corrupts the data by mixing in Gaussian noise

After $T$ steps: $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, I)$ - pure noise. The model learns the **reverse process**: given noisy $\mathbf{x}_t$, predict the slightly denoised $\mathbf{x}_{t-1}$. Starting from pure noise and iteratively denoising produces new samples.

**The VAE connection:** A diffusion model is a hierarchical VAE with $T$ levels. Each level is an approximate Gaussian encoder ($q(\mathbf{x}_t | \mathbf{x}_{t-1})$ adds noise) and Gaussian decoder ($p(\mathbf{x}_{t-1} | \mathbf{x}_t)$ removes noise). The ELBO for the diffusion model decomposes into $T$ reconstruction terms - one per noise level - weighted by the signal-to-noise ratio at each level.

**The denoising autoencoder connection:** Denoising autoencoders (Section 7.3.3) are one-step diffusion models. The learned denoising function approximates the score function $\nabla_\mathbf{x} \log p(\mathbf{x})$ - the gradient of the log data density. Diffusion models can be seen as training a denoiser at every noise level simultaneously, then chaining these denoisers together to generate samples.



## 7.9 Applications

**Anomaly detection:** Train on normal data; flag inputs with high reconstruction error as anomalies. No anomaly labels needed during training. Applications: industrial defect detection, network intrusion detection, medical screening.

**Representation learning:** Encoder features from a pretrained autoencoder generalize across tasks. Fine-tune on small labeled datasets using the encoder's representations as features - often outperforms training from scratch when labels are scarce.

**Drug discovery:** VAEs trained on molecular representations (SMILES strings, molecular graphs) learn smooth latent spaces over chemical structures. Optimization in latent space - finding codes that decode to molecules with desired properties - is far more efficient than search in the discrete molecular space.

**Image editing:** Identify the latent direction that encodes a specific attribute (e.g., "smiling" in face images). Add or subtract this direction from any input image's code to add or remove the attribute in the reconstruction.



## 7.10 Summary

Autoencoders introduced a new paradigm: learning the structure of the data distribution by forcing networks to compress and reconstruct. The bottleneck information pressure discovers the manifold structure - the most efficient coordinate system for the data.

The VAE adds probabilistic rigor: by treating the latent space as a distribution rather than a single point, and by regularizing this distribution toward a standard Gaussian prior, the VAE enables principled generation and smooth interpolation. The ELBO objective balances reconstruction quality against latent space regularity. The reparameterization trick makes the stochastic objective differentiable.

The deeper theoretical story: the ELBO is rate-distortion theory instantiated in a neural network. Reconstruction quality is the distortion; KL divergence is the rate (bits used to represent each example). The balance between them, controlled by $\beta$ in the $\beta$-VAE, reflects the fundamental tradeoff between compression and fidelity.



*Autoencoders learn to compress; RNNs learn to process sequences; CNNs learn to see. Chapter 8 brings us to the current frontier: architectures that process sequences as efficiently as RNNs but with the long-range memory of Transformers - State Space Models and Mamba.*

---
*Continue to **[Chapter 8: State Space Models & Mamba — The Sequence Revolution](/DeepLearning/08_Mamba_and_SSMs.md)***