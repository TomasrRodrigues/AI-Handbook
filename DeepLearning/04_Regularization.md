# Chapter 4: Regularization - The Science of Generalization

> *"The goal is not to fit the data. The goal is to fit the truth. Regularization is our best approximation of what the truth looks like, when we can only see a sample of it"*



## 4.1 The Aha! Moment: The Generalization Paradox

Here is a fact that should stop you in your tracks.

A neural network with 100 million parameters, trained on 50,000 images, should fail catastrophically. Classical statistics - which governed machine learning for decades - is unequivocal: a model with 2,000 times more parameters than data points will **overfit**. It will memorize the training examples instead of learning the underlying pattern. It will achieve near-zero training error and near-chance test accuracy.

Yet these same massively overparameterized networks work spectacularly well. ResNets, GPT models, BERT - all have vastly more parameters than training examples, yet generalize to new data with remarkable precision.

This is the generalization paradox of deep learning. Understanding it is not a theoretical exercise - it is the key to designing models that do not just train well, but *work* when deployed.

The classical answer was: "Regularization = add a penalty to the loss to control model complexity". This is true but incomplete. The modern answer is: "Generalization emerges from the interaction of the loss function, the optimizer, the architecture, the data, and explicit regularization - each one encoding assumptions about what good solutions look like".

This chapter covers all of them.



## 4.2 The Formal Problem: Training Error vs. True Error

### 4.2.1 Two Kinds of Risk

When we train a network, we minimize error on the training set. But we care about error on data we have never seen - the true "test" distribution.

**Empirical risk** (what we actually minimize):

$$\hat{R}_S(h) = \frac{1}{N} \sum_{i=1}^{N} \ell\!\left(h(\mathbf{x}^{(i)}), y^{(i)}\right)$$

Reading:
- $\hat{R}_S(h)$: the empirical risk of hypothesis $h$ on training set $S$. The hat ($\hat{}$) denotes "estimated" (from finite data). The subscript $S$ denotes "computed on sample $S$"
- $\frac{1}{N}$: average over $N$ training examples
- $\ell(h(\mathbf{x}^{(i)}), y^{(i)})$: the loss for a single example - how wrong the model's prediction $h(\mathbf{x}^{(i)})$ is for the true label $y^{(i)}$

**Population risk** (what we actually care about):

$$R(h) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}\!\left[\ell\!\left(h(\mathbf{x}), y\right)\right]$$

Reading:
- $R(h)$: the true (population) risk of hypothesis $h$
- $\mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}[\cdot]$: expected value over all possible $(\mathbf{x}, y)$ pairs drawn from the true distribution $\mathcal{D}$. The subscript says "draw from the true distribution"
- The loss inside: same formula, but now averaged over the infinite population, not just our finite training set

**The generalization gap:**

$$\text{Gap} = R(h) - \hat{R}_S(h) \geq 0$$

This is always non-negative (the training error is optimistically biased - we chose the model to minimize it). Regularization's goal is to bound and minimize this gap.

### 4.2.2 Why the Gap Exists

Suppose you take a random sample of 1,000 voters and survey their preferences. You get an estimate of the true population's preference. How good is this estimate? The more complex the question (the more we let the model look for patterns), the more opportunity there is to find "patterns" in the random noise of the sample that do not generalize to the population. A very simple question ("Do they prefer A or B overall?") generalizes well. A very complex question ("Given all 500 questions, predict each voter's answers to all of them?") can be "fit" to the sample perfectly while predicting the population terribly.

Neural network training is the same: a very flexible model can fit training data perfectly (finding spurious patterns in the noise) while failing on new data. Regularization constrains the model's flexibility - forcing it to find patterns that are likely to hold in the population.



## 4.3 Bias and Variance: The Fundamental Tension

Every prediction error can be decomposed into three parts. Understanding this decomposition makes every regularization technique make sense.

For a regression problem with true relationship $y = f^*(\mathbf{x}) + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ is irreducible noise:

$$\mathbb{E}_S\!\left[(f_S(\mathbf{x}) - y)^2\right] = \underbrace{\left(\mathbb{E}_S[f_S(\mathbf{x})] - f^*(\mathbf{x})\right)^2}_{\text{Bias}^2} + \underbrace{\text{Var}_S(f_S(\mathbf{x}))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

Reading:
- $\mathbb{E}_S[\cdot]$: expectation over all possible training sets $S$ of the same size drawn from $\mathcal{D}$
- $f_S(\mathbf{x})$: the model trained on training set $S$, evaluated at test point $\mathbf{x}$
- $\mathbb{E}_S[f_S(\mathbf{x})]$: the average prediction across all possible training sets (the "average model")
- $f^*(\mathbf{x})$: the true function value at $\mathbf{x}$

**Bias$^2$:** How far is the average prediction from the truth? High bias means the model is systematically wrong - perhaps it always predicts too high, or cannot represent curved relationships because it is too simple. A linear model fit to quadratic data has high bias.

**Variance:** How much does the prediction vary across different training sets? High variance means small changes in the training data produce very different models. A high-degree polynomial fit to noisy data has high variance - fit it to a slightly different 100 points and get a wildly different curve.

**Irreducible noise $\sigma^2$:** The noise inherent in the data. No model can predict $y$ better than the noise level - this term is fixed.

**Regularization's role:** By adding a penalty that favors simpler solutions, regularization *increases bias* (forces the model away from the training-set-specific optimum) to *decrease variance* (makes the model more stable across different training sets). The net effect is lower total error when variance is the dominant problem - which is almost always true for overparameterized models.

> TODO: <!-- DIAGRAM: [Four target diagrams labeled "High Bias, Low Variance" (shots clustered away from center), "Low Bias, High Variance" (shots spread around center), "High Bias, High Variance" (shots spread and off-center), and "Low Bias, Low Variance" (shots clustered at center). Below: a U-shaped curve of "Test Error" vs "Model Complexity" with "Bias$^2$" decreasing and "Variance" increasing, crossing at the optimal complexity. Caption: "Bias is systematic error; Variance is sensitivity to training data. Regularization adds a controlled amount of bias to achieve a large reduction in variance".] -->



## 4.4 L2 Regularization: Penalizing Large Weights

**L2 regularization** (also called **weight decay**) adds a penalty to the loss proportional to the squared magnitude of all weights:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2}\sum_j w_j^2 = \mathcal{L} + \frac{\lambda}{2}\|W\|_2^2$$

Reading:
- $\mathcal{L}$: the original loss (cross-entropy or MSE)
- $\frac{\lambda}{2}$: the regularization strength. The $\frac{1}{2}$ is a convenience factor that cancels with the derivative of $w^2$ to give a clean gradient. $\lambda$ (lambda) is a hyperparameter you choose (typically $10^{-4}$ to $10^{-2}$)
- $\sum_j w_j^2$: sum of squared weights over all weights $j$ in the network
- $\|W\|_2^2$: shorthand - the squared L2 norm of all weights (L2 norm is the Euclidean length)

**The gradient of the regularized loss with respect to weight $w_j$:**

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial w_j} + \lambda w_j$$

Reading: the regularized gradient is the original gradient plus $\lambda w_j$ - a term proportional to the weight's current value.

**The regularized SGD update:**

$$w_j \leftarrow w_j - \eta\left(\frac{\partial \mathcal{L}}{\partial w_j} + \lambda w_j\right) = (1 - \eta\lambda)\, w_j - \eta\, \frac{\partial \mathcal{L}}{\partial w_j}$$

Reading: each step, the weight is first *multiplied* by $(1 - \eta\lambda)$ - a factor slightly less than 1 (e.g., $1 - 0.001 \times 0.01 = 0.99999$) - then the gradient step is applied. This multiplicative shrinkage is "weight decay" - at every step, all weights decay toward zero by a small factor.

**What this achieves:**
- Weights that are driven to large values by the data gradient are "resisted" by the penalty. The final weight is smaller than it would be without regularization
- No single weight can become arbitrarily large - the penalty cost grows with $w_j^2$, eventually overpowering the data gradient for very large weights
- The model is forced toward solutions where the "work" is distributed across many weights, rather than concentrated in a few very large ones - typically more robust solutions

**Bayesian interpretation:** L2 regularization is equivalent to placing a Gaussian prior on weights: $p(W) \propto \exp(-\lambda\|W\|^2/2)$. Maximizing the posterior $p(W|\text{data}) \propto p(\text{data}|W) p(W)$ is equivalent to minimizing $\mathcal{L} + \frac{\lambda}{2}\|W\|^2$. The regularizer encodes the belief that weights are probably small (close to zero) before seeing data.



## 4.5 L1 Regularization: Encouraging Sparsity

**L1 regularization** (also called **Lasso**) penalizes the *absolute value* rather than the square:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda\sum_j |w_j| = \mathcal{L} + \lambda\|W\|_1$$

Reading:
- $|w_j|$: the absolute value of weight $j$ - always positive, equal to $w_j$ if positive, $-w_j$ if negative
- $\|W\|_1$: the L1 norm - sum of absolute values of all weights

**The gradient (subgradient, since $|w|$ is not differentiable at 0):**

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial w_j} + \lambda\, \text{sign}(w_j)$$

Reading:
- $\text{sign}(w_j)$: the sign function - equals $+1$ if $w_j > 0$, $-1$ if $w_j < 0$, and $0$ if $w_j = 0$

So the regularization gradient is simply $\pm\lambda$ - a constant push toward zero, regardless of the weight's magnitude.

**The key property: sparsity.** Unlike L2 (which shrinks weights proportionally - large weights shrink more, small weights shrink less, but all stay nonzero), L1 applies a *constant* push toward zero. For small weights where the data gradient is small, the $\pm\lambda$ push dominates and drives the weight to exactly zero.

The result: L1 regularization produces **sparse solutions** - most weights become exactly zero. The model effectively performs automatic feature selection: only weights connected to genuinely informative inputs survive.

**Geometric intuition:** The L1 penalty constrains weights to lie within a "diamond" shape in weight space (a hypercube rotated 45°). The loss function's contours (ellipses) most likely first touch this diamond at a corner - where many weights are exactly zero. L2 constrains weights to a sphere; sphere surfaces have no corners, so the optimum is almost never exactly zero.

> TODO: <!-- DIAGRAM: [Two-dimensional version showing the "diamond and circle" picture. Left panel: L1. The weight space has axes $w_1$, $w_2$. A rotated square (diamond) centered at the origin represents the L1 constraint region. Elliptical contours of the loss function intersect the diamond at one of its corners, where $w_1 = 0$. Right panel: L2. A circle (L2 constraint) replaces the diamond. The loss contours touch the circle at a smooth boundary point where both $w_1 \neq 0$ and $w_2 \neq 0$. Caption: "L1's corners are on the coordinate axes - the loss most likely first touches the constraint at a corner, producing a zero weight. L2's smooth circle has no corners; sparse solutions do not arise".] -->



## 4.6 Dropout: Ensemble Learning Through Noise

**Dropout** (Srivastava et al., 2014) is the most famous regularization technique in deep learning. Its premise is almost shockingly simple: during training, randomly set some fraction of neuron activations to zero at each forward pass.

Specifically, for each layer, each neuron is independently set to zero with probability $p$ (the "dropout rate") and kept with probability $1-p$. The surviving neurons are scaled by $\frac{1}{1-p}$ to maintain the expected total activation magnitude (this is **inverted dropout** - the standard implementation):

**During training, for each forward pass:**
$$a'_j = \begin{cases} \frac{a_j}{1-p} & \text{with probability } 1-p \text{ (keep)} \\ 0 & \text{with probability } p \text{ (drop)} \end{cases}$$

Reading:
- $a_j$: the activation of neuron $j$ (computed normally)
- $a'_j$: the activation after dropout
- With probability $1-p$: keep the neuron, but scale by $\frac{1}{1-p}$ to compensate for the missing neurons
- With probability $p$: set to zero - this neuron is "turned off" for this forward pass

**During inference (test time):** Use all neurons normally, without any scaling. Because the training-time scaling already compensated, the expected value is the same.

### 4.6.1 Why Dropout Works: The Ensemble View

A network with $n$ neurons has $2^n$ possible dropout masks (each neuron either present or absent). At each training step, a different random subset of neurons is active. Dropout trains exponentially many different "thinned" networks simultaneously, all sharing the same weights.

At test time, using the full network (with all neurons active) approximates averaging the predictions of all $2^n$ thinned networks. Ensemble averaging reduces variance without increasing bias - the same principle behind Random Forests but at exponential scale.

The shared weights mean that this exponential ensemble is trained at essentially the same cost as a single network. This is the computational miracle that makes dropout practical.

### 4.6.2 Why Dropout Works: The Co-Adaptation View

When neurons know they can always rely on specific other neurons, they develop **co-adaptation**: neurons form tight partnerships where each is only useful in the presence of its partners. These partnerships are brittle - if any partner fails or the input distribution shifts slightly, the whole group fails.

Dropout randomly disables neurons, breaking these partnerships. Each neuron must learn to be independently useful - to develop features that work regardless of which other neurons happen to be active. This forces more robust, redundant representations.

### 4.6.3 The Bayesian Interpretation

Gal & Ghahramani (2016) showed that dropout at test time (running multiple forward passes with different random masks and averaging) is equivalent to **approximate Bayesian inference** over the network weights. The variance between multiple dropout-sampled predictions measures the network's epistemic uncertainty - how uncertain it is due to limited data.

This enables **Monte Carlo Dropout**: run $T$ forward passes with dropout enabled, compute the mean prediction (the point estimate) and the standard deviation (the uncertainty estimate). For a medical diagnosis system, knowing that the network's predictions have high variance is as important as knowing the prediction itself.

### 4.6.4 Dropout Variants

**Spatial Dropout:** For convolutional layers, adjacent pixels in the same feature map are highly correlated. Dropping individual pixels provides little regularization - the neighboring pixels provide essentially the same information. Spatial dropout drops *entire feature maps* for each example in the batch, providing genuine regularization by forcing the network to be redundant across channels.

**Stochastic Depth:** For very deep residual networks, randomly bypass entire layers with probability $p_\ell$ (increasing with depth):

$$\mathbf{y} = \begin{cases} \mathcal{F}(\mathbf{x}) + \mathbf{x} & \text{with probability } 1 - p_\ell \\ \mathbf{x} & \text{with probability } p_\ell \end{cases}$$

During training, the network is effectively a shorter, random depth - avoiding the degradation that can occur even in residual networks at extreme depth. At inference, all layers are used.

> TODO: <!-- DIAGRAM: [Two network drawings side by side. Left: "Standard Network" - all nodes visible, all edges drawn, full connectivity between adjacent layers. Right: "Dropout Network (one training step)" - some nodes have X marks through them, edges to/from X'd nodes are dashed or absent. Caption: "At each training step, a different random subset of neurons is active. Each forward pass trains a different 'thinned' subnetwork. At test time, the full network approximates the average of all these subnetworks".] -->



## 4.7 Batch Normalization: Stabilizing the Signal

**Batch Normalization** (Ioffe & Szegedy, 2015) was introduced to solve the problem of **internal covariate shift** - the fact that as the network trains, the distribution of each layer's inputs keeps changing, making it hard for subsequent layers to learn.

For each mini-batch $\mathcal{B} = \{z_1, z_2, \ldots, z_B\}$ of pre-activations at a given layer and dimension:

**Step 1 - Compute batch statistics:**
$$\mu_\mathcal{B} = \frac{1}{B}\sum_{i=1}^B z_i$$
$$\sigma^2_\mathcal{B} = \frac{1}{B}\sum_{i=1}^B (z_i - \mu_\mathcal{B})^2$$

Reading:
- $\mu_\mathcal{B}$: the batch mean - average pre-activation value across the $B$ examples in the batch
- $\sigma^2_\mathcal{B}$: the batch variance - average squared deviation from the mean. Measures how spread out the values are

**Step 2 - Normalize:**
$$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \varepsilon}}$$

Reading:
- $z_i - \mu_\mathcal{B}$: subtract the mean - centers the values around zero
- $\sqrt{\sigma^2_\mathcal{B} + \varepsilon}$: standard deviation of the batch plus a tiny $\varepsilon$ (typically $10^{-5}$) to prevent division by zero
- The division: scales the centered values to have unit variance ($\approx 1$)
- $\hat{z}_i$: the normalized pre-activation - guaranteed to have approximately mean 0 and variance 1 across the batch

**Step 3 - Rescale and shift with learnable parameters:**
$$y_i = \gamma\, \hat{z}_i + \beta$$

Reading:
- $\gamma$ (gamma): a learnable scale parameter - initialized to 1
- $\beta$ (beta): a learnable shift parameter - initialized to 0
- $y_i$: the final BatchNorm output, ready for the activation function

**Why do we add $\gamma$ and $\beta$ back?** Because normalizing to mean-0, variance-1 might be wrong for this layer. What if the best representation for this layer has mean 2 and variance 5? With $\gamma = 5$ and $\beta = 2$, the network can undo the normalization. The parameters $\gamma$ and $\beta$ allow the network to learn the optimal distribution for each layer, while the normalization ensures it starts from a well-behaved state.

Without $\gamma$ and $\beta$: a sigmoid activation after BatchNorm would always operate near the linear region (since $\hat{z} \approx 0$), losing the sigmoid's nonlinearity. With them, the network can shift the operating point to wherever the sigmoid is useful.

### 4.7.1 Training vs. Inference

During training: $\mu_\mathcal{B}$ and $\sigma^2_\mathcal{B}$ are computed from the current mini-batch. They are different for every batch - this introduces a small amount of noise into the activations, which acts as a regularizer (like dropout, but softer).

During inference: we cannot use a batch of just one image to compute statistics (the mean of one number is itself - useless). Instead, BatchNorm maintains **running averages** of the mean and variance, updated exponentially during training:

$$\mu_{\text{running}} \leftarrow 0.9 \times \mu_{\text{running}} + 0.1 \times \mu_\mathcal{B}$$
$$\sigma^2_{\text{running}} \leftarrow 0.9 \times \sigma^2_{\text{running}} + 0.1 \times \sigma^2_\mathcal{B}$$

At inference, these fixed running statistics are used instead of batch statistics, so single-example inference works correctly.

### 4.7.2 Layer Normalization: The Transformer Alternative

BatchNorm normalizes across the batch (comparing each example's activation to others in the same batch). This fails for:
- Batch size 1 (single example - no statistics to compute)
- Variable-length sequences (different examples have different numbers of valid positions)
- Online learning (future examples unavailable)

**Layer Normalization** normalizes across the feature dimension for each example independently:

$$\mu = \frac{1}{d}\sum_{j=1}^d z_j, \quad \sigma^2 = \frac{1}{d}\sum_{j=1}^d (z_j - \mu)^2, \quad \hat{z}_j = \frac{z_j - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \quad y_j = \gamma_j \hat{z}_j + \beta_j$$

Reading:
- The mean and variance are computed over the $d$ *features* of a single example, not over the batch
- $\gamma_j$ and $\beta_j$: learnable per-feature scale and shift (now a vector, not a scalar)
- The normalization is identical at training and inference time - no running statistics needed

LayerNorm is the standard for Transformers because each token is normalized independently, working correctly regardless of batch size or sequence length.

| Method | Normalizes over | Best for |
|--------|----------------|---------|
| Batch Norm | Batch dimension (per channel) | CNNs with large batches |
| Layer Norm | Feature dimension (per example) | Transformers, RNNs |
| Instance Norm | Per channel, per example | Style transfer |
| Group Norm | Channel groups, per example | Detection with small batches |



## 4.8 Data Augmentation: Manufacturing Experience

If the model is overfitting because it has too little data relative to its complexity, the most direct solution is more data. When we cannot collect more data, we can *synthesize* it through **data augmentation**: applying label-preserving transformations to existing examples.

**The mathematical principle (vicinal risk minimization):** Rather than minimizing the loss at discrete training points, minimize the loss in a *neighborhood* around each point. Augmented examples are samples from these neighborhoods - they represent the same underlying data point, but with small variations that the model should be invariant to.

For image classification, a cat photograph has the same label whether it is:
- Flipped horizontally (most cats look the same from either side)
- Slightly rotated ($\pm 15°$)
- Slightly cropped (the cat is still present)
- Slightly brighter or lower contrast (the cat's identity does not depend on exact lighting)

Training on these augmented versions forces the model to learn features that are robust to these transformations - invariant to what does not affect the label, sensitive only to what does.

### 4.8.1 Standard Augmentations

For image classification (ImageNet recipe):
- Random horizontal flip (with probability 0.5)
- Random crop to $224 \times 224$ from a $256 \times 256$ image
- Color jitter: random changes to brightness ($\pm 20\%$), contrast ($\pm 20\%$), saturation ($\pm 20\%$), hue ($\pm 5\%$)
- Normalization by ImageNet mean and standard deviation

These simple augmentations alone typically improve top-1 accuracy on ImageNet by 2–3 percentage points - a significant gain from pure engineering.

### 4.8.2 Mixup: Interpolating Between Examples

**Mixup** (Zhang et al., 2018) creates synthetic training examples by linearly interpolating between *pairs* of examples:

$$\tilde{\mathbf{x}} = \lambda\, \mathbf{x}_i + (1-\lambda)\, \mathbf{x}_j, \qquad \tilde{\mathbf{y}} = \lambda\, \mathbf{y}_i + (1-\lambda)\, \mathbf{y}_j, \qquad \lambda \sim \text{Beta}(\alpha, \alpha)$$

Reading:
- $\lambda$ (lambda): a mixing coefficient between 0 and 1, drawn from a Beta distribution. The Beta distribution with both parameters equal to $\alpha$ (typically $\alpha \in [0.1, 0.4]$) produces values mostly near 0 or 1 (mostly pure examples) with occasional values near 0.5 (strong mixtures)
- $\lambda\, \mathbf{x}_i + (1-\lambda)\, \mathbf{x}_j$: a weighted average of two images. If $\lambda = 0.7$: the mixed image is 70% from image $i$ and 30% from image $j$ - a ghostly double exposure
- $\tilde{\mathbf{y}}$: the mixed label, weighted by the same $\lambda$. If image $i$ is a cat (label $[1,0]$) and image $j$ is a dog (label $[0,1]$): $\tilde{\mathbf{y}} = 0.7[1,0] + 0.3[0,1] = [0.7, 0.3]$ - "70% cat, 30% dog"

**Why Mixup helps:** Training on mixed examples forces the model to behave linearly between training examples - it cannot memorize "this exact pixel pattern = cat" but must learn smooth, interpolable features. This strongly regularizes the decision boundary, especially in the regions between training examples where the model's predictions would otherwise be unconstrained.

### 4.8.3 CutMix: Pasting Patches

**CutMix** (Yun et al., 2019) cuts a rectangular patch from one image and pastes it into another, adjusting the label by the area of the patch:

$$\tilde{\mathbf{x}}[r] = \mathbf{x}_j[r], \quad \tilde{\mathbf{x}}[\bar{r}] = \mathbf{x}_i[\bar{r}], \qquad \tilde{\mathbf{y}} = \lambda\, \mathbf{y}_i + (1-\lambda)\, \mathbf{y}_j$$

Where $r$ denotes the pasted rectangular region, $\bar{r}$ the rest of the image, and $\lambda$ is the fraction of the image *not* covered by the patch.

Unlike Mixup's blurry double-exposure, CutMix maintains crisp image content - just in different spatial locations. This better simulates real occlusion (objects partially hidden behind other objects) and forces the model to identify objects from partial views.

### 4.8.4 Label Smoothing: Soft Targets

Standard training uses one-hot labels: $[0, 0, 1, 0, \ldots]$ for class 3. This tells the model to be exactly 100% confident in the correct class - an impossible standard that drives internal logits (the pre-softmax scores) to arbitrarily large values.

**Label smoothing** replaces one-hot with soft targets:

$$\tilde{y}_k = (1-\varepsilon)\, y_k + \frac{\varepsilon}{K}$$

Reading:
- $y_k$: the original one-hot label (1 for true class, 0 for others)
- $\varepsilon$: the smoothing parameter, typically $0.1$
- $\frac{\varepsilon}{K}$: a small uniform probability over all $K$ classes
- Result: the true class gets probability $1 - \varepsilon + \varepsilon/K \approx 0.9$ (not 1.0); all other classes get $\varepsilon/K \approx 0.0001$ (not 0.0)

**What this achieves:**
- The model is never rewarded for predicting probability 1 - this would require infinitely large logits, which we do not want
- The model is gently pushed to maintain small probabilities for incorrect classes, preventing it from becoming completely "blind" to non-true classes
- Training is better calibrated: the model's confidence scores correlate more accurately with its actual accuracy



## 4.9 Implicit Regularization: The Optimizer's Hidden Hand

One of the most surprising discoveries in modern deep learning theory: the optimizer itself acts as a regularizer, even without any explicit penalty term.

### 4.9.1 Gradient Descent Prefers Minimum-Norm Solutions

For overparameterized linear regression ($d > N$: more parameters than examples), there are infinitely many weight vectors that fit the training data perfectly. Gradient descent, initialized at zero and run until convergence, finds a specific one: the **minimum L2-norm interpolator**.

**Theorem (Gradient Descent Implicit Bias):** For linear regression $y = Wx$ with more parameters than data ($d > N$), starting from $W_0 = 0$ and running gradient descent with small $\eta$, the solution converges to:

$$W_\infty = \arg\min_{W: XW^T = Y} \|W\|_F^2$$

Reading:
- $\arg\min_{W: XW^T = Y}$: "find $W$ that minimizes... subject to the constraint that $W$ perfectly fits the training data ($XW^T = Y$)"
- $\|W\|_F^2$: the Frobenius norm squared - sum of all squared entries of $W$

Among all solutions that fit the training data, gradient descent gravitates to the one with the smallest total weight magnitude. This is not L2 regularization (no explicit penalty was added) - it is the intrinsic geometric preference of gradient descent for the solution closest to the initialization (which is zero).

This implicit minimum-norm preference is why neural networks can memorize training data (zero training error) while still generalizing - the memorizing solution is the minimum-norm one, which tends to be smooth and simple.

### 4.9.2 SGD Noise Favors Flat Minima

Stochastic gradient descent uses a mini-batch gradient $g_t$, which is a noisy estimate of the true gradient. The noise level is approximately:

$$\text{Noise} \propto \frac{\text{Gradient Variance}}{\text{Batch Size}} \propto \frac{\eta}{\text{Batch Size}}$$

Reading: the "effective noise temperature" of SGD is proportional to the learning rate $\eta$ and inversely proportional to the batch size.

**High noise temperature** (small batch, large $\eta$): the optimizer bounces vigorously, able to escape narrow valleys and saddle points. It tends to find flat, broad minima that are robust to perturbations - and hence generalize well.

**Low noise temperature** (large batch, small $\eta$): the optimizer settles wherever it lands. If it found a sharp minimum, it stays there. Large-batch training consistently produces worse generalization than small-batch training at the same final training loss - a concrete demonstration of implicit regularization by SGD noise.

**Practical implication:** If you increase the batch size (to use GPU compute more efficiently), you must also increase the learning rate proportionally to maintain the same noise temperature: $\eta_{\text{new}} = \eta_{\text{old}} \times \frac{\text{batch\_size\_new}}{\text{batch\_size\_old}}$. This "linear scaling rule" is empirically well-validated.



## 4.10 Early Stopping: Knowing When to Quit

**Early stopping** monitors validation loss during training and halts when validation loss stops improving. This is not just a heuristic - it has a deep theoretical connection to L2 regularization.

### 4.10.1 The Equivalence to L2 Regularization

For linear models trained with gradient descent for $t$ steps with learning rate $\eta$, early stopping is mathematically equivalent to L2 regularization with penalty:

$$\lambda_{\text{equivalent}} \approx \frac{1}{2\eta t}$$

Reading: stopping early (small $t$) is like strong regularization (large $\lambda$). Training longer (large $t$) is like weak regularization (small $\lambda$). Time acts as an inverse regularization strength.

**Intuitively:** Gradient descent learns the "easy" patterns first - those corresponding to large eigenvalues of the input covariance (directions of high data variance). Noise and spurious correlations tend to correspond to small eigenvalues, learned late in training. Early stopping filters out the late-learned noise while keeping the early-learned signal. This is called **spectral filtering**.

### 4.10.2 Implementing Early Stopping Carefully

Training loss is smooth (we optimize it directly), but validation loss is noisy - small fluctuations from batch-to-batch randomness, or from the stochastic nature of dropout. Stopping at the first sign of non-improvement would prematurely halt most training runs.

**Patience-based early stopping:**
1. Keep track of the best validation loss seen so far
2. Count consecutive epochs without improvement: the **patience counter**
3. If the counter exceeds the **patience** threshold $P$ (typically 5–20 epochs): stop training
4. Restore the model checkpoint from when the best validation loss was achieved

**Example:** Best validation loss was at epoch 50. Epochs 51–60 all have higher validation loss. With patience $P = 10$: stop at epoch 60, restore weights from epoch 50.

The right patience depends on the size of your validation set and the noisiness of your loss - noisier settings require more patience to avoid false alarms.



## 4.11 Double Descent: When Classical Theory Breaks Down

Classical statistics confidently predicts: as model complexity increases, test error first decreases (underfitting → good fit) then increases (good fit → overfitting). The optimal model has just the right complexity. Use more and overfit. This is the bias-variance tradeoff.

**Double descent** (Belkin et al., 2019) showed this picture is incomplete. For modern overparameterized models:

- **First descent:** Test error decreases as model complexity increases (classical region)
- **Interpolation threshold:** Test error spikes when the model first becomes complex enough to exactly fit the training data
- **Second descent:** Test error *decreases again* as the model becomes even more overparameterized, eventually reaching very low test error even while perfectly memorizing training data

> TODO: <!-- DIAGRAM: [A single curve showing test error (y-axis) vs. model size / complexity (x-axis). The curve traces a U-shape (first descent), then a spike upward at the "interpolation threshold", then a downward slope continuing to the right (second descent). The right end reaches lower test error than the classical minimum. Labels: "Underfitting", "Classical Minimum", "Interpolation Threshold", "Overparameterized Regime". Caption: "Classical theory predicts only the first U-shaped curve. Double descent extends this picture: beyond the interpolation threshold, more parameters help rather than hurt".] -->

**Why does the second descent occur?** In the overparameterized regime, there are many weight vectors that perfectly fit the training data. Gradient descent picks the minimum-norm one. For smooth, natural data (like images), the minimum-norm interpolating solution is surprisingly well-behaved - it fits the noise components in "irrelevant" directions of weight space, leaving the signal directions intact. The more overparameterized the model, the more directions are "irrelevant" and the smoother the noise fitting becomes.

**Practical implication:** Modern deep learning practitioners should not fear overparameterization. Large models with appropriate implicit and explicit regularization consistently outperform smaller models. The "right" model size is often "very large", not "just right".



## Summary

Regularization is the science of bridging training and test performance. It has many faces:

**Explicit penalties** (L1, L2) directly constrain the solution: "prefer weights near zero". L2 produces smooth, distributed weights; L1 produces sparse weights with many exact zeros.

**Dropout** creates an implicit ensemble of exponentially many thinned networks, forcing each neuron to develop robust, independent features. Bayesian interpretation: approximate inference over the posterior of weights.

**Normalization** (BatchNorm, LayerNorm) stabilizes training dynamics, smooths the loss landscape, and provides implicit regularization through stochastic batch statistics.

**Data augmentation** manufactures training examples by applying label-preserving transformations, teaching invariances the model should have and smoothing decision boundaries.

**Implicit regularization** arises from the optimizer itself: gradient descent is biased toward minimum-norm solutions; SGD noise favors flat minima; early stopping filters out late-learned noise.

Together, these techniques address the fundamental tension of all machine learning: a model flexible enough to represent the true function is also flexible enough to represent noise. Regularization navigates this tension by encoding our prior knowledge of what "the truth" looks like - smooth rather than jagged, simple rather than complex, robust rather than brittle.


*With the core machinery in place - networks, backpropagation, loss functions, optimizers, and regularization - we now turn to architectures designed for specific data types. The first and most successful is the Convolutional Neural Network, which encodes the spatial structure of images directly into its design. That is the subject of Chapter 5.*


---
*Continue to **[Chapter 5: The Visual Cortex of AI - Convolutional Neural Networks](/DeepLearning/05_CNNs.md)***
