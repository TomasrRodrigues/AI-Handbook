# Chapter 5: Convolutional Neural Networks - Teaching Machines to See

> *"Vision is the art of seeing what is invisible to others"* - Jonathan Swift



## 5.1 The Aha! Moment: Why Fully Connected Layers Cannot See

Imagine trying to recognize a handwritten "3" using a fully connected network. You flatten the $28 \times 28$ image into a vector of 784 pixel values. Layer 1 connects every one of those 784 inputs to every neuron. The problem: the weight connecting pixel $(5, 7)$ to neuron 42 has no mathematical relationship to the weight connecting the adjacent pixel $(5, 8)$ to the same neuron. **Spatial proximity is invisible to the model.** The image might as well be a shuffled list of random numbers.

Yet human vision is fundamentally spatial. An edge exists *between neighboring pixels*. A curve is a sequence of oriented edges. An eye is a specific spatial arrangement of curves and colors. All meaningful visual structure is local. Any architecture that ignores this has to rediscover locality from scratch, requiring enormous amounts of data and parameters.

The second problem: **translation sensitivity**. A stop sign in the top-left corner of an image shares no learned weights with a stop sign in the bottom-right. The fully connected model must learn to recognize the stop sign at every location independently - an exponential data requirement.

**Convolutional Neural Networks** solve both problems with one architectural constraint: **local filters with shared weights**. The same small filter (say, $3 \times 3$ weights) slides across the entire image, detecting the same feature at every location. This is simultaneously:
- **Local**: each filter application looks at only a small spatial neighborhood
- **Shared**: the same weights detect the same feature everywhere

This doubles as a statement about images: *features that are useful in one location are useful everywhere.* Corners, edges, and curves do not care where in the image they appear. The CNN's architecture encodes this truth.



## 5.2 The Convolution Operation

### 5.2.1 The Mathematical Definition

A **discrete 2D convolution** (strictly, a cross-correlation, but the term "convolution" is standard in deep learning) of image $I$ with filter $K$ produces output $Y$:

$$(I \otimes K)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i+m,\; j+n] \cdot K[m, n]$$

Reading this completely:
- $[i, j]$: the output location (row $i$, column $j$)
- $\sum_{m=0}^{k-1} \sum_{n=0}^{k-1}$: sum over all $k^2$ positions within the filter window ($k$ is the filter size)
- $I[i+m, j+n]$: the pixel of the image at position $(i+m, j+n)$ - the local patch centered at $(i,j)$
- $K[m, n]$: the filter weight at position $(m, n)$
- The product $I[i+m,j+n] \cdot K[m,n]$: weight the pixel by the filter value
- The sum: add up all weighted pixels in the local window

**In plain English:** slide the filter over the image, position by position. At each position, multiply each filter value by the overlapping pixel value, and add everything up. The result is one output value. Repeat for every position to produce the full output (feature map).

**Why does this detect features?** A filter whose weights look like an oriented edge (positive values on one side, negative on the other) will produce a large positive output when placed over an actual edge in that orientation, and near-zero output over uniform regions. By learning filter weights through backpropagation, the network discovers filters that detect whatever features are most useful for the task.

### 5.2.2 Translation Equivariance: The Core Property

The single most important mathematical property of convolution is **translation equivariance**.

**Definition:** A function $\Phi$ is translation equivariant if, for any translation $T_\mathbf{v}$ (shift by vector $\mathbf{v}$):

$$\Phi(T_\mathbf{v} f) = T_\mathbf{v}(\Phi(f))$$

Reading: shift the input first, then apply $\Phi$ - you get the same result as applying $\Phi$ first, then shifting. In plain language: if you shift the image, the output shifts by exactly the same amount.

**Why this matters:** A cat-ear detector that works in the top-left also works in the bottom-right - the same filter weights detect the feature at any location. The feature map faithfully tracks *where* features are in the input.

**Theorem (Kondor & Trivedi, 2018):** Under mild conditions, convolution is the *unique* linear translation-equivariant operator. If you want a linear layer that respects spatial shifts, you are mathematically constrained to use convolution. This is not an architectural choice - it is a mathematical inevitability.

### 5.2.3 Multi-Channel Convolution

Real images have three channels (R, G, B). Feature maps have many channels (one per filter). A convolutional layer with $C_{in}$ input channels and $C_{out}$ output channels applies $C_{out}$ different filters, each of shape $k \times k \times C_{in}$:

$$Y[i, j, c] = \sigma\!\left(\sum_{c'=1}^{C_{in}}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X[i+m, j+n, c'] \cdot W[m, n, c', c] + b[c]\right)$$

Reading:
- $Y[i, j, c]$: output at position $(i,j)$ in channel $c$ of the output
- $\sum_{c'=1}^{C_{in}}$: sum over all input channels $c'$
- $X[i+m, j+n, c']$: input at position $(i+m, j+n)$ in input channel $c'$
- $W[m, n, c', c]$: weight connecting input channel $c'$ at position $(m,n)$ to output channel $c$
- $b[c]$: bias for output channel $c$
- $\sigma$: activation function applied after the weighted sum

**Parameter count:** $C_{out} \times (k^2 \times C_{in} + 1)$. For $k=3$, $C_{in}=C_{out}=64$: only $64 \times (9 \times 64 + 1) = 36{,}928$ parameters - compared to a fully connected layer for the same input/output size, which would need millions.

<!-- DIAGRAM: [Left: a $5\times5$ input image patch with values shown. A $3\times3$ filter matrix with learned weights. Arrow showing element-wise multiplication and sum to produce one output value. Right: the filter sliding across a full $7\times7$ image, producing a $5\times5$ output feature map. Arrows show that the same filter weights are reused at every position. Caption: "The filter slides: at each position, multiply corresponding values and sum. The same weights produce different outputs depending on what is under the filter - that is how a single filter can detect a feature across the entire image."] -->



## 5.3 Pooling: Spatial Summarization

After convolution, pooling layers reduce spatial resolution, building invariance and expanding the receptive field.

### 5.3.1 Max Pooling

$$P_{\max}[i, j] = \max_{\Delta i \in [0,k), \Delta j \in [0,k)} X[s \cdot i + \Delta i,\; s \cdot j + \Delta j]$$

Reading:
- $P_{\max}[i, j]$: the output at position $(i,j)$ after max pooling
- $\max_{\Delta i, \Delta j}$: take the maximum over a $k \times k$ window
- $s$: the stride - how far the window moves between positions
- $X[s \cdot i + \Delta i, s \cdot j + \Delta j]$: values in the window at position $(s \cdot i + \Delta i, s \cdot j + \Delta j)$

**The question max pooling asks:** "Was this feature present *anywhere* in this region?" Not "how strongly present on average" - just "was there a strong detection somewhere?" This is appropriate for detecting sparse features: even one strong edge response in a region indicates an edge exists there.

**Backpropagation through max pooling:** The gradient flows only to the "winning" pixel (the one that achieved the maximum). All others receive zero gradient. This is a form of implicit attention - the gradient tracks which pixels were responsible for each detection.

**Global Average Pooling (GAP):** A special case that averages the *entire* feature map to one value per channel:

$$\text{GAP}(X)_c = \frac{1}{H \times W} \sum_{i=1}^{H}\sum_{j=1}^{W} X[i, j, c]$$

Reading: sum all values in channel $c$ across all $H \times W$ spatial positions, then divide by $H \times W$ to get the average. GAP is exactly translation-invariant: no matter where a feature appears, the global average captures its presence equally.

Modern networks (ResNet, EfficientNet) use GAP before the final classification layer, replacing the large fully connected layers used in early architectures like AlexNet - reducing parameters dramatically while providing better invariance.



## 5.4 Receptive Fields: How Depth Creates Global Understanding

A single $3 \times 3$ convolution sees only a $3 \times 3$ patch of the image. How does a CNN eventually "understand" the whole image?

**The receptive field (RF)** of a neuron is the region of the original input that can influence its value. For a single $3 \times 3$ conv layer: RF = $3 \times 3$. After two layers: the second layer sees a $3 \times 3$ patch of first-layer outputs, each of which already "saw" $3 \times 3$ pixels. So the second-layer neuron effectively sees a $5 \times 5$ region of the original image.

The formula for RF after $L$ layers of kernel size $k$ and stride 1:

$$\text{RF}_L = 1 + L(k-1)$$

Reading: starts at 1 (a single pixel), grows by $k-1$ for each additional layer. For $k=3$: grows by 2 per layer. After 50 layers: RF = $1 + 50 \times 2 = 101 \times 101$.

**Why deep networks are needed:** A ResNet-50 has ~50 layers of $3\times3$ convolutions, giving a theoretical RF of $101\times101$ - roughly 20% of a $224\times224$ image. To "see" large objects that span most of the image, very deep networks are necessary.

**Dilated convolutions** expand the RF without adding layers or parameters:

$$Y[i, j] = \sum_{m, n} X[i + r \cdot m,\; j + r \cdot n]\, W[m, n]$$

Reading: instead of sampling adjacent pixels, sample at intervals of $r$ (the dilation rate). A $3 \times 3$ filter with dilation $r=2$ samples a $5 \times 5$ area (skipping every other row and column), using only 9 parameters instead of 25. For $L$ layers with dilation rates $\{r_\ell\}$:

$$\text{RF}_L = 1 + \sum_{\ell=1}^{L} r_\ell(k-1)$$

Using exponentially growing dilation ($r_\ell = 2^{\ell-1}$): RF grows exponentially with depth while parameter count stays constant. This is the foundation of WaveNet (audio) and DeepLab (segmentation).



## 5.5 Architectural Milestones

### 5.5.1 The Degradation Problem and ResNets

Before 2016, adding more layers to a network beyond a certain depth consistently *increased* training error - not from overfitting, but from the fundamental difficulty of optimization. A deeper network should in principle be at least as good as a shallower one (the extra layers could learn identity mappings). But gradient-based optimization could not find this solution.

**The residual connection:** Instead of learning the full mapping $\mathcal{H}(\mathbf{x})$, a residual block learns the *residual* $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$:

$$\mathbf{y} = \mathcal{F}(\mathbf{x};\, W_1, W_2) + \mathbf{x}$$

Reading:
- $\mathcal{F}(\mathbf{x}; W_1, W_2)$: the residual branch - two convolutional layers applied to $\mathbf{x}$
- $+ \mathbf{x}$: the skip connection - add the original input directly
- $\mathbf{y}$: the block's output - the sum of the transformed and original signals

**Why residual learning is easier:** If the optimal mapping is close to identity (often true in deeper layers), then the residual $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x} \approx 0$ - easy to achieve (drive weights toward zero). If the full mapping $\mathcal{H}(\mathbf{x})$ were the identity, learning it directly requires the network to hit exactly the right combination of weights - much harder.

**Gradient flow through residuals:** The gradient of the output with respect to the input:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} + \frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + I$$

The identity matrix $I$ is always present. Even if $\partial \mathcal{F}/\partial \mathbf{x} \to 0$ (the residual branch contributes nothing), the gradient is $I$ - it passes unchanged. Multiplied across 100 blocks: the gradient flows from the output to any early layer through the chain of identity additions, without multiplicative decay.

### 5.5.2 Depthwise Separable Convolutions

Standard $3 \times 3$ convolution with $C_{in}$ input and $C_{out}$ output channels: $k^2 \times C_{in} \times C_{out}$ parameters and $O(k^2 C_{in} C_{out} HW)$ computation.

**Depthwise separable convolution** factorizes this into:

1. **Depthwise conv:** Apply one filter per input channel independently. Parameters: $k^2 \times C_{in}$. Learns spatial features within each channel separately.
2. **Pointwise conv:** Apply a $1 \times 1$ convolution to combine channels. Parameters: $C_{in} \times C_{out}$. Learns to combine the per-channel spatial features.

Total parameters: $k^2 \times C_{in} + C_{in} \times C_{out} = C_{in}(k^2 + C_{out})$.

**Reduction ratio vs. standard:** $\frac{C_{in}(k^2 + C_{out})}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2}$.

For $k=3$ and $C_{out}=256$: reduction of approximately $\frac{1}{256} + \frac{1}{9} \approx 11\%$ of original cost - an $8$–$9\times$ reduction. This is why MobileNets and EfficientNets use depthwise separable convolutions for mobile and edge deployment.



## 5.6 Feature Hierarchy: What CNNs Actually Learn

Visualization studies (Zeiler & Fergus, 2014) revealed that CNN layers learn a consistent hierarchy:

**Layer 1:** Oriented edges, color blobs, simple Gabor-like patterns - strikingly similar to receptive fields of neurons in primate primary visual cortex (V1). The network discovers, without being told, the same primitives that evolution gave the brain.

**Layers 2–3:** Junctions, curves, simple textures composed of edges. The second layer combines multiple edge detectors to form higher-level structures.

**Layers 4–5:** Object parts - wheels, eyes, beaks, legs. Specific configurations of mid-level features that correspond to semantic parts.

**Final layers:** Whole objects, scenes, categories. The highest-level features discriminate between image categories.

**Why this hierarchy emerges:** The network discovers that representing images as hierarchical compositions (edges → parts → objects) is the most information-efficient encoding for classification. This is not programmed - it is discovered by minimizing cross-entropy through backpropagation.



## 5.7 Vision Transformers: Questioning the Convolution Assumption

The CNN's fundamental assumption - local spatial structure is the most important prior - is an **inductive bias**. It helps with limited data (the model does not need to learn that nearby pixels are related - it is built in). But it constrains the model: long-range relationships between distant image regions require many layers to emerge.

**Vision Transformers (ViT)** (Dosovitskiy et al., 2020) test whether this inductive bias is necessary. Instead of local filters:

1. Divide the image into non-overlapping $16 \times 16$ patches
2. Linearly project each patch to a vector (a "patch embedding")
3. Apply a standard Transformer (self-attention) over these patch vectors

Self-attention allows every patch to directly attend to every other patch - arbitrarily long-range dependencies at every layer, with no limitation from receptive field growth.

**The data tradeoff:** CNNs generalize well with moderate data (the spatial inductive bias helps). ViT, lacking this prior, requires much more data to learn that nearby patches are related. On ImageNet alone (~1.2M images), CNNs outperform ViT. On massive datasets (300M+ images), ViT matches or exceeds CNNs - when enough data is available, the flexible attention mechanism learns richer representations than convolutional priors allow.

**The insight:** The convolutional inductive bias is a form of regularization - it encodes correct prior knowledge about images and helps with limited data. When data is abundant, the prior becomes a constraint rather than a help.



## 5.8 Summary

Convolutional Neural Networks encode a deep architectural truth: **the structure of the model should match the structure of the data**. Images are local (nearby pixels are more related than distant ones) and translation-equivariant (features can appear anywhere). Convolution is the unique linear operation satisfying both properties.

The history of CNN architecture is a history of discovering how to exploit depth more effectively: from AlexNet's GPU-powered ReLU networks to ResNets' gradient highways, to depthwise separable convolutions and compound scaling. Each innovation solved a specific failure of the previous approach.

What unifies them: **the right prior makes learning efficient**. Less data needed, fewer parameters required, better generalization achieved - all because the model's structure assumes what is true about the world.



---
*Continue to **[Chapter 6: Memory and the Flow of Time - Recurrent Networks and LSTMs](/DeepLearning/06_RNNs_and_LSTMs.md)***
