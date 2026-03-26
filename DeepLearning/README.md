# Deep Learning

This section aims to cover all the topics of **Deep Learning**, from its foundation to the current state of the art. The goal isn't just to memorize equations, but to understand the Aha! moments where a specific mathematical constraint forced a new architectural evolution. Instead of hand-translating human knowledge into brittle code, these summaries track the development of mechanisms for acquiring knowledge.


| Chapter | Topic | Key Insight | 
|-----|----------|------------------------|
| [01](/DeepLearning/01_NeuralNetworks.md) | [Foundations](/DeepLearning/01_NeuralNetworks.md) | "The ""Architecture of Thought"" and the Universal Approximation Theorem." | 
| [02](/DeepLearning/02_Backpropagation.md) | [Backpropagation](/DeepLearning/02_Backpropagation.md) | "The ""Art of Assigning Blame"" through the multi-dimensional chain rule." | 
| [03](/DeepLearning/03_Optimizers_and_Loss.md) | [Loss & Optimization](/DeepLearning/03_Optimizers_and_Loss.md) | Loss functions as probabilistic statements about the structure of reality. |
| [04](/DeepLearning/04_Regularization.md) | [Regularization](/DeepLearning/04_Regularization.md) | Navigating the Bias-Variance tradeoff and the phenomenon of Double Descent. | 
| [05](/DeepLearning/05_CNNs.md) | [CNNs](/DeepLearning/05_CNNs.md) | CNNs and the encoding of spatial locality and translation equivariance. | 
| [06](/DeepLearning/06_RNNs_and_LSTMs.md) | [RNNs and LSTMs](/DeepLearning/06_RNNs_and_LSTMs.md) | "RNNs and LSTMs: Creating ""memory in motion"" to handle temporal data." |
| [07](/DeepLearning/07_Autoencoders_and_VAEs.md) | [Autoencoders and VAEs](/DeepLearning/07_Autoencoders_and_VAEs.md) | Autoencoders and VAEs: Learning the geometric manifolds of data. | 
| [08](/DeepLearning/08_Mamba_and_SSMs.md) | [Mamba and SSMs](/DeepLearning/08_Mamba_and_SSMs.md) | SSMs and Mamba: Breaking the quadratic wall of Transformers. |



## Citations

Citations (Foundational Papers)
You can organize your bibliography roughly by architectural era or topic:

### Foundations & Optimization
- McCulloch-Pitts Neuron: McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity.
- Stochastic Gradient Descent: Robbins, H., & Monro, S. (1951). A stochastic approximation method. Annals of Mathematical Statistics.
- Huber Loss: Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics.
- Backpropagation: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature.
- Universal Approximation: Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems.
- Adam Optimizer: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. ICLR.

### Regularization
- L1 Regularization (Lasso): Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society.
- Dropout: Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research.
- Batch Normalization: Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. ICML.

### Spatial & Vision Models (CNNs to ViTs)

- LeNet-5: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.
- AlexNet: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems.
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. CVPR.
- Vision Transformers: Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.

### Temporal & Sequence Models (RNNs to SSMs)
- SRNs: Elman, J. L. (1990). Finding structure in time. Cognitive Science.
- LSTM: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
- GRU: Chung, J., Gulcehre, C., Cho, K. H., et al. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling.
- Structured State Spaces (S4): Gu, A., Goel, K., & Ré, C. (2021). Efficiently modeling long sequences with structured state spaces. ICLR.
- Mamba: Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv.

### Generative Models & Emerging Architectures

- Variational Autoencoders: Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. ICLR.
- Kolmogorov-Arnold Networks: Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv.

### Further Reading

For a broader educational perspective, I highly recommend adding these comprehensive resources:

- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016): Often considered the definitive textbook of the field, it rigorously covers the underlying linear algebra, probability, and foundational machine learning techniques that power deep networks.
- "Dive into Deep Learning" by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola (2023): An excellent, interactive open-source textbook. It stands out because it blends mathematical exposition with runnable, self-contained code implementations in PyTorch, JAX, and TensorFlow.
- "From S4 to Mamba: A Comprehensive Survey on Structured State Space Models" by Shriyank Somvanshi et al. (2025): If readers want to understand the very latest shift away from Transformers, this recent survey perfectly outlines the evolution of sequence modeling from RNNs to S4, Mamba, and Jamba architectures.