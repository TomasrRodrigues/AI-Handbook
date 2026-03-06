# Large Language Models: Architectural Foundations

This repository serves as a comprehensive guide to the structural pillars of modern Artificial Intelligence. It provides a high-level map of how Large Language Models (LLMs) are built, trained, and utilized.


## Table of Contents

1. [LLM Architecture Overview](#1-llm-architecture-overview)
2. [Tokenization](#2-tokenization)
3. [Attention Mechanisms](#3-attention-mechanisms)
4. [Sampling Techniques](#4-sampling-techniques)


## 1. [LLM Architecture Overview](./LLMArchitectureOverview.md)
The transition from sequential Recurrent Neural Networks (RNNs) to the **Transformer** architecture in 2017 enabled the massive scaling of AI. Modern LLMs contain billions or trillions of parameters and exhibit "emergent capabilities"-skills like complex reasoning that only appear at great scale.

* **Core Structure**: Built using stacked layers of Multi-Head Attention and Feed-Forward networks.
* **Stability Mechanisms**: Uses Residual Connections (shortcuts for gradient flow) and Layer Normalization to keep numerical values stable during training.
* **Architectural Families**: Models are categorized as **Encoder-only** (e.g., BERT) for understanding, **Decoder-only** (e.g., GPT) for generation, or **Encoder-Decoder** (e.g., T5) for flexible text-to-text tasks.



## 2. [Tokenization](./Tokenization.md)
Tokenization is the "bridge" that converts raw human text into the numerical integers required by neural networks. The choice of tokenization method fundamentally shapes a model's efficiency and its ability to understand rare words.

* **Subword Solution**: Modern models use subword units to balance vocabulary size with sequence length, ensuring they can handle any word without falling back on "Unknown" tokens.
* **Key Algorithms**: Includes **Byte Pair Encoding (BPE)**, which is data-driven and merges frequent character sequences, and **WordPiece**, which optimizes for the likelihood of the training data.
* **Multilingual Support**: Byte-level BPE allows models to process any UTF-8 string, making them truly universal across languages and emojis.

## 3. [Attention Mechanisms](./AttentionMechanisms.md)
The **Attention Mechanism** allows a model to directly access any part of an input sequence simultaneously, removing the "information bottleneck" found in older sequential models.

* **The Analogy**: Much like a database, it uses **Queries (Q)**, **Keys (K)**, and **Values (V)** to determine which tokens are most relevant to the current word.
* **The Formula**: Attention is calculated using a scaled dot-product: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
* **Multi-Head Attention**: By running multiple attention mechanisms in parallel, the model can simultaneously track different linguistic facets, such as syntax, semantics, and reference.



## 4. [Sampling Techniques](./SamplingTechniques.md)
Sampling governs how a model chooses the next token from the probability distribution it generates at each step. It is the primary tool for balancing factual accuracy with creative variety.

* **Greedy Decoding**: Always picks the most likely word; efficient but often leads to repetitive or bland text.
* **Temperature**: Scales probabilities; low temperature (0.2) is best for factual tasks like coding, while high temperature (0.8+) is best for creative storytelling.
* **Nucleus (Top-P) Sampling**: An adaptive method that only considers a "nucleus" of tokens whose cumulative probability exceeds a threshold $p$, allowing the model to be exploratory when uncertain.

