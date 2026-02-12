# LLM Architecture Foundations

#### Table of Contents

1. [Tokenization](#tokenization)


---
## Tokenization

Tokenization is the critical first step of converting raw text into discrete units (tokens) that can be processed by language models. The choice of tokenization method significantly impacts model performance, vocabulary size and computational efficiency.

This seemingly simple preprocessing step profoundly impacts:
- Model vocabulary size (memory footprint)
- Sequence length (computational cost)
- Out-of-vocabulary handling
- Multilingual capability
- Downstream task performance


### Granularity Levels

#### Word-Level Tokenization

Each word is treated as a single token. It is easy to understand and interpret and there are fewer tokens per sequence which leads to faster inference. On the other hand, vocabulary is really extensive, which might lead to a Out-of-Vocabulary (OOV) problem for unseen words. Also there are many morphological variations of the same words (run, running, ran) which can lead to poor handling.

Example: $\text{"teddy bear"} \rightarrow ["teddy","bear"]$


#### Subword-Level Tokenization

Words are divided into meaningful subunits. This creates a smaller, more balanced vocabulary size and handles rare/unseen words better, and captures morphological patterns (prefixxes, suffixes). On the other hand, this creates longer sequences than word-level. Not only that but this approach is much more complex to implement.

Example: $\text{"teddy bear"} \rightarrow \text{["ted", "\#\#dy", "bear"]}$


#### Character-Level Tokenization

Each character is a token. The advantage of this is that there is a very small vocabulary (size of the alphabet, 26-256 tokens) and there is no OOV problem as there are not new letters being created. On the other hand, this generates very long sequences which makes inference really slow. Also, each token would lack semantic meaning as it would only be a letter.

Example: $\text{"teddy bear"} \rightarrow \text{["t", "e", "d", "d", "y", " ", "b", "e", "a", "r"]}$ 


#### Byte-Level Tokenization

Raw bytes (UTF-8 encoding) as tokens. This has a fixed vocabulary of 256, is universal as it handles any language, emoji or special characters and needs no preprocessing. On the other hand, it creates extremely long sequences and it is difficult to interpret byte-level patterns.

Example: $\text{"teddy bear"} \rightarrow [116,101,100,100,121,32,98,101,97,114]$


### Subword Tokenization Algorithms

#### Byte-Pair Encoding (BPE): 

Byte-Pair Encoding was originally a data compression algorithm and was adapted for neural machine translation by and has become the most widely used tokenization method for modern LLMs.The idea of this is to iteratively merge the most frequent pairs of tokens to build vocabulary of varying granularity from characters to full words. Training process:
1. Initialize vocabulary with all unique characters in corpus
2. Count frequency of all adjacent token pairs
3. Merge the most frequent pair into a new token
4. Repeat steps 2-3 until reaching desired vocabulary size.

Encoding process:
1. Start with character-level segmentation
2. Apply learned merge rules in order
3. Greedily merge longest matching sequences

This is really simple and efficient and generates a vocabulary built by data. This has a good compression ratio. The problem is that greedy merging may not be globally optimal as it is sensitive to character frequency distribution. 

**BPE Variants**:
1. Byte-Level BPE: Instead of character-level, start with 256 bytes. This handles any unicode text without preprocessing. There are no language-specific rules needed.
2. Dropout BPE (data augmentation): Randomly skip merges during encoding for robustness. As a result this creates multiple tokenizations of same text and builds a more robust model.


This is simple and intuitive, constructs data-driven vocabulary, handles rare words through character fallback, is language-agnostic and has a good compression ratio.

On the other hand, greedy merging may not be globallly optimal, it is sensitive to character frequency distribution, can split words inconsistently and has no explicit modeling of morphology.


#### WordPiece

The core idea is similar to BPE, but merges based on likelihood increase rather than frequency. The key difference from BPE is that BPE merges most frequent pairs and WordPiece merges pairs that maximizes likelihood of training data. Merge selection follows this formula:

$$
score(pair) = \frac{count(pair)}{count(first) \times count(second)}
$$

A special token `##` is used as a prefix to indicate subword continuation. For example `##dy` means that *dy* is not a word start

This is better at handling linguistic structure and more principled than frequency-based BPE.


#### Unigram Language Model

This start with large vocabulary and iteratively removes tokens to minimize loss. The training process is:
1. Initialize with all possible substrings from corpus
2. Train probabilistic model: $P(sentence)= \prod P(token_i)$
3. For each token, compute loss increase if removed
4. Remove tokens with smallest loss increase (keep top 80%)
5. Repeat until reaching target vocabulary size

This creates a probabilistic framework and can generate multiple segmentation candidates. Is much more flexible than BPE/WordPiece. On the other hand, this is computationally more expensive and requires EM algorithm for training.


Most tokenizers include special tokens for specific purposes:

| Token | Purpose | Usage Example |
|-------|---------|---------------|
| `[PAD]` | Padding | Fills sequences to fixed length for batching |
| `[UNK]` | Unknown | Represents OOV words |
| `[CLS]` | Classification | Sentence representation (BERT) |
| `[SEP]` | Separator | Delimits sentence pairs |
| `[MASK]` | Mask | Masked language modeling |
| `[BOS]` | Begin-of-sequence | Marks sequence start (GPT) |
| `[EOS]` | End-of-sequence | Marks sequence end |

### Normalization Techniques

Before tokenization, text often undergoes normalization:

| Method | Purpose | Example |
|--------|---------|---------|
| Lowercasing | Remove case variations | `Teddy` → `teddy` |
| Accent removal | Handle diacritics | `café` → `cafe` |
| Unicode normalization | Standardize characters | Various forms of é → single form |
| Whitespace normalization | Consistent spacing | Multiple spaces → single space |


---
## Embeddings

### From Discrete to Continuous

Tokens are discrete symbols that must be converted to continuous vectors for neural network processing.

#### One-Hot Encoding
This is a baseline, not used in practice. This works by keeping keeping an array with the size of the vocabulary for each word. This has a big problem which is dimensionality (very high). It also has no semantic relationships, all tokens are equidistant and it is sparse and inefficient.

#### Learned Dense Embeddings
This is the standard approach. This works by keeping a vocabulary V with $|V|$ tokens. The embedding dimension $d_model$. An embedding matrix is kept: 

$$
E \in \mathbb{R}^{|V| \times d_{model}}
$$

#### Word2Vec

The core insight is that "we shall know a word by the company it keeps" (Distributional Hypothesis). It has two variants:
1. Continuous Bag of Words (CBOW): Predicts target word from context words