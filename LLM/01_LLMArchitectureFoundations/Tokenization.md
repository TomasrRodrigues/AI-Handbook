# Tokenization

#### Table of Contents

1. [Introduction](#introduction)
2. [Character-Level Tokenization](#character-level-tokenization)
3. [The Subword Solution](#the-subword-solution)
4. [Byte Pair Encoding](#byte-pair-encoding)
5. [WordPiece and Other Approaches](#wordpiece-and-other-approaches)
6. [Practical Considerations and Challenges](#practical-considerations-and-challenges)
7. [The Impact of Tokenization on Model Behavior](#the-impact-of-tokenization-on-model-behavior)
8. [The Future of Tokenization](#the-future-of-tokenization)

## Introduction

Before any Large Language Model can begin to process text, that text must first be broken down into units that the model can work with. This fundamental preprocessing setp is called **tokenization**, and while it might seem like a minor technical detail, the way we choose to tokenize text has profound implications for model performance, efficiency and capabilities. ***Tokenization is the bridge between raw text as humans write it and the numerical representations that neural networks require.***

At its most basic level, tokenization is the process of **converting a string of text into a sequence of smaller units called tokens**. These tokens become the vocabulary items that the model understands - the building blocks from which the model constructs its understanding of language. 

When we type a sentence into a language model interface, that sentence is first tokenized before any of the sophisticated neural network processing we discussed earlier can begin. The choice of how to break text into tokens fundamentally shapes:
- What patterns the model can learn
- How efficiently it can process different types of text
- What representations it builds internally


To understand why tokenization matters, consider the challenge that language model face. ***Neural networks work with numbers, not words.*** Each token in the model's vocabulary is assigned a unique integer identifier, and these identifiers are used to look up the embeddings that we discussed earlier. 

The tokens we choose as our vocabulary directly determine what the model considers to be **atomic units of meaning**:
- If our tokens are **individual characters**, the model must learn to assemble those characters into meaningful words
- If our tokens are **complete words**, the model can work directly with word-level meaning but struggles with rare or unknown words


The question of how to tokenize text might seem to have an obvious answer: just split on spaces to separate words. This approach, called **word-level tokenization**, is intuitive and was used by many early natural language processing systems. For a sentence like $\text{"The cat sat on the mat ".}$, word-level tokenization produces six word tokens plus punctuation:

$$
\text{"The cat sat on the mat ".} \quad \xrightarrow{Tokenization} \quad \text{["The" ",cat" ",sat" ",on" ",the" ",mat" ", ".]}
$$

This seems natural because it matches how we think about language - as sequence of words. However, this straightforward approach reveals significant problems when we examine it more closely.

The first issue with word-level tokenization is **vocabulary size**. Natural languages contain enormous numbers of distinct words:
- English has **hundreds of thousands of words** in common use
- The number grows even larger when we include proper names, technical terms, and inflected forms

If we want our model to handle all possible words, we need an enormous vocabulary, which means an **enormous embedding matrix**. Remember that each word needs its own embedding vector, so a vocabulary of 500,000 words with embedding dimension 768 would require **384 million parameters** just for the embeddings - a substantial portion of the total parameters even in large models.


More problematic is how word-level tokenization handles rare and unknown words. During training, the model only sees a finite amount of text, which means many valid words will be extremely rare or completely absent from the training data:
- **Rare words**: Words that appear only once or twice won't have well-learned embeddings because there weren't enough examples for the model to understand how they're used
- **Unknown words**: Words that never appear in the training data pose an even worse problem. The standard solution is to include a special unknown word token, typically written as $\text{[UNK]}$ or $\text{<unk>}$, that represents any word not in the vocabulary

**The problem with this approach**: It loses information. The words "antidisestablishmentarianism" and "pseudopseudohypoparathyroidism" are both rare medical or technical terms, but if neither appears in your vocabulary, they would both be converted to the same $\text{[UNK]}$ token, losing the distinction between them.

Word-level tokenization also misses important relationships between related words. Consider the words:
- $\text{"care"}$, $\text{"cares"}$, $\text{"cared"}$, $\text{"caring"}$
- $\text{"careful"}$, $\text{"carefully"}$
- $\text{"careless"}$, $\text{"carelessly"}$, $\text{"carelessness"}$

These are all **morphologically related** - they're different forms or derivatives of the same root word $\text{"care"}$. But word-level tokenization treats each as a completely separate vocabulary item with its own embedding. 

The model must learn the relationships between these forms from scratch based on how they're used in context, rather than being able to leverage their obvious structural similarity. This makes learning **less efficient** and requires more training data to achieve the same level of understanding.


## Character-Level Tokenization

At the opposite extreme from word-level tokenization is **character-level tokenization**, where text is broken down into individual characters:

$$
\text{"CAT"} \quad \xrightarrow{Tokenization} \quad \text{["C" ",A" ",T"]}
$$

This approach solves the **vocabulary size problem completely**. English only needs about 100 characters including uppercase and lowercase letters, digits, punctuation, and a few special characters. There's **no concept of an unknown word** because any word, no matter how rare or how it's spelled, can be represented as a sequence of known characters.

**Handling morphology naturally**: Character-level tokenization naturally handles morphological relationships. Since $\text{"care"}$, $\text{"cared"}$, and $\text{"caring"}$ all share the character sequence $\text{[c, a, r, e]}$, a character-level model can potentially learn to recognize this shared structure and understand that these words are related.

**Flexibility with spelling variations**: The model can handle misspellings, creative spellings, and made-up words more gracefully because it's working at the level of characters rather than trying to match against a fixed vocabulary of complete words. If someone writes $\text{"soooo"}$ with extra o's for emphasis, a character-level model can process this just as easily as $\text{"so"}$, whereas a word-level model might not have $\text{"soooo"}$ in its vocabulary.

However, character-level tokenization introduces **severe drawbacks** that make it impractical for most applications.

The most obvious problem is **sequence length**. That simple sentence "The cat sat on the mat" has:
- **7 word-level tokens**
- **22 character-level tokens** (when we count spaces and punctuation)

For a long document, character-level tokenization might produce sequences **tens of thousands of tokens long**. This is a serious problem because transformer models have computational complexity that grows **quadratically with sequence length**. Processing a sequence that's three times longer requires **nine times as much computation** in the attention layers. This makes character-level tokenization prohibitively expensive for long documents.

More fundamentally, character-level tokenization makes the **model's learning task much harder**. Individual characters carry very little semantic information by themselves. The character $\text{"c"}$ in isolation doesn't mean anything - meaning emerges from:
- Combinations of characters into words
- Combinations of words into sentences

This means a character-level model must:
1. First learn to assemble characters into word-like units
2. Then learn word meanings
3. Then learn how words combine into sentences

We're asking the model to learn at a **very low level of abstraction**, which requires more model capacity and more training data to achieve the same level of performance as a model working with more meaningful units.


## The Subword Solution

The solution that modern Large Language Models use is **subword tokenization**, which occupies the middle ground between word-level and character-level approaches. The key insight is that **words can be broken into meaningful pieces** that are larger than individual characters but smaller than complete words:
- **Common words** remain intact as single tokens
- **Less common words** are split into subword pieces that the model can compose to understand the full word

Consider the word $\text{"unhappiness"}$. A subword tokenizer might break this into:
- $\text{"un"}$ - a common prefix meaning $\text{"not"}$
- $\text{"happy"}$ - a complete word
- $\text{"ness"}$ - a common suffix that turns adjectives into nouns

By learning embeddings for these subword pieces, the model can **understand new words by composition**. Even if it never saw $\text{"unhappiness"}$ during training, if it learned embeddings for $\text{"un"}$, $\text{"happy"}$, and $\text{"ness"}$ from other contexts, it can compose these to understand the meaning of $\text{"unhappiness"}$.


This approach gives us several crucial benefits. First, we can control the vocabulary size to find a **sweet spot**:
- **Large enough** to represent common words efficiently
- **Small enough** to be manageable

**Typical vocabulary sizes** for modern models range from **30,000 to 50,000 tokens**. This is:
- Much smaller than the hundreds of thousands of words in a language
- Much larger than the hundred or so characters needed for character-level tokenization

With careful choice of subwords, this vocabulary size is sufficient to represent any text efficiently.


Second, subword tokenization **eliminates the unknown word problem**. Even if a word never appeared in the training data, it can be represented as a sequence of subword tokens.

**Example**: The word $\text{"antidisestablishmentarianism"}$ might be broken into pieces like:
- $\text{"anti"}$, $\text{"dis"}$, $\text{"establish"}$, $\text{"ment"}$, $\text{"ariam"}$, $\text{"ism"}$

All of which are common morphemes that likely appeared in many other words during training. The model can understand the unfamiliar word by **composing the meanings of its familiar pieces**, similar to how a human might puzzle out the meaning of a long word by recognizing its morphological components.


Third, subword tokenization naturally captures morphological relationships. Words like $\text{"run"}$, $\text{"running"}$, $\text{"runner"}$, and $\text{"runs"}$ all contain the subword $\text{"run"}$, allowing the model to learn that these words are related. Prefixes like $\text{"un"}$, $\text{"re"}$, $\text{"pre"}$, $\text{"dis"}$ and suffixes like $\text{"ing"}$, $\text{"ed"}$, $\text{"ness"}$, $\text{"tion"}$ can be learned as meaningful units, and the model can learn their systematic effects on word meaning. This makes learning **more efficient** and allows better generalization to new words formed by familiar morphological processes.

## Byte Pair Encoding

The most widely used subword tokenization algorithm is **Byte Pair Encoding (BPE)**. This algorithm was originally developed for data compression but was adapted brilliantly for natural language processing. The beauty of BPE is that it's **completely data-driven** - it learns what subwords are useful by analyzing actual text, rather than relying on linguistic knowledge about morphemes or word structure.

BPE starts with a vocabulary containing individual characters. For English text, this initial vocabulary includes:
- All letters (uppercase and lowercase)
- Digits
- Punctuation marks
- Any special characters that appear in the text

At this starting point, every word is represented as a sequence of character tokens. The word $\text{"lower"}$ would be represented as the five character tokens $\text{"l"}$, $\text{"o"}$, $\text{"w"}$, $\text{"e"}$, $\text{"r"}$, typically with a special end-of-word symbol appended to mark word boundaries.

**The iterative merging process**:
1. Find the most frequent pair of consecutive tokens in the training corpus
2. Merge them into a single new token
3. Add this new token to the vocabulary
4. Replace all occurrences of that pair in the corpus with the new merged token
5. Repeat until the vocabulary reaches the desired size

Let's walk through this process with a concrete example. Imagine we're training a BPE tokenizer on a tiny corpus containing the words $\text{"low"}$, $\text{"lower"}$, $\text{"lowest"}$, $\text{"newer"}$, $\text{"wider"}$, each appearing with some frequency. We represent each word with its character plus an end-of-word marker, which I'll write as $\text{"\\_"}$ for clarity: 

**Initial corpus**: $\text{"low\\_"}$, $\text{"lower\\_"}$, $\text{"lowest\\_"}$, $\text{"newer\\_"}$, $\text{"wider\\_"}$

**Iteration 1**: Count adjacent token pairs
- Pair $\text{"e r"}$ appears in $\text{"lower"}$, $\text{"newer"}$, and $\text{"wider"}$
- Pair $\text{"lo"}$ appears in $\text{"low"}$, $\text{"lower"}$, and $\text{"lowest"}$
- Let's say $\text{"e r"}$ is most frequent

We merge $\text{"e r"}$ into a new token $\text{"er"}$ and update our corpus:
- $\text{"lower"}$ becomes $\text{"l o w er \\_"}$
- $\text{"newer"}$ becomes $\text{"n e w er \\_"}$
- $\text{"wider"}$ becomes $\text{"w i d er \\_"}$

**Iteration 2**: Find the next most frequent pair
- Perhaps $\text{"er\\_"}$ (the $\text{"er"}$ token followed by the end-of-word marker) is now most frequent
- We merge this into $\text{"er\\_"}$, representing the common suffix

Our words become: $\text{"l o w er \\_"}$, $\text{"n e w er \\_"}$, $\text{"w i d er \\_"}$

**Continuing iterations**:
- Next merging $\text{"l o"}$ into $\text{"lo"}$
- Then $\text{"lo w"}$ into $\text{"low"}$
- And so on


After many iterations, our vocabulary contains not just individual characters but also **common subwords discovered by the frequency-based merging process**:
- **Common complete words** like $\text{"the"}$, $\text{"a"}$, $\text{"is"}$ will likely end up as single tokens because all their characters get merged together early (since these words appear so frequently)
- **Less common words** will be split into pieces, with split points generally corresponding to natural morpheme boundaries (because morphemes are reused across many words)

The key insight that makes BPE work so well is that it automatically discovers meaningful subwords **without any linguistic knowledge**:
- If $\text{"ing"}$ appears at the end of many words, those $\text{"i"}$, $\text{"n"}$, $\text{"g"}$ sequences will be frequently adjacent and get merged together early, creating an $\text{"ing"}$ token
- If $\text{"un"}$ appears at the beginning of many words, it will be merged into a token
- The algorithm discovers these patterns purely from frequencies in the data, but because language has systematic morphological structure, the discovered patterns correspond to linguistically meaningful units


When we want to tokenize new text with a trained BPE tokenizer, we apply the learned merge operations **in the same order they were learned during training**: 

1. Start with the character-level representation of each word
2. Apply each merge rule in sequence. 

Example, for the word $\text{"newer"}$: 
- Start with $\text{"n e w e r"}$
- Apply the merge creating $\text{"er"}$: $\text{"n e w er "}$ 
- Apply any other relevant merges. 

The final tokenization depends on what merges were learned during training and what order they were learned in.

One particularly clever variant of BPE is **byte-level BPE**, which is used by models like GPT-2 and GPT-3. Instead of starting with characters as the base vocabulary, byte-level BPE start with **bytes** - the raw UTF-8 byte encoding of text. 

Since any text in any language can be represented as a sequence of bytes from the range 0-255, this makes the tokenizer **completely universal**: 
- Can handle English, Chinese, Arabic, emoji, mathematical symbols or any other content without special handling. 
- Treats everything as sequences of 256 possible byte values 
- Learns to merge frequent byte sequences, exactly like standard BPE but operating at the byte level instead of the character level.

**Advantages of byte-level BPE**:
- **No need for different tokenizers** for different languages or writing systems
- **No special preprocessing** to handle unusual characters or symbols
- **Universal representation**: Everything is just bytes, and the merge operations learn whatever patterns are frequent in the training data
- **Truly multilingual**: Makes it much easier to build models that can handle the full diversity of text found on the internet

Byte-level BPE has become increasingly popular because of this universality, enabling modern LLMs to work seamlessly across languages and character sets.


## WordPiece and Other Approaches

While BPE is the most widely used subword tokenization algorithm, other approaches exist with slightly different design choices. 

**WordPiece** is very similar to BPE but uses a different criterion for choosing which pairs to merge. Instead of simply picking the most frequent pair, WordPiece chooses the pair that, when merged, **maximizes the likelihood of the training data** according to a language model.

The intuition behind WordPiece's criterion is that we want to create tokens that are **predictive** - tokens that help us predict the surrounding context. Simply merging the most frequent pair might not always be the best choice if that pair doesn't have consistent meaning or usage patterns. By considering likelihood, WordPiece tries to find merges that create more coherent and meaningful tokens. 

In practice, BPE and WordPiece often produce similar vocabularies, but the likelihood-based criterion can lead to better tokenizations in some cases.

WordPiece also uses a **special convention** where subword tokens that don't begin a word are marked with a prefix, typically $$\\#\\#$$. So the word $\text{"unhappiness"}$ might be tokenized as 
- $\text{"un"}$, $$\text{"\\#\\#happiness"}$$ or 
- $\text{"un"}$, $$\text{"\\#\\#happy"}$$, $$\text{"\\#\\#ness"}$$

The $$\\#\\#$$ markers make it immediately clear which tokens are word beginnings and which are continuations. This can be useful for downstream tasks where we need to know word boundaries, such as **named entity recognition** where we need to identify which tokens are part of the same entity.

Another approach is the **Unigram language model tokenization algorithm**, which takes a fundamentally different strategy - Instead of starting with characters and building up by merging:
1. **Starts large**: Begins with a large vocabulary of possible subwords that appear in the corpus with sufficient frequency, creating a very large initial vocabulary
2. **Prunes iteratively**: Iteratively removes tokens that, when removed, cause the smallest increase in loss according to a unigram language model fit to the corpus

The Unigram approach has some **theoretical advantages**. BPE is a **greedy algorithm** that makes locally optimal choices at each step - merge the most frequent pair - but these choices might not lead to a globally optimal vocabulary. Unigram tries to find a better solution by **considering the global effect** of including or excluding each token. It asks: *Given all the other tokens in the vocabulary, does this particular token help us encode the corpus efficiently?* 

If removing a token doesn't hurt much because the words it appeared in can be efficiently encoded using other tokens then that token can be pruned.


A practical implementation of these subword tokenization approaches is provided by the **SentencePiece library**, which has become widely used in modern natural language processing.

**Key features**:
- **Multiple algorithms**: Implements both BPE and Unigram algorithms with several useful enhancements
- **Language-agnostic design**: Treats spaces as regular characters, encoding them explicitly rather than treating them as special delimiters
  - This seemingly small choice makes the tokenizer truly language-agnostic
  - Can handle languages that don't use spaces to separate words (like Chinese or Japanese) using exactly the same algorithm as languages that do use spaces

**Training a SentencePiece model**:
1. Provide a corpus of raw text
2. Specify the desired vocabulary size and algorithm choice
3. The system samples from the corpus, runs the training algorithm
4. Produces a model file containing the learned vocabulary and merge rules

This model can then be used to consistently tokenize any new text.

**For multilingual models**: You would train SentencePiece on a mixed corpus containing text in all the target languages, and it would learn a **unified vocabulary** that efficiently represents common patterns across all those languages.

## Practical Considerations and Challenges

Choosing the right vocabulary size for a tokenizer involves balancing multiple competing considerations.

**Smaller vocabularies** (e.g., 8,000 tokens):
- Fewer parameters in the embedding layer
- More compact overall model
- However: Words get split into more pieces, producing longer sequences
- Slower generation for autoregressive models
- Fewer dedicated tokens for specialized vocabulary

**Larger vocabularies** (e.g., 100,000 tokens):
- Keeps more words intact as single tokens
- Produces shorter sequences
- Better for rare or technical terms
- However: Much larger embedding matrix
- Many rare tokens with poorly-learned embeddings

**The standard compromise**: Most modern large language models use vocabularies between **32,000 and 50,000 tokens**:
- Large enough that most common words are single tokens (keeping sequences relatively short)
- Small enough that the vocabulary is manageable and most tokens appear frequently enough during training to learn good embeddings
- **For multilingual models**: Vocabularies tend to be larger (100,000 to 250,000 tokens) to accommodate common words and patterns across many languages

One fascinating challenge is handling languages that don't use spaces to separate words. In Chinese, sentences are written as continuous strings: "我喜欢学习自然语言处理" means "I like studying natural language processing ", but there are no visual breaks between words.

Subword tokenization approaches like BPE handle this effectively when trained on sufficient Chinese text. The algorithm learns to merge frequently co-occurring characters, discovering common multi-character words. If "自然语言" (natural language) appears frequently, those characters merge together. The tokenizer discovers these patterns purely from frequency, yet the learned tokens often correspond to meaningful linguistic units.

Similar challenges arise with Japanese (which mixes hiragana, katakana, and kanji), Thai and Southeast Asian languages (no spaces), and Arabic (right-to-left with complex connection rules). Subword tokenization methods have proven robust across this diversity.

**Numbers**: Should "2023" be one token, "20" + "23", or four digits? Different tokenizers make different choices. Keeping numbers as atomic units works well for most applications, but digit-level representation might help models learn arithmetic. Many tokenizers keep short numbers together but split longer ones, leading to inconsistent behavior.

**Punctuation**: Should "don't" be one token or split into "don" + "'t"? What about "it's"? These choices affect how models learn contractions and possessives. The handling of quotation marks and paired punctuation also matters for certain tasks.

A tokenizer trained primarily on English text represents English efficiently, with common words as single tokens. But text in other languages, especially with different scripts, can be much less efficient. Chinese text might split into many small tokens because Chinese characters are rare in English-heavy training data. This means the same content takes more tokens in Chinese than English, hitting context length limits faster and increasing inference costs.

This motivates training multilingual tokenizers on balanced corpora with substantial representation of many languages. However, true balance is difficult - higher-resource languages like English often get more efficient tokenization simply because there's more training data available.


## The Impact of Tokenization on Model Behavior


The way text is tokenized has profound but often subtle effects on model behavior. These effects aren't always obvious but understanding them is important for building and using language models effectively.

One fundamental impact is that **tokenization determines what the model sees as atomic units**. The model learns embeddings for tokens and learns patterns of how tokens relate to each other, but it doesn't have direct access to the sub-token structure.

For example, if the word $\text{"unhappy"}$ is always tokenized as a single token, the model must learn its meaning as an atomic unit. It might learn that $\text{"unhappy"}$ is similar to $\text{"sad"}$ and opposite of $\text{"happy"}$ purely from the contexts in which these words appear. But if $\text{"unhappy"}$ is tokenized as $\text{"un"}$ and $\text{"happy"}$, the model can potentially leverage its knowledge that $\text{"un"}$ typically negates the following word. If it knows $\text{"happy"}$ means positive emotion and it knows $\text{"un"}$ typically means negation, it can more easily understand that $\text{"unhappy"}$ means negative emotion, even if it never saw that specific combination during training.

This compositional understanding is one of the **key benefits of subword tokenization**. The model can generalize to new words formed by familiar morphological processes. 

If the model has learned embeddings for $\text{"care"}$ and $\text{"less"}$ and $\text{"ness"}$, it can make educated guesses about $\text{"carelessness"}$ even if that exact word never appeared in the training data. This isn't perfect - the model might not fully capture the subtle meaning shifts that happen in word formation - but it's much better than treating rare words as completely unknown.

Tokenization affects how models handle different types of tasks:
- **Spelling correction**: Character-level or fine-grained subword tokenization works better because the model needs to reason about individual characters
- **Word meaning tasks**: Coarser tokenization that keeps whole words intact might be more effective

The optimal granularity depends on what the model needs to learn for its intended tasks.

One particularly interesting phenomenon is how tokenization affects the model's ability to do tasks that **require careful character-level reasoning**.
**Example**: Consider the task "count the number of $\text{'e'}$ characters in the word $\text{'serendipity'}$":
- **If $\text{"serendipity"}$ is a single token or large subword chunks**: The model never sees the individual characters and must try to answer based on having memorized something about this word during training
- **If tokenized character by character**: The model can actually count the $\text{'e'}$ characters in its input

In practice, most models struggle with tasks requiring precise character-level reasoning, partly because their tokenization operates at a coarser grain.

Debugging tokenization issues can be an important part of working with language models. Sometimes a model produces surprising or incorrect outputs because of how the input was tokenized.

**Common issues**:
- A technical term is being split in an unexpected way, with each piece getting interpreted separately in a manner that distorts the overall meaning
- A foreign name is being broken into pieces that coincidentally match common words in the model's training language, leading to confusion

Looking at the actual tokenization can often reveal why the model behaved unexpectedly.

Modern tools make it easy to visualize tokenization. Libraries like Hugging Face provide functions to show exactly how text gets broken into tokens and what integer IDs those tokens map to. When debugging model behavior, it's often worth checking the tokenization to verify it matches your expectations. If a model seems to struggle with particular inputs, examining how those inputs are tokenized might reveal that they're being split in unusual ways that make the model's task harder.


## The Future of Tokenization


While subword tokenization has proven very effective and has become the standard approach for essentially all modern large language models, research continues into whether we can do better or whether we even need tokenization as a separate preprocessing step. 

Some experimental models work directly on **bytes or even raw audio**, learning to segment and structure the input as part of the neural network training rather than using a fixed preprocessing pipeline.

These tokenization-free approaches are appealing because:
- More flexible and truly language-agnostic
- A model working directly on bytes doesn't need to know anything about the structure of different languages or writing systems
- Can handle any text, in any language, with any special characters, all with the same algorithm
- The model learns to discover meaningful units automatically during training rather than having those units imposed by a separate tokenization algorithm


However, tokenization-free models face significant challenges. Working directly on bytes means dealing with **very long sequences**, which is computationally expensive in transformer architectures. Some research has explored **hierarchical approaches** where the model first processes bytes at a fine granularity, then builds up to larger units, then builds up to even larger units. This allows the model to capture both fine-grained patterns and long-range dependencies without quadratic cost in the full sequence length. These hierarchical architectures show promise but are still largely experimental.

For now, **subword tokenization remains the practical standard**. It provides a good balance of:
- Efficiency
- Flexibility
- Effectiveness

By learning data-driven vocabularies that capture common patterns at an appropriate level of granularity, subword methods allow models to handle diverse text efficiently while avoiding the worst problems of both word-level and character-level approaches.

As models continue to grow and evolve, tokenization will undoubtedly continue to be refined and improved, but the fundamental insight - that **working with learned subword units provides the right level of abstraction for language models** - seems likely to remain central to how these systems work.



<div style="display: flex; justify-content: space-between;">
  <a href="LLMArchitectureOverview.md">Back</a>
  <a href="AttentionMechanism.md">Next</a>
</div>