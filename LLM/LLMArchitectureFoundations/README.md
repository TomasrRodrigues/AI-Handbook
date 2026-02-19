# Large Language Models: Architectural Foundations

#### Table of Contents

1. [LLM Architecture Overview](#llm-architecture-overview)
2. [Tokenization](#tokenization)
3. [Attention Mechanisms](#attention-mechanisms)
4. [Sampling Techniques](#sampling-techniques)

---
## LLM Architecture Overview


**Large Language Models** have fundamentally transformed how we interact with artificial intelligence and how machines process human language. To truly understand how these remarkable systems work, we need to explore two fundamental aspects that form the foundation of every large language model: **their underlying architecture** and **the critical process of tokenization**.

When we talk about large language models, we are referring to AI systems that have been specifically designed to process and generate human-like text. These models differ from traditional natural language processing approaches in several key ways:

- **Scale**: They contain billions or even trillions of parameters
- **Training methodology**: They learn from massive amounts of text data
- **Versatility**: They can perform multiple tasks without task-specific training

Unlike older systems that required explicit programming for each specific task, large language models **learn patterns and relationships in language** by analyzing massive amounts of text data, developing an internal understanding that allows them to perform a wide variety of tasks without task-specific training.

### Understanding Large Language Model Architecture

The term **"large"** in LLM refers to several interconnected characteristics that define these systems. First and most obviously, these models contain an **enormous number of parameters** - the internal numerical values that the model adjusts during training to learn patterns in data. Modern large language models can have billions or even trillions of these parameters.

To put this in perspective, the human brain contains approximately 100 billion neurons with around 100 trillion connections between the,. The largest language models are approaching similar scales of complexity, though they work in fundamentally different ways than biological brains.

The size of these models is not arbitrary. Researchers have discovered that as models grow larger, they exhibit what are called **"emergent capabilities"** - abilities that only appear once the model reaches a certain scale:

- **Small models** might struggle with basic grammatical tasks
- **Medium-sized models** can handle simple language understanding
- **Truly large models** demonstrate sophisticated reasoning, can follow complex instructions, and can learn new tasks from just a few examples

This relationship between size and capability means that building more powerful language models often requires making them larger, though recent research has also focused on making models more efficient so they can do more with fewer parameters.

Beyond just the number of parameters, these models are **"large" in terms of the data they consume during training**. Training a modern large language model requires processing hundreds of billions or even trillions of words of text. This training data comes from diverse sources including books, websites, scientific papers, and other written materials. 

The model learns by repeatedly trying to **predict what comes next** in text sequences, gradually adjusting its internal parameters to become better at this prediction task. Through this seemingly simple objective, the model develops sophisticated understanding of grammar, facts about the world, reasoning patterns, and much more.

### How LLMs Process Language

At the most fundamental level, large language models face a challenge that might seem simple but is actually quite profound: **computers work with numbers** (more specifically, binary), but **language is made of words and sentences**. To bridge this gap, language models must convert text into numerical representations that can be processed by mathematical operations. This conversion happens through a **multi-stage process** that transforms raw text into rich numerical representations that capture meaning, context and relationships. 

#### From Text to Tokens

The first step in this process is **tokenization**. We will cover this topic later, but for now we only need to understand that it breaks text into smaller units called **tokens**: 

$$
\text{"Some text about LLMs"} \quad \xrightarrow{tokenization} \quad [\text{"Some"}, \text{"Text"},\text{"About"},\text{"LLMs"}]
$$

These tokens might be whole words, parts of words, bytes, etc. depending on the tokenization approach being used. 

#### From Tokens to Embeddings

Once text has been broken into tokens, each token is converted into a **dense vector of numbers** called an **embedding**. These embeddings are not arbitrary - they are learned during training so that tokens with similar meanings end up with similar numerical representations.

Consider this example. If we represented the word $\text{"cat"}$ with one set of numbers and $\text{"dog"}$ with another set, we want those number sets to be relatively similar as they are similar (both animals). On the other hand, the numbers representing these words should be fundamentally different from $\text{"justice"}$ (for example), as the concepts are fundamentally different. 

The model learns these relationships automatically during training by observing how words are used in context. Words that appear in similar contexts end up with similar embeddings.


Once tokens have been converted to embeddings, the model must account for the **order of words**. This is crucial because $\text{"the dog chased the cat"}$ means something different from $\text{"the cat chased the dog"}$ even though both sentences have the exact same words. 

To encode position information, transformers add **positional encodings** to the embeddings. These positional encodings are carefully designed patterns of numbers that give each position in a sequence a unique signature. By adding these patterns to the word embeddings, the model can distinguish between the same word appearing in different positions.


### The Transformer Revolution

The architectural breakthrough that enabled modern large language models was the invention of the **transformer** in 2017. Before transformers, most language models used **recurrent neural networks (RNNs)**, which processed text sequentially one word at a time. While this sequential processing seemed natural - after all, we read and speak one word at a time - it created serious limitations:
- Processing had to happen in strict order, preventing parallelization across the sequence
- Training was very slow due to sequential dependencies
- Information from early in a sequence had to pass through many computational steps to reach later parts, gradually degrading or becoming distorted along the way

#### The Attention Mechanism

The transformer architecture solved these problems through a mechanism called **attention**. To understand attention, imagine we are reading a sentence like $\text{"The trophy is big because it is not small."}$ When we read the word $\text{"it"}$, we automatically associate $\text{"it"}$ with $\text{"trophy"}$ by looking at the earlier part of the sentence. The attention mechanism allows transformer models to do something similar - when processing each word, the model can look at all other words in the sequence and determine which ones are most relevant for understanding the current word.

**The mathematics behind attention** is elegant. For each word in a sequence, the model creates three different representations:
- **Query** - Represents what information the word is looking for
- **Key** - Represents what information the word has to offer
- **Value** - The actual information content

To compute attention, the model compares the query from one word with the keys from all other words. These scores are then used to take a weighted combination of the values, producing an output that incorporates relevant information from across the entire sequence.

#### Why Attention Is Powerful

What makes this mechanism so powerful is that it allows **every word to directly interact with every other word** in a single computational step. There is no sequential processing, no gradual degradation of information through many intermediate steps. This brings two major advantages:
- **Faster training**: Transformers can be trained much faster than recurrent models because all words can be processed in parallel
- **Better long-range dependencies**: Transformers capture long-range relationships more effectively because the path length between any two words is constant rather than growing with their distance

#### Multi-Head Attention

Modern transformers don't use just one attention mechanism. Instead, they use **multiple attention "heads" operating in parallel**. Each head learns to capture different types of relationships between words:
- One head might specialize in linking pronouns to the nouns they refer to
- Another might focus on identifying syntactic relationships like subject-verb agreement
- Another might track semantic relationships between conceptually related words

By combining information from multiple attention heads, the model builds a **rich, multifaceted understanding** of the text. 

### The Complete Transformer Architecture

A complete transformer consists of **multiple identical layers stacked on top of each other**. Each layer contains two main components:
- **Multi-Head Attention Mechanism** - Allows words to exchange information with each other
- **Feed-Forward Neural Network** - Allows each word's representation to be independently transformed and refined. 

Around these core components are several important technical details that make training stable and effective.

#### Residual Connections: The Highway for Gradients

One crucial detail is the use of **residual connections**. In a residual connection, the input to a component is added to its output. This might seem pointless at first - why add the input to the output instead of just using the output? 

The answer has to do with how neural networks are trained. Training involves computing **gradients** that indicate how to adjust each parameter to improve performance and these gradients must flow backward through the network. In very deep networks, gradients can become vanishingly small as they propagate back through many layers, making it hard to train the early layers effectively. **Residual connections provide shortcut paths** that allow gradients to flow more easily, enabling the training of much deeper networks.

#### Layer Normalization: Keeping Values in Check

Another important detail is **layer normalization**, which standardizes the representations at various points in the network. Without normalization, the scale of values can grow or shrink through the layers in ways that make training unstable. Layer normalization ensures that the values stay in a reasonable range, which leads to more stable and faster training. 

The exact placement of normalization layers has been studied extensively, with modern architectures typically using what's called **pre-norm**, where normalization happens before each sub-layer rather than after.

#### The Role of Feed-Forward Networks

The feed-forward network in each transformer layer might seem simple compared to the complex attention mechanism, but it plays a **crucial role**. This network consists of two linear transformations with a non-linear activation function in between. Despite its conceptual simplicity, the feed-forward network often contains **more parameters than the attention mechanism** and is essential for the model's ability to learn complex transformations of the linguistic representations.

You can think of the attention mechanism as allowing information to **move around and interact**, while the feed-forward network allows that information to be **transformed and refined**.

### Different Architectural Approaches

While the basic transformer architecture is remarkably versatile, researchers have developed **several important variants** optimized for different types of tasks. The original transformer architecture consisted of two parts: an **encoder** that processes the input text and a **decoder** that generates output text. This encoder-decoder structure works well for tasks like translation where we need to read in one piece of text and produce another.

However, for many tasks, only one part of this architecture is necessary. This led to the development of three main architectural families: 

#### Encoder-Only Models (BERT)

**BERT** (Bidirectional Encoder Representations from Transformers) uses only the encoder part of the transformer and is trained by randomly masking some words in the input and asking the model to predict what the masked words were. This training approach allows BERT to build **bidirectional representations** where each word's meaning is informed by both the words that come before it and the words that come after it. 

BERT excels at **understanding tasks** like:
- Question answering
- Text classification
- Named entity recognition

The goal in these tasks is to analyze or extract information from existing text rather than generate new text.

#### Decoder-Only Models (GPT)

On the other end of the spectrum are **decoder-only models** like the GPT family. These models use only the decoder part of the transformer and are trained **autoregressively** - that is, they learn to predict the next word given all previous words. When generating text, they produce one word at a time, with each newly generated word being fed back as input for generating the next word. 

This architecture makes GPT models particularly good at **text generation tasks**. They can complete prompts, answer questions, write code, and perform many other tasks through clever prompting, even though they were only trained on the simple objective of predicting the next word.

#### Encoder-Decoder Models (T5)

The encoder-decoder architecture is preserved in models like **T5** (Text-to-Text Transfer Transformer), which frames every natural language processing task as a **text-to-text problem**. Whether the task is translation, summarization, question answering, or classification, T5 treats it as taking input text and producing output text. This unified framework allows a single model to handle many different tasks, and the encoder-decoder structure provides the flexibility to transform input text of one form into output text of potentially very different form.

#### Comparing the Architectures

Each architectural variant has its strengths:

- **Encoder-only models** can use context from both directions, making them powerful for understanding tasks
- **Decoder-only models** are naturally suited for generation and have shown remarkable versatility through in-context learning, where they can learn new tasks from examples provided in the prompt
- **Encoder-decoder models** offer maximum flexibility but require more computation

The choice of architecture depends on the intended application and the types of tasks the model needs to perform.


### Training Large Language Models

Training a Large Language Model is one of the most computationally intensive tasks in modern artificial intelligence. The process typically unfolds in two main phases: **pretraining** and **fine-tuning**. 

During pretraining, the model learns general language understanding from enormous amounts of text. This phase requires **massive computational resources** and can take weeks or months even on hundres or thousands of powerful processors. The goal of pretraining is to give the model broad knowledge about language, reasoning and the world, which it can then apply to many different tasks.

***The pretraining process is fundamentally about prediction***. For a decoder-only model like GPT, pretraining involves showing the model sequences of text and training it to predict what comes next. If the model sees $\text{"The cat sat on the"}$, it should predict something like $\text{"mat"}$ or $\text{"chair"}$ or another plausible continuation. For encoder-only models like BERT, pretraining involves masking out random words and training the model to predict what the masked words were. Through billions of these prediction tasks, the model gradually learns patterns in language - grammar rules, common phrases, factual relationships, reasoning patterns and much more.

***The data used for pretraining is critically important***. Modern large language models are trained on datasets containing hundreds of billions or trillions of words, drawn from books, websites, scientific papers and other text sources. The quality and diversity of this data directly impacts what the model learns. If the training data contains biased viewpoints, factual errors or problematic content, the model will learn and potentially reproduce these issues. Careful curation and filtering of training data is therefore essential, though it remains a challenging problem to ensure training data is both large enough and high quality enough.

During training, the model starts with **randomly initialized parameters** and gradually adjusts them to improve its predictions. This adjustment happens through a process called **backpropagation with gradient descent**:

1. The model makes a prediction
2. A **loss function** measures how wrong the prediction was
3. The system computes **gradients** that indicate how each parameter should be adjusted to reduce the loss
4. The parameters are updated in small steps in the direction that reduces the loss
5. This process repeats millions of times with different batches of data

Gradually, the model's parameters converge to values that make good predictions across a wide range of text.

Training stability is a major concern for very large models. Various technical problems can derail training:
- **Exploding gradients**: Parameter updates become so large they destroy previously learned patterns
- **Vanishing gradients**: Updates become so small that learning slows to a crawl


Modern training techniques to ensure stability include:
- Careful initialization of parameters
- Gradient clipping to cap the maximum size of updates
- Learning rate schedules that adjust how quickly parameters change
- Architectural choices like residual connections and layer normalization

After pretraining creates a model with broad language understanding, **fine-tuning** adapts it for specific tasks or domains. Fine-tuning uses **much smaller amounts of data** - thousands or millions of examples rather than billions - and takes much less time than pretraining. During fine-tuning, the model's parameters are adjusted to specialize it for a particular application. For instance, a model might be fine-tuned on medical literature to improve its performance on medical questions, or on code repositories to enhance its programming capabilities, or on conversational data to make it better at dialogue.

**Types of fine-tuning:**

- **Supervised fine-tuning**: The model is trained on input-output pairs specific to a task. For sentiment analysis, we provide sentences paired with sentiment labels. For question answering, we provide questions and contexts paired with answers.

- **Instruction tuning**: The model is fine-tuned on a diverse set of instructions, even for tasks it hasn't explicitly seen. This has become increasingly popular in recent years.

- **Reinforcement Learning from Human Feedback (RLHF)**: This approach was crucial in developing conversational AI systems like ChatGPT. Humans provide feedback on model outputs, comparing different responses and indicating which are better. This feedback trains a reward model that predicts how a human would evaluate any given response. Then the language model is further trained using reinforcement learning to generate responses that score highly according to the reward model. This process helps align the model's behavior with human preferences, making it more helpful, harmless, and honest in its responses.

### Model Scaling and Computational Requirements

The scale of modern Large Language Models is difficult to overstate. Training the largest models requires computational resources that only major technology companies and research labs can afford. Training GPT-3 from scratch, for example, would cost **millions of dollars** in computing resources. The process requires distributing the computation across hundreds or thousands of specialized processors, typically **graphics processing units (GPUs)** or **tensor processing units (TPUs)** that are optimized for the matric operations that neural networks perform.

Several forms of parallelism are used to train these massive models:

- **Data parallelism**: Distributes different batches of training data to different processors, each running a copy of the model. The gradients computed by each processor are then averaged to update the shared model parameters.

- **Model parallelism**: Splits the model itself across multiple processors when it's too large to fit on a single device. Different layers or different parts of layers live on different processors, and activations are passed between processors as needed.

- **Pipeline parallelism**: Divides the model into stages and processes multiple batches simultaneously in a pipeline fashion, with each stage working on a different batch.

These computational requirements mean that **most organizations do not train foundation models from scratch**. Instead, they use existing pretrained models in two main ways:

- Through **APIs** provided by companies like OpenAI and Anthropic
- By downloading **open-weight models** from organizations like Meta and using them directly or fine-tuning them for specific needs

This democratization of access to large language models, where the most expensive pretraining is done once by well-resourced organizations and then made available to others, has been crucial for the widespread adoption of these technologies.

Researchers have also discovered interesting patterns in how model performance scales with size. There appear to be **power law relationships** between model size, amount of training data, amount of computation, and resulting performance: 

- Doubling the model size tends to produce a **predictable improvement** in performance
- Doubling the amount of training data or computation has similar effects
- These scaling laws have guided decisions about how to allocate resources when training models

More recently, research has shown that many models have been **under-trained** - they had plenty of parameters but didn't see enough training data. The optimal approach seems to be to **scale up both model size and training data in a balanced way**.

As models have grown larger, they have exhibited **emergent capabilities** that don't appear in smaller models. For example:

- A small model might not be able to do basic arithmetic reliably
- A sufficiently large model suddenly becomes quite competent at it, even though it wasn't explicitly trained on arithmetic

This emergence of new capabilities at scale has been one of the most surprising and exciting findings in large language model research, though it's not yet fully understood why this happens or how to predict what capabilities will emerge at what scales.


### Real-World Applications

Large language models have found applications across virtually every domain that involves text, transforming how professionals work and how services are delivered.

#### Healthcare Applications

In healthcare, these models assist doctors by:
- Summarizing patient records
- Helping interpret medical literature
- Suggesting potential diagnoses based on symptoms

The ability to quickly process and synthesize information from vast medical databases can help healthcare providers make more informed decisions, though these systems are **tools to assist human judgment rather than replace it**.

#### Legal Industry

In the legal field, large language models help lawyers **review contracts, research case law, and draft legal documents**. The ability to quickly analyze hundreds of contracts to find specific clauses, or to search through decades of legal precedents to find relevant cases, can save enormous amounts of time and ensure more thorough analysis. The models can also help make legal services more accessible by providing preliminary guidance to individuals who might not otherwise be able to afford legal consultation.

#### Education and Learning

Education has been transformed by these models, which can serve as **tutoring systems that adapt to individual students**. Unlike traditional educational software that follows rigid scripts, large language models can engage in natural dialogue with students, explaining concepts in multiple ways, generating practice problems, and providing personalized feedback. A model can recognize when a student is struggling with a particular concept and try different explanatory approaches until something clicks, much like a human tutor would.

#### Business Applications

The business world uses these models for:
- **Customer service**: Chatbots that handle complex inquiries, understanding context from previous messages and providing thoughtful, relevant responses
- **Content creation**: Generating marketing materials and analyzing customer feedback
- **Data analysis**: Processing and summarizing large amounts of business data
- **Code development**: Writing, explaining, and debugging code

#### Programming and Software Development

One particularly impactful application has been in **programming and software development**. Large language models trained on code can:
- Understand multiple programming languages
- Explain what existing code does
- Generate new code from natural language descriptions
- Suggest fixes for bugs

A programmer can describe what they want a function to do in plain English, and the model can generate working code that implements that functionality. While the generated code isn't always perfect and needs human review, this capability dramatically speeds up development and makes programming more accessible to people who are learning.

### Understanding the Limitations

Despite these impressive capabilities, it's crucial to understand the **limitations of large language models**.

These systems **don't truly understand language the way humans do** - they are sophisticated pattern matching systems that have learned statistical relationships between words and concepts. This means they can sometimes generate text that sounds confident and fluent but is factually incorrect. 

**Hallucinations**: Researchers call these errors "hallucinations" - the model generates plausible-sounding information that isn't actually true. The model might:
- Cite a scientific paper that doesn't exist
- Attribute a quote to the wrong person
- Invent facts that sound plausible but are incorrect

This happens because models learn patterns about how information is formatted without having access to a reliable database of facts.


Large language models struggle with certain types of reasoning that humans find relatively straightforward:

- **Mathematical calculations**: Difficulty with precise calculations, especially with larger numbers
- **Multi-step logical reasoning**: Can be challenging, though this has improved significantly with techniques like **chain-of-thought prompting** that encourage the model to break down its reasoning into explicit steps
- **Physical world understanding**: Models lack true understanding of the physical world - they know about physics only through textual descriptions, not through embodied experience, which can lead to mistakes in reasoning about spatial relationships or physical causality

Another important limitation is that these models **learn from their training data**, which means they can perpetuate and amplify biases present in that data. If the training data contains biased viewpoints about certain groups of people, the model will learn and may reproduce those biases. 

**Key considerations**:
- Significant research effort has gone into understanding and mitigating these biases, but it remains an ongoing challenge
- The models have **no inherent sense of ethics or values** - they generate content based on patterns in their training data
- They need **careful oversight and safety measures** to prevent misuse


---
## Tokenization

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


The question of how to tokenize text might seem to have an obvious answer: just split on spaces to separate words. This approach, called **word-level tokenization**, is intuitive and was used by many early natural language processing systems. For a sentence like $\text{"The cat sat on the mat."}$, word-level tokenization produces six word tokens plus punctuation:

$$
\text{"The cat sat on the mat."} \quad \xrightarrow{Tokenization} \quad \text{["The","cat","sat","on","the","mat","."]}
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


### Character-Level Tokenization

At the opposite extreme from word-level tokenization is **character-level tokenization**, where text is broken down into individual characters:

$$
\text{"CAT"} \quad \xrightarrow{Tokenization} \quad \text{["C","A","T"]}
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


### The Subword Solution

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

### Byte Pair Encoding

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


### WordPiece and Other Approaches

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

### Practical Considerations and Challenges

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

One fascinating challenge is handling languages that don't use spaces to separate words. In Chinese, sentences are written as continuous strings: "我喜欢学习自然语言处理" means "I like studying natural language processing," but there are no visual breaks between words.

Subword tokenization approaches like BPE handle this effectively when trained on sufficient Chinese text. The algorithm learns to merge frequently co-occurring characters, discovering common multi-character words. If "自然语言" (natural language) appears frequently, those characters merge together. The tokenizer discovers these patterns purely from frequency, yet the learned tokens often correspond to meaningful linguistic units.

Similar challenges arise with Japanese (which mixes hiragana, katakana, and kanji), Thai and Southeast Asian languages (no spaces), and Arabic (right-to-left with complex connection rules). Subword tokenization methods have proven robust across this diversity.

**Numbers**: Should "2023" be one token, "20" + "23", or four digits? Different tokenizers make different choices. Keeping numbers as atomic units works well for most applications, but digit-level representation might help models learn arithmetic. Many tokenizers keep short numbers together but split longer ones, leading to inconsistent behavior.

**Punctuation**: Should "don't" be one token or split into "don" + "'t"? What about "it's"? These choices affect how models learn contractions and possessives. The handling of quotation marks and paired punctuation also matters for certain tasks.

A tokenizer trained primarily on English text represents English efficiently, with common words as single tokens. But text in other languages, especially with different scripts, can be much less efficient. Chinese text might split into many small tokens because Chinese characters are rare in English-heavy training data. This means the same content takes more tokens in Chinese than English, hitting context length limits faster and increasing inference costs.

This motivates training multilingual tokenizers on balanced corpora with substantial representation of many languages. However, true balance is difficult - higher-resource languages like English often get more efficient tokenization simply because there's more training data available.


### The Impact of Tokenization on Model Behavior


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


### The Future of Tokenization


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



---
## Attention Mechanisms

Large language models have transformed artificial intelligence, and at the heart of this transformation lies a deceptively simple yet profound concept: the **attention mechanism**. When researchers published the landmark paper *"Attention Is All You Need"* in 2017, they introduce an architecture that would become the foundation for virtually every major advance in natural language processing over the following years. Understanding attention mechanisms is not merely an academic exercise - it is essential to comprehending how modern AI systems think, reason and generate language.

To appreciate the revolutionary nature of attention mechanisms, we must first understand the limitations of what came before. In the early 2010s, the dominant approach to sequence processing in natural language tasks relied on **recurrent neural networks**, particularly a variant called **Long Short-Term Memory networks (LSTMs)**. These architectures process text sequentially, reading one word at a time and maintaining an internal "memory" state that supposedly captures all the relevant information seen so far.

Imagine reading a complex legal document where a crucial term defined on page one becomes relevant again on page fifty. As we read, we must somehow compress and maintain all the important information from the intervening forty-nine pages in your working memory. This is essentially what recurrent networks were asked to do. 

**How RNNs work**:
1. The network processes the first word and creates a **hidden state** - a vector of numbers representing what it has learned
2. It processes the second word, updates its hidden state
3. Continues this pattern through the entire sequence
4. By the time it reaches word 10,000, that hidden state must somehow encode all the relevant information from all previous words

This sequential processing created several fundamental problems:
1. **Speed**
    - **Inherently slow**: Unlike operations that can be parallelized across modern graphics processors, recurrent networks must wait for step one to complete before starting step two
    - When training on documents with thousands of words, this sequential constraint made training prohibitively expensive and time-consuming
2. **Information Degradation**
    - More critically, information from early in the sequence tended to **degrade or disappear** as it passed through hundreds or thousands of sequential updates to the hidden state
    - Researchers call this the **"vanishing gradient problem"** and the **"information bottleneck problem"**


The attention mechanism solved these problems through a radical insight: **What if, instead of trying to compress all information into a single vector that must be maintained and updated sequentially, we allow the model to directly acecss any part of the input whenever it needs to?** 

When trying to understand word 10,000, the model can look back directly at word 1, or word 5,432 or any other word. This happens **without that information having to pass through a long chain of intermediate steps**. This direct access eliminates both the information bottleneck and the need for strictly sequential processing.

### Queries, Keys and Values

The mathematical formulation of attention draws an elegant analogy from database systems. When we query a database, we provide a search query, the database matches it against keys in its index and returns the corresponding values. The attention mechanism works similarly, though all three components - **queries, keys and values** - are learned representations rather than fixed data structures.

Consider the sentence **$\text{"The trophy doesn't fit in the suitcase because it is too big"}$**. 

When the model encouters the word $\text{"it"}$, it must determine what $\text{"it"}$ refers to. Does $\text{"it"}$ refer to the trophy or the suitcase? A human reader immediately recognizes that $\text{"it"}$ refers to the trophy. The attention mechanism allows the model to make this determination by computing how much $\text{"it"}$ should $\text{"attend to"}$ each previous word.

The process begins by transforming each word's representation into three different vectors: 
- **Query vector**: Represents what information the word is seeking
  - The query for $\text{"it"}$ might be thought of as asking $\text{"what noun am I referring to?"}$
- **Key vector**: Represents what information a word can provide
  - The key for $\text{"trophy"}$ essentially says $\text{"I am a concrete noun that could be a pronoun referent"}$
- **Value vector**: Contains the actual information that will be incorporated if a word is deemed relevant

#### How Attention is Computed

**Step 1: Compare queries with keys**

To compute attention, we compare the query vector from one word with the key vectors from all other words. This comparison is typically done using the **dot product** - a mathematical operation that measures how aligned or similar two vectors are: 
- When we compute the dot product between the query for $\text{"it"}$ and the key for $\text{"trophy"}$, we get a **high score** (they are semantically aligned - pronouns typically refer to nouns)
- When we compute the dot product between the query for $\text{"it"}$ and the key for $\text{"the"}$, we get a **low score** (determiners are rarely pronoun referents)

**Step 2: Convert scores to probabilities**

These dot product scores are then passed through a mathematical function called **softmax**, which converts them into a proper probability distribution - a set of positive numbers that sum to one. These probabilities are the **attention weights** and they tell us how much to weight each word when forming our new representation of $\text{"it"}$. 

**Step 3: Weighted average**

Finally, we take a wrighted average of the value vectors, where each word's value is weighted by its attention weight. The result is a **new, enriched representation of $\text{"it"}$** that incorporates relevant information from $\text{"trophy"}$ while largely ignoring irrelevant words.


The complete mathematical formulation, which has become one of the most famous equations in modern machine learning, is written as: 

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


Each component of this equation serves a specific purpose:
- **$QK^T$** (query-key dot product): Measures relevance
- **$\frac{1}{\sqrt{d_k}}$** (scaling factor): Prevents the dot products from growing too large, which would cause the softmax function to produce overly sharp probability distributions that essentially select just one word and ignore all others
- **softmax**: Converts raw scores into probabilities
- **Multiplication by $V$**: Weights the information from each word according to its relevance


### Multi-Head Attention

One of the key innovations is the transformer architecture was the realization that a single attention mechanism might not be sufficient to capture all the different types of relationships that exist in language. 

**The challenge**: When processing a word, we might simultaneously need to identify:
- Its syntactic role (is it a subject or object?)
- Its semantic relationships (what concepts is it related to?)
- Its referential properties (what does it refer to?)

Asking a single attention mechanism to capture all of this at once is asking too much.

Multi-head attention addresses this by running **several attention mechanisms in parallel**, each with its own learned query, key and value transformations. Instead of having one set of attention weights, we have eight or sixteen sets, each potentially specializing in different aspects of language: 
- **One head** might learn to track subject-verb relationships, consistently giving high attention weights when a verb attends to its subject
- **Another head** might specialize in tracking modifiers, with adjectives attending strongly to the nouns they modify
- **A third** might capture semantic similarity, with words attending to conceptually related words even when they are far apart in the sentence

Through careful analysis of trained transformer models, researchers have discovered that these attention heads do indeed **specialize**, though the specialization emerges from training rather than being explicitly programmed. 

In a landmark study analyzing BERT, researchers found heads that consistently performed specific linguistic functions: 
- Some heads tracked **coreference** - pronouns attending to their referents
- Others tracked **syntactic dependencies** like subject-verb agreement
- Still others seemed to capture **positional information**, with each word attending primarily to its immediate neighbors

The implementation of multi-head attention is elegant and computationally efficient. Instead of using the model's full dimensionality for one attention computation, we split the dimension into multiple smaller pieces and run parallel attention computations on each piece. 

For example, if our model has dimension 768 and we use 12 attention heads, each head operates on dimension 64. The total amount of computation is roughly the same as single attention head operationg on the full dimension, but we get **12 different perspectives** on the relationships in the text. 

After all heads have computed their outputs, these outputs are concatenated back together and passed through a final linear transformation.

This multi-head design has proven **crucial for achieving strong performance** on language tasks. Models with a single attention head, even when given the same total parameter count, consistently underperform models with multiple heads. 

The ability to maintain **multiple parallel interpretations** of the input, each capturing different facets of linguistic structure, appears to be fundamental to how transformers achieve their impressive language understanding capabilities.


### The Mathematical Foundations

The choice of dot product as the similarity function in attention is not arbitrary - it has important mathematical and computational properties. The dot product between two vectors produces: 
- A **large positive value** when the vectors point in similar directions
- **Zero** when they're orthogonal
- **Negative values** when they point in opposite directions

This makes it a natural measure of alignment or similarity between the query (what we're looking for) and the key (what information is available).

However, there's a subtle problem with using dot products directly in high-dimensional spaces. As the dimension of the vectors increases, the magnitude of dot products tends to grow. For vectors of dimension 512 or 768, which are typical in transformer models, dot products can become quite large. This causes problems when we apply the softmax function to convert these dot products into attention weights.

**The softmax issue**: The softmax function is designed to convert any set of numbers into a valid probability distribution - positive values that sum to one. It does this by exponentiating each input value and then dividing by the sum of all exponentiated values. 

**The problem**: When the input values are very large, softmax becomes extremely "**peaky**" - almost all the probability mass goes to the single largest value, with vanishingly small probabilities for everything else. This means the model would essentially hard-select one token and completely ignore all others, losing the valuable property of being able to softly weight multiple tokens according to their relevance.

The solution is elegantly simple: **scale the dot products before applying softmax**. Specifically, divide each dot product by the square root of the dimension of the key vectors:
- If the key vectors have dimension 64, divide by $\sqrt{64} = 8$
- If they have dimension 512, divide by $\sqrt{512} \approx 22.6$

This scaling keeps the dot products in a range where softmax produces **balanced probability distributions**, allowing the model to spread its attention across multiple relevant tokens rather than focusing all attention on a single token.

The original *"Attention Is All You Need"* paper included ablation studies demonstrating the importance of this scaling:
- When comparing scaled dot-product attention to unscaled dot-product attention on machine translation tasks, the scaled version **significantly outperformed** the unscaled version
- The improvement was particularly notable for **longer sequences** and **higher dimensions**
- While additive attention mechanisms (which use a small feedforward network to compute alignment scores) can perform comparably to dot-product attention for smaller dimensions, **scaled dot-product attention** is both more computationally efficient and more effective for the larger dimensions used in practical transformer models

### Positional Encoding

One crucial detail about attention mechanisms that might not be immediately obvious is that they have **no inherent notion of position or order**. The attention computation involves comparing vectors and taking weighted sums - operations that don't depend on where in the sequence each vector came from. 

If we were to randomly suffle all the words in a sentence, a pure attention mechanism would compute exactly the same attention weights and produce the same output. This is both a strength and a significant limitation.

**The paradox**:
- **Strength**: Allows complete parallelization - every position in a sequence can compute its attention simultaneously, without waiting for previous positions to finish (makes transformers much faster to train than recurrent networks)
- **Limitation**: Word order matters tremendously in language - $\text{"The dog bit the man"}$ means something entirely different from $\text{"The man bit the dog"}$, even though they contain exactly the same words

The model needs some way to know which words come before which other words.

The solution is to **add positional information to the token embeddings** before they enter the attention mechanism. The original transformer paper introduced a clever approach using **sine and cosine functions with different frequencies**. 

For each position in the sequence, they computed a unique pattern of sine and cosine values:
- Position 0 gets one pattern
- Position 1 gets a slightly different pattern
- Position 2 gets another pattern
- And so on

These positional encodings are simply **added to the word embeddings**, effectively annotating each word with information about where it appears in the sequence.

The choice of sine and cosine functions was not arbitrary. These functions have several useful properties:
1. Can extend to any sequence length, even lengths longer than those seen during training
2. Have a mathematical property where the encoding for position $k$ can be written as a linear function of the encoding for position $j$, which might help the model learn about relative positions
3. The combination of multiple frequencies creates unique patterns for each position, ensuring the model can distinguish between different positions

However, many modern transformer implementations have moved away from fixed sinusoidal encodings to **learned positional embeddings**. Instead of using a predetermined mathematical function, these models simply learn position-specific embedding vectors during training, just like they learn word embeddings. 

**Trade-offs**:
- **Advantage**: Flexibility - the model can learn whatever positional representation works best for its task
- **Disadvantage**: Learned embeddings don't naturally extend to sequence lengths beyond those seen in training, which can be problematic for models that need to handle variable-length inputs

More recently, researchers have developed sophisticated alternatives like **Rotary Position Embedding (RoPE)** which has become increasingly popular in state-of-the-art models. RoPE works by rotating the query and key vectors by an amount that depends on their positions before computing attention scores. This rotation is carefully designed so that the attention score between positions i and j depends primarily on their **relative distance** rather than their absolute positions. This makes the model more robust and better able to handle sequences longer than those seen during training, which has become increasingly important as models are asked to process longer and longer documents.

### Advanced Attention Variants

As transformers have become the dominant architecture in natural language processing and beyond, researchers have developed numerous variations on the basic attention mechanism, each addressing different limitations or targeting specific applications.

Flash Attention represents a significant advance in **computational efficiency** without changing the mathematical formulation of attention. 

The problem was that standard attention implementation requires loading the full attention matrix into memory - an $N \times N$ matrix where $N$ is the sequence length. For long sequences, this matrix becomes prohibitively large.

Flash Attention achieves dramatic speedups by reformulating the attention computation to work with **smaller blocks** that fit efficiently into the fast on-chip memory of modern GPUs:
- Instead of computing the full attention matrix at once, Flash Attention processes the input in **tiles**
- Carefully orchestrates memory access patterns to minimize expensive memory transfers
- Allows processing of much longer sequences without running out of memory
- Computes exactly the same output as standard attention

**Key insight**: Attention can be computed **incrementally** - we don't need to materialize the full attention matrix if we're clever about the order of operations.

Sparse Attention addresses the quadratic computational complexity of attention from a different angle.

The problem is that standard attention requires every token to attend to every other token, leading to computations that grow quadratically with sequence length. For documents with 100,000 tokens, this becomes prohibitively expensive. 

Sparse attention mechanisms selectively compute attention only for certain pairs of tokens, reducing the computational cost to something closer to **linear in sequence length**. 

**Different sparse patterns**:
- **Local attention**: Only attends to nearby tokens, capturing local dependencies efficiently
- **Strided attention**: Attends to every $k$-th token, allowing the model to capture longer-range dependencies without the full quadratic cost
- **Longformer**: Combines local attention with a small number of global tokens that can attend to everything, giving a mix of local processing and global awareness

These patterns achieve strong performance on long documents while maintaining reasonable computational costs.

**Grouped-Query Attention (GQA)** represents a middle ground between standard multi-head attention and an extreme efficiency optimization called multi-query attention. 

**The spectrum**:
- **Standard multi-head attention**: Each head has its own query, key, and value matrices
- **Multi-query attention**: Each head has its own query matrix but shares a single set of key and value matrices across all heads
  - Dramatically reduces memory bandwidth requirements and speeds up inference
  - Some cost to model quality

**GQA approach**: Divides attention heads into groups, with each group sharing key and value matrices while maintaining separate queries.

**Why it works**: Provides much of the efficiency benefit of multi-query attention while better preserving model quality. GQA has become popular in recent large language models because it allows for **faster inference with minimal degradation in performance**.

### Self-Attention vs. Cross-Attention

An important distinction exists between **self-attention** and **cross-attention**, though both use the same underlying attention mechanism. 

In self-attention, the queries, keys and values all come from the **same source**. When a tranformer processes an input sentence, it uses self-attentionto allow each word to attend to every other word in that same sentence. This is what enables the model to capture dependencies and relationships within the input.

**Example**: In the sentence "The cat sat on the mat," self-attention allows "sat" to attend to "cat" to understand the subject performing the action, and to "mat" to understand where the sitting occurred.

Cross-attention, by contrast, allows **one sequence to attend to a different sequence**. 

In the original transformer architecture designed for machine translation: 
- The decoder used cross-attention to allow the target language words being generated to attend to the source language words
- The **queries** came from the partially generated translation
- The **keys and values** came from the encoder's representation of the source sentence
- This allowed the model to focus on relevant parts of the source text when generating each target word

Modern decoder-only language models like GPT primarily use self-attention, as they process a single sequence of tokens that includes both the input prompt and the generated output. 

**Decoder-only models (like GPT)**: Primarily use self-attention, as they process a single sequence of tokens that includes both the input prompt and the generated output.

**Cross-attention becomes important** in models that need to integrate multiple modalities or types of information:

- **Vision-language models**: Models that can answer questions about images use cross-attention to allow text representations to attend to image features
- **Text-to-image models**: Models that generate images from text descriptions use cross-attention to allow the image generation process to focus on relevant parts of the text description

Understanding when to use self-attention versus cross-attention is crucial for designing architectures that can effectively integrate different types of information.

### The Broader Impact

The introduction of attention mechanisms and transformer architectures represents one of those rare moments in science where a relatively simple idea leads to transformative change across an entire field. Within just a few years of the publication of *"Attention Is All You Need"*, transformers had essentially replaced recurrent and convolutional architectures in natural language processing. Today, virtually every state-of-the-art language model uses transformer-based attention.

The reasons for this rapid adoption go beyond the specific technical advantages we've discussed. Attention mechanisms are fundamentally more aligned with how computation works on modern hardware: 
- **Modern accelerators** (GPUs and TPUs) are designed for massive parallelism - performing many independent computations simultaneously
- **Recurrent networks**, with their sequential dependencies, couldn't fully utilize this parallel processing power
- **Transformers**, by contrast, can process entire sequences in parallel, making them ideally suited for modern accelerators

Attention mechanisms are also more interpretable than the hidden states of recurrent networks:
- When we want to understand why a model made a particular decision, we can **examine the attention weights** to see which inputs the model focused on
- While attention weights don't tell the complete story of how information flows through a deep network, they provide valuable insight that simply wasn't available with recurrent architectures
- This interpretability has been important both for **building trust** in these systems and for **understanding their failures**

Perhaps most significantly, attention mechanisms have proven to **scale remarkably well**. Researchers have discovered that making transformer models larger - adding more layers, more attention heads, more parameters - consistently leads to better performance, at least up to the enormous scales we've reached with today's models. Key findings: 
- This scaling property has enabled the creation of models with **hundreds of billions or even trillions of parameters**
- Something that would have been impractical with earlier architectures
- The fact that transformer performance continues to improve with scale, without obvious saturation, suggests we may not yet have reached the limits of what these architectures can achieve

The influence of attention has extended far beyond natural language processing: 
- **Computer Vision**: Vision transformers have shown that attention-based architectures can match or exceed convolutional neural networks for image understanding
- **Biology**: Attention is being applied to protein structure prediction
- **Finance**: Time series forecasting
- **Scientific Computing**: Modeling physical systems

The transformer architecture has become a **general-purpose tool for processing sequential data across domains**. This universality suggests that attention mechanisms capture something fundamental about how information should be processed, not just in language but in many types of structured data.





---
## Sampling Techniques

When we interact with a large language model, requesting it to write a story, answer a question, or generate code, what we see as output is the result of **thousands of individual decisions**. At each step of generation, the model doesn't simply produce a word - it produces a **probability distribution over its entire vocabulary**, assigning a likelihood to each of the tens of thousands of tokens it knows. 

The question of how to select from this distribution of possibilities, how to navigate the trade-off between predictability and creativity, between safety and surprise, is what **sampling techniques** address. This choice profoundly shapes every aspect of the generated text, from its coherence and accuracy to its creativity and diversity.

### Understanding the Probability Distribution

To understand sampling, we must first understand what the model actually produces at each generation step. 

After processing all the input through its many layers of transformers and attention mechanisms, the model's final layer outputs what are called **logits** - raw numerical scores for every token in its vocabulary. These aren't probabilities yet; they're unbounded real numbers that can be positive or negative, small or large. A common vocabulary might contain 50.000 tokens, so the model outputs 50.000 logits, one for each possible next token.

To convert these logits into probabilities, the model applies the **softmax function**, which we encountered earlier in the context of attention mechanisms. Softmax takes any collection of numbers and transforms them into a valid probability distribution - positive values that sum to one. It does this by:
1. Exponentiating each logit (raising $e$ to the power of the logit)
2. Dividing by the sum of all exponentiated values

The exponential function means that even small differences in logits can lead to large differences in probabilities, and large differences in logits lead to enormous differences in probabilities.

Consider a concrete example. The model has just generated $\text{"The cat sat on the"}$ and must predict the next word. 
The logits might be something like 
- $\text{"mat: "}8.2$
- $\text{"floor: "}7.9$
- $\text{"chair: "}7.5$
- $\text{"table: "}6.8$
- $\text{"roof: "}4.2$
- $\text{"computer: "}1.1$
- $\text{"quantum: "}-3.1$
- ... and so on for all 50.000 tokens. 

**After applying softmax, these convert to probabilities**:
- $\text{"mat: "}0.31$ (31% chance)
- $\text{"floor: "}0.25$ (25% chance)
- $\text{"chair: "}0.18$ (18% chance)
- $\text{"table: "}0.10$ (10% chance)
- $\text{"roof: "}0.02$ (2% chance)
- $\text{"computer: "}0.001$ (0.1% chance)
- $\text{"quantum: "}0.0001$ (0.01% chance)
- ...with the remaining tiny bits of probability spread across tens of thousands of other words

This probability distribution reveals something fundamental about language: **even given perfect context, there are often many plausible continuations**. The model isn't certain that "mat" is the next word - it's the most likely, but "floor" and "chair" are also quite reasonable.

This inherent uncertainty is a **feature of language itself**, not a limitation of our models. Different authors might complete the sentence differently, and all could be correct. The challenge is deciding how to select from this distribution in a way that produces high-quality, useful text for the task at hand.

### Greedy Decoding

The most straightforward way to select the next token is **greedy decoding**: always pick the token with the highest probability. Given our esample, greedy decoding would select $\text{"mat"}$. Then, with $\text{"The cat sat on the mat"}$ as context, we generate probabilities for the next word and again pick the highest probability token. Perhaps that's a period, giving us $\text{"The cat sat on the mat"}$ as our complete generation.

Greedy decoding has several obvious advantages:
- **Deterministic**: Given the same input, the model always produces the same output, which can be valuable for reproducibility and consistency
- **Computationally efficient**: Requires no random sampling or complex decision-making
- **Works well for factual tasks**: For many applications where there's a clear correct answer, greedy decoding works perfectly well 

For example, if the model is answering a factual question like $\text{"What is the capital of France?"}$, greedy decoding will reliably produce $\text{"Paris"}$ if that's the highest probability token.

However, greedy decoding has severe limitations that become apparent in tasks requiring longer or more creative outputs. 

**The local vs. global optimality problem**: By always choosing the locally optimal next token, it can lead to globally suboptimal sequences. The issue is that the highest probability next token right now might lead the model down a path where the only continuations are awkward or incorrect.

**Repetitive loops**: Greedy decoding can get "stuck" in repetitive loops, particularly in longer generations:
- Text that repeats the same phrase over and over
- Text that becomes increasingly redundant and boring
- The model essentially gets trapped in a pattern where each next step is locally optimal but the overall sequence is poor

**Poor creative output**: For creative tasks like story writing, greedy decoding produces especially unsatisfying results:
- The highest probability continuation is often generic and predictable
- Stories tend to be bland, using only common words and familiar phrases
- Never takes the creative risks that make fiction engaging
- The text is grammatically correct and factually plausible but entirely lacking in surprise or originality

This is why virtually all practical language model applications use more sophisticated sampling methods.

### Temperature

Temperature is perhaps the single most important parameter for controlling the character of generated text. It doesn't change which token has the highest probability, but it dramatically changes the shape of the probability distribution before we sample from it. 

Temperature is applied by dividing all the logits by the temperature value before applying softmax. This simple scaling has profound effects.

At low temperature, we divide the logits by a number less than one, which is equivalent to multiplying them, making the differences between logits larger. When these larger differences go through the softmax exponential function, the result is a **sharper, more peaked distribution**:
- High-probability tokens become even more likely
- Low-probability tokens become even less likely
- At temperature approaching zero, the distribution becomes so peaked that it's essentially deterministic, like greedy decoding
- The model becomes very confident and conservative, almost always picking the highest probability token

At high temperature, we divide logits by a number greater than one, making the differences between logits smaller. The resulting softmax distribution is **flatter, more spread out**:
- Tokens that originally had low probability see their probabilities increase
- High-probability tokens see their probabilities decrease
- The model becomes less certain, more willing to explore less probable options
- At extremely high temperatures, the distribution becomes nearly uniform, approaching random selection from the vocabulary

Let's see this with our $\text{"The cat sat on the"}$ example. 

**At temperature 0.5** (conservative):
- "mat": 0.31 → 0.55
- "floor": 0.25 → 0.20
- "chair": 0.18 → 0.10

The model has become more confident that "mat" is correct.

**At temperature 1.5** (exploratory):
- "mat": 0.31 → 0.22
- "floor": 0.25 → 0.20
- "chair": 0.18 → 0.17
- "roof": 0.02 → 0.08

The model is now much more willing to explore alternatives.

The practical implications are enormous. For tasks requiring factual accuracy - answering questions, extracting information, summarizing technical documents - low temperature around 0.2 to 0.4 is typically best. The model sticks close to its highest confidence predictions, minimizing the chance of generating incorrect information. For creative writing, higher temperature around 0.8 to 1.0 produces more interesting and varied text. The model takes chances, uses less common words, constructs unexpected phrases. For programming, very low temperature around 0.2 is usually appropriate because even small errors in code can be catastrophic - a single misplaced bracket breaks everything.

One crucial insight is that **temperature interacts with model quality**. A well-trained model with accurate probability estimates can be used effectively at higher temperatures because even its lower-probability choices are reasonable. A poorly trained model might assign inappropriate probabilities and high temperature would amplify these errors, leading to nonsense. This is why larger, better-trained models often perform better at creative tasks - they can safely operate at temperatures that would break smaller models.

### Top-K Sampling



---