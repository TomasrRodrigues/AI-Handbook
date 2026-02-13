# Large Language Models: Architectural Foundations

#### Table of Contents

1. [LLM Architecture Overview](#llm-architecture-overview)
2. [Tokenization](#tokenization)

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

Before any Large Language Model can begin to process text, that text must first be broken down into units that the model can work with. This fundamental preprocessing setp is called tokenization, and while it might seem like a minor technical detail, the way we choose to tokenize text has profound implications for model performance, efficiency and capabilities. Tokenization is the bridge between raw text as humans write it and the numerical representations that neural networks require.

At its most basic level, tokenization is the process of converting a string of text into a sequence of smaller units called tokens. These tokens become the vocabulary items that the model understands - the building blocks from which the model constructs its understanding of language. When we type a sentence into a language model interface, that sentence is first tokenized before any of the sophisticated neural network processing we disussed earlier can begin. The choice of how to break text into tokens fundamentally shapes what patterns the model can learn and how efficiently it can process different types of text.

To understand why tokenization matters, consider the challenge that language model face. Neural networks work with numbers, not words. Each token in the model's vocabulary is assigned a unique integer identifier, and these identifiers are used to look up the embeddings that we discussed earlier. The tokens we choose as our vocabulary directly determine what the model considers to be atomic units of meaning. If our tokens are individual characters, the model must learn to assemble those characters into meaningful words. If our tokens are complete words, the model can work directly with word-level meaning but struggles with rare or unknown words.

The question of how to tokenize text might seem to have an obvious answer: just split on spaces to separate words. This approach, called word-level tokenization, is intuitive and was used by many earlt natural language processing systems. For a sentence like $\text{"The cat sat on the mat."}$, word-level tokenization produces six word tokens plus punctuation:

$$
\text{"The cat sat on the mat."} \quad \xrightarrow{Tokenization} \quad \text{["The","cat","sat","on","the","mat","."]}
$$

This seems natural because it matches how we think about language - as sequence of words. However, this straightforward approach reveals significant problems when we examine it more closely.

The first issue with word-level tokenization is vocabulary size. Natural languages contain enormous numbers of distinct words. English has hundreds of thousands of words in common use and the number grows even larger when we include proper names, technical terms and inflected forms. If we want our model to handle all possible words, we need an enormous vocabulary, which means an enormous embedding matrix. Remember that each word needs its own embedding vector, so a vocabulary of 500,000 words with embedding dimension 768 would require 384 million parameters just for the embeddings - a substantial portion of the total parameters even in large models.


More problematic is how word-level tokenization handles rare and unknown words. During training, the model only sees a finite amount of text, which means many valid words will be extremely rare or completely absent from the training data. Words that appear only once or twice won't have well-learned embeddings because there weren't enough examples for the model to understand how they're used. Words that never appear in the training data pose an even worse problem. The standard solution is to include a special unknown word token, typically written as ${\text[UNK]}$ or $\text{<unk>}$, that represents any word not in the vocabulary. But this approach loses information. The words "antidisestablishmentarianism" and "pseudopseudohypoparathyroidism" are both rare medical or technical terms, but if neither appears in your vocabulary, they would both be converted to the same ${\text[UNK]} token, losing the distinction between them.

Word-level tokenization also misses important relationships between related words. Consider the words "care", "cares", "cared", "caring", "careful", "carefully", "careless", "carelessly", and "carelessness". These are all morphologically related - they're different forms or derivatives of the same root word "care". But word-level tokenization treats each as a completely separate vocabulary item with its own embedding. The model must learn the relationships between these forms from scratch based on how they're used in context, rather than being able to leverage their obvious structural similarity. This makes learning less efficient and requires more training data to achieve the same level of understanding.


### Character-Level Tokenization

At the opposite extreme from word-level tokenization is character-level tokenization, where text is broken down into individual characters:

$$
\text{"CAT"} \quad \xrightarrow{Tokenization} \quad \text{["C","A","T"]}
$$

This approach solves the vocabulary size problem completely. English only needs about 100 characters including uppercase and lowercase letters, digits, punctuation and a few special characters. There's no concept of an unknown word because any word, no matter how rare or how it's spelled, can be represented as a sequence of know characters.

Character-level tokenization also naturally handles morphology. Since $\text{"care"}$, $\text{"cared"}$ and $\text{"caring"}$ all share the character sequence $\text{"c-a-r"}$, a character-level model can potentially learn to recognize this shared structure and understand that these words are related. The model can handle misspelings, creative spellings and made-up words more gracefully because it's working at the level of characters rather than trying to match against a fixed vocabulary of complete words. If someone writes $\text{"soooo"}$ with extra o's for emphasis, a character-level model can process this just as easily as $\text{"so"}$, whereas a word-level model might not have $\text{"soooo"}$ in its vocabulary.

However, character-level tokenization introduces severe drawbacks that make it impractical for most applications. The most obvious problem is sequence length. That simple sentence $\text{"The cat sat on the mat"}$ has seven word-level tokens but 22 character-level tokens when we count spaces and punctuation. For a long document, character-level tokenization might produce sequences tens of thousands of tokens long. This is a serious problem because transformer models have computational complexity that grows quadratically with sequence length. Processing a sequence that's three times longer requires nine times as much computation in the attention layers. This makes character-level tokenization prohibitively expensive for long documents.

More fundamentally, character-level tokenization makes the model's learning task much harder. Individual characters carry very little semantic information by themselves. The character $\text{"c"}$ in isolation doesn't mean anything - meaning emerges from combinations of characters into words and from combinations of words into sentences. This means a character-level model must first learn to assemble characters into word-like units, then learn word meanings, then learn how words combine into sentences. We're asking the model to learn at a very low level of abstraction, which requires more model capacity and more training data to achieve the same level of performance as a model working with more meaningful units.

### The Subword Solution

The solution that modern Large Language Models use is subword tokenization, which occupies the middle ground between word-level and character-level approaches. The key insight is that words can be broken into meaningful pieces that are larger than individual characters but smaller than complete words. Common words remain intact as single tokens, while less common words are split into subword pieces that the model can compose to understand the full word.

Consider the word $\text{"unhappiness"}$. A subword tokenizer might break this into $\text{"un"}$, $\text{"happy"}$ and $\text{"ness"}$. Each of these pieces carreis meaning: $\text{"un"}$ is a common prefix meaning $\text{"not"}$, $\text{"happy"}$ is a complete word, and $\text{"ness"}$ is a common suffix that turns adjectives into nouns. By learning embeddings for these subword pieces, the model can understand new words by composition. Even if it never saw $\text{"unhappiness"}$ during training, if it learned embeddings for $\text{"un"}$, $\text{"happy"}$ and $\text{"ness"}$ from other contexts, it can compose these to understand the meaning of $\text{"unhappiness"}$.

This approach gives us several crucial benefits. First, we can control the vocabulary size to find a sweet spot - large enough to represent common words efficiently, but small enough to be manageable. Typical vocabulary sizes for modern models range from 30,000 to 50,000 tokens. This is much smaller than the hundreds of thousands of words in a language, but much larger than the hundred or so characters needed for character-level tokenization. With careful choice of subwords, this vocabulary size is sufficient to represent any text efficiently.

This approach gives us several crucial benefits. First, we can control the vocabulary size to find a sweet spot - large enough to represent common words efficiently. Typical vocabulary sizes for modern models range from 30,000 to 50,000 tokens. This is much smaller than the hundreds of thousands of words in a language, but much larger than the hundred or so characters needed for character-level tokenization. With careful choice of subwords, this vocabulary size is sufficient to represent any text efficiently.


Second, subword tokenization eliminates the unknown word problem. Even if a word never appeared in the training data, it can be represented as a sequence of subword tokens. The word "antidisestablishmentarianism" might be broken into pieces like "anti", "dis", "establish", "ment", "arian", "ism" - all of which are common morphemes that likely appeared in many other words during training. The model can understand the unfamiliar word by composing the meanings of its familiar pieces, similar to how a human might puzzle out the meaning of a long word by recognizing its morphological components.

Third, subword tokenization naturally captures morphological relationships. Words like $\text{"run"}$, $\text{"running"}$, $\text{"runner"}$, and $\text{"runs"}$ all contain the subword $\text{"run"}$, allowing the model to learn that these words are related. Prefixes like $\text{"un"}$, $\text{"re"}$, $\text{"pre"}$, $\text{"dis"}$ and suffixes like $\text{"ing"}$, $\text{"ed"}$, $\text{"ness"}$, $\text{"tion"}$ can be learned as meaningful units, and the model can learn their systematic effects on word meaning. This makes learning more efficient and allows better generalization to new words formed by familiar morphological processes.

### Byte Pair Encoding

The most widely used subword tokenization algorithm is Byte Pair Encoding, or BPE. This algorithm was originally developed for data compression but was adapted brilliantly for natural language processing. The beauty of BPE is that it's completely data-driven - it learns what subwords are useful by analyzing actual text, rather than relying on linguistic knowledge about morphemes or word structure.

BPE starts with a vocabulary containing individual characters. For English text, this initial vocabulary includes all the letters, digits, punctuation marks and any special characters that appear in the text. At this starting point, every word is represented as a sequence of character tokens. The word $\text{"lower}$ would be represented as the five character tokens $\text{"l"}$, $\text{"o"}$, $\text{"w"}$, $\text{"e"}$, $\text{"r"}$, typically with a special end-of-word symbol appended to mark word boundaries.

The training process then repeatedly funds the most frequent pair of consecutive tokens in the training corpus and merges them into a single new token. This new token is added to the vocabulary and wherever that pair appears in the corpus, it's replaced by the new merged token. This process continues iteratively, with each iteration merging the most frequent pair and adding one new token to the vocabulary, until the vocabulary reaches the desired size.
Let's walk through this process with a concrete example. Imagine we're training a BPE tokenizer on a tiny corpus containing the words $\text{"low"}$, $\text{"lower"}$, $\text{"lowest"}$, $\text{"newer"}$, $\text{"wider"}$, each appearing with some frequency. We represent each word with its character plus an end-of-word marker, which I'll write as $\text{"\_"}$ for clarity. So our corpus contains sequences like $\text{"low\_"}$, $\text{"lower\_"}$, $\text{"lowest\_"}$, $\text{"newer\_"}$ and $\text{"wider\_"}$.

Now we count how often each pair of adjacent tokens appears. The pair $\text{"e r"}$ appears in $\text{"lower"}$, $\text{"newer"}$ and $\text{"wider"}$. The pair $\text{"lo"}$ appears in $\text{"low"}$, $\text{"lower"}$ and $\text{"lowest"}$. Let's say $\text{"e r"}$ is most frequent. We merge this pair into a new token $\text{"e r"}$ and update our corpus. Now $\text{"lower"}$ becomes $\text{"l o w er \_"}$, $\text{"newer"}$ becomes $\text{"n e w er \_"}$ and $\text{"wider"}$ becomes $\text{"w i d er \_"}$.

In the next iteration, we might find that $\text{"er"}$ *(the $\text{"er"}$ token followed by the end-of-word marker) is now the most frequent pair*. We merge this into $\text{"er"}$, representing the common suffix. Our words become $\text{"l o w er \_"}$, $\text{"n e w er \_"}$, $\text{"w i d er \_"}$. We continue this process, perhaps next merging $\text{"l o"}$ into $\text{"lo"}$, then $\text{"lo w"}$ into $\text{"low"}$ and so on.

After many iterations, our vocabulary contains not just individual characters but also common subwords that were discovered by the frequency-based merging process. Common complete words like $\text{"the"}$, $\text{"a"}$, $\text{"is"}$ will likely end up as single tokens because all their characters get merged together early in the training process since these words appear so frequently. Less common words will be split into pieces, with split points generally corresponding to natural morpheme boundaries because morphemes are reused across many words.

The key insight that makes BPE work so well is that it automatically discovers meaningful subwords without any linguistic knowledge. If $\text{"ing"}$ appears at the end of many words, those $\text{"i"}$, $\text{"n"}$, $\text{"g"}$ sequences will be frequently adjacent and will get merged together early in training, creating an $\text{"ing"}$ token. If $\text{"un"}$ appears at the beginning of many words, it will be merged into a token. The algorithm discovers these patterns purely from frequencies in the data, but because language has systematic morphological structure, the discovered patterns correspond to linguistically meaningful units.

When we want to tokenize new text with a trained BPE tokenizer, we apply the learned merge operations in the same order they were learned during training. We start with the character-level representation of each word, then apply each merge rule in sequence. For the word $\text{"newer"}$, we'd start with $\text{"n e w e r"}$, apply the merge that creates $\text{"er"}$ to get $\text{"n e w er "}$, apply the merge that creates $\text{"er"}$ to get $\text{"n e w er"}$ and apply any other relevant merges. The final tokenization depends on what merges were learned during training and what order they were learned in.

One particularly clever variant of BPE is byte-level BPE, which is used by models like GPT-2 and GPT-3. Instead of starting with characters as the base vocabulary, byte-level BPE start with bytes - the raw UTF-8 byte encoding of text. Since any text in any language can be represented as a sequence of bytes from the range 0-255, this makes the tokenizer completely universal. It can handle English, Chinese, Arabic, emoji, mathematical symbols or any other content without special handling. The tokenizer simply treats everything as sequences of 256 possible byte values and learns to merge frequent byte sequences, exactly like standard BPE but operating at the byte level instead of the character level.

Byte-level BPE has become increasingly popular because of this universitality. We don't need different tokenizers for different languages or writing systems. We don't need special preprocessing to handle unusual characters or symbols. Everything is just bytes and the merge operations learn whatever patterns are frequent in our training data, whether those patterns are English words, Chinese characters, emoji sequences or anything else. This makes it much easier to build truly multilingual models and models that can handle the full diversity of text found on the internet.

### WordPiece and Other Approaches

While BPE is the most widely used subword tokenization algorithm, other approaches exist with slightly different design choices. WordPiece used by BERT and related models, is very similar to BPE but uses a different criterion for choosing which pairs to merge. Instead of simply picking the most frequent pair, WordPiece chooses the pair that, when merged, maximizes the likelihood of the training data according to a language model.

The intuition behind WordPiece's criterion is that we want to create tokens that are predictive - tokens that help us predict the surrounding context. Simply merging the most frequent pair might not always be the best choice if that pair doesn't have consistent meaning or usage patterns. By considering likelihood, WordPiece tries to find merges that create more coherent and meaningful tokens. In practice, BPE and WordPiece often produce similar vocabularies, but the likelihood-based criterion can lead to better tokenizations in some cases.

WordPiece also uses a special convention where subword tokens that don't begin a word are marked with a prefix, typically $\text{"##"}$. So the word $\text{"unhappiness"}$ might be tokenized as $\text{"un"}$, $\text{"##happiness"}$ or $\text{"un"}$, $\text{"##happy"}$, $\text{"##ness"}$. The $\text{"##"}$ markers make it immediately clear which tokens are word beginnings and which are continuations. This can be useful for downstream tasks where we need to know word boundaries, such as named entity recognition where we need to identify which tokens are part of the same entity.

Another approach is the Unigram language model tokenization algorithm, which takes a fundamentally different strategy. Instead of starting with characters and building up by merging, Unigram start with a large vocabulary of possible subwords that appear in the corpus with sufficient frequency, creating a very large intial vocabulary. It then iteratively removes tokens that, when removed, cause the smallest increase in loss according to a unigram language modek fit to the corpus.

The Unigram approach has some theoretical advantages. BPE is a greedy algorithm that makes locally optimal choices at each step - merge the most frequent pair - but these choices might not lead to a globally optimal vocabulary. Unigram tries to find a better solution by considering the global effect of including or excluding each token. It asks: given all the other tokens in the vocabulary, does this particular token help us encode the corpus efficiently? If removing a token doesn't hurt much because the words it appeared in can be efficiently encoded using other tokens then that token can be pruned.


A practical implementation of these subword tokenization approaches is provided by the SentencePiece library, which has become widely used in modern natural language processing. SentencePiece implements both BPE and Unigram algorithms with several useful enhancements. One key feature is that it treats spaces as regular characters, encoding them explicitly rather than treating them as special delimiters. This seemingly small choice makes the tokenizer truly language-agnostic - it can handle languages that don't use spaces to separate words, like Chinese or Japanese, using exactly the same algorithm as languages that do use spaces.

When training a SentencePiece model, you provide a corpus of raw text and specify the desired vocabulary size and algorithm choice. The system samples from the corpus, runs the training algorithm, and produces a model file containing the learned vocabulary and merge rules. This model can then be used to consistently tokenize any new text. For multilingual models, you would train SentencePiece on a mixed corpus containing text in all the target languages, and it would learn a unified vocabulary that efficiently represents common patterns across all those languages.

### PRactical Considerations and Challenges





---