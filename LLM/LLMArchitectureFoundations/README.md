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

#### Encoding Position Information

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

Training a Large Language Model is one of the most computationally intensive tasks in modern artificial intelligence. The process typically unfolds in two main phases: pretraining and fine-tuning. During pretraining, the model learns general language understanding from enormous amounts of text. This phase requires massive computational resources and can take weeks or months even on hundres or thousands of powerful processors. The goal of pretraining is to give the model broad knowledge about language, reasoning and the world, which it can then apply to many different tasks.

The pretraining process is fundamentally about prediction. For a decoder-only model like GPT, pretraining involves showing the model sequences of text and training it to predict what comes next. If the model sees $\text{"The cat sat on the"}$, it should predict something like $\text{"mat"}$ or $\text{"chair"}$ or another plausible continuation. For encoder-only models like BERT, pretraining involves masking out random words and training the model to predict what the masked words were. Through billions of these prediction tasks, the model gradually learns patterns in language - grammar rules, common phrases, factual relationships, reasoning patterns and much more.

The data used for pretraining is critically important. Modern large language models are trained on datasets containing hundreds of billions or trillions of words, drawn from books, websites, scientific papers and other text sources. The quality and diversity of this data directly impacts what the model learns. If the training data contains biased viewpoints, factual errors or problematic content, the model will learn and potentially reproduce these issues. Careful curation and filtering of training data is therefore essential, though it remains a challenging problem to ensure training data is both large enough and high quality enough.

During training, the model starts with randomly initialized parameters and gradually adjusts them to improve its predictions. This adjustment happens through a process called backpropagation with gradient descent. After the model makes a prediction, a loss function measures how wrong the prediction was. Then, the system computes gradients that indicate how each parameter should be adjusted to reduced the loss. The parameters are updated in small steps in the direction that reduces the loss and this process repeats millions of times with different batches of data. Gradually, the model's parameters converge to values that make good predictions across a wide range of text.

Training stability is a major concern for very large models. Various technical problems can derail training, such as exploding gradients where parameter updates become so large they destroy previously learned patterns, or vanishing gradients where updates become so small that learning slows to a crawl. Modern training techniques include careful initialization of parameters, gradient clipping to cap the maximum size of updates, learning rate schedules that adjust how quickly parameters change and architectural choices like residual connections and layer normalization that promote stable training.

After pretraining creates a model with broad language understanding, fine-tuning adapts it for specific tasks or domains. Fine-tuning uses much smaller amounts of data - thousands or millions of examples rather than billions - and takes much less time than pretraining. During fine-tuning, the model's parameters are adjusted to specialize it for a particular application. For instance, a model might be fine-tuned on medical literature to improve its performance on medical questions, or on code repositories to enhance its programming capabilities, or on conversational data to make it better at dialogue.

Fine-tuning can take several forms depending on the goal. In supervised fine-tuning, the model is trained on input-output pairs specific to a task. For sentiment analysis, we would provide sentences paired with sentiment labels and the model learns to predict the correct label for new sentences. For question answering, we provide questions and contexts paired with answers. More recently, instruction tuning has become popular, where the model is fine-tuned on a diverse set of instructions even for tasks it hasn't explicitly seen.

An especially important fine-tuning technique is reinforcement learning from human feedback, or RLHF. This approach was crucial in developing conversational AI systems like ChatGPT. In RLHF, humans provide feedback on model outputs, comparing different responses and indicating which are better. This feedback is used to train a reward model that predicts how a human would evaluate any given response. Then the language model is further trained using reinforcement learning to generate responses that score highly according to the reward model. This process helps align the model's behavior with human preferences, making it more helpful, harmless and honest in its responses.

### Model Scaling and Computational Requirements









---
## Tokenization