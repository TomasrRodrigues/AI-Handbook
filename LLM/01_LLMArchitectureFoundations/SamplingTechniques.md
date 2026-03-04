# Sampling Techniques

#### Table of Contents

1. [Introduction](#introduction)
2. [Understanding the Probability Distribution](#understanding-the-probability-distribution)
3. [Greedy Decoding](#greedy-decoding)
4. [Temperature](#temperature)
5. [Top-K Sampling](#top-k-sampling)
6. [Top-P (Nucleus)](#top-p-nucleus)
7. [Beam Search](#beam-search)
8. [Practical Configuration](#practical-configuration)

## Introduction

When we interact with a large language model, requesting it to write a story, answer a question, or generate code, what we see as output is the result of **thousands of individual decisions**. At each step of generation, the model doesn't simply produce a word - it produces a **probability distribution over its entire vocabulary**, assigning a likelihood to each of the tens of thousands of tokens it knows. 

The question of how to select from this distribution of possibilities, how to navigate the trade-off between predictability and creativity, between safety and surprise, is what **sampling techniques** address. This choice profoundly shapes every aspect of the generated text, from its coherence and accuracy to its creativity and diversity.

## Understanding the Probability Distribution

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

## Greedy Decoding

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

## Temperature

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

## Top-K Sampling

While temperature adjusts how we weight probabilities, it doesn't change which tokens are considered. Even at low temperature, there's a nonzero probability of selecting extremely unlikely tokens like $\text{"quantum"}$ after $\text{"The cat sat on the"}$. Top-k sampling addresses this by imposing a **hard cutoff**: only consider the k most probable tokens and ignore everything else.

**How it works**: 
1. Look at the probability distribution and identify the $k$ tokens with highest probabilities
2. Set the probabilities of all other tokens to zero
3. Renormalize the remaining probabilities so they sum to one
4. Sample from this restricted distribution

The parameter $k$ typically ranges from 10 to 100, though it can be tuned for specific applications.

For example, with $k=10$ for our example, we might keep $\text{"mat"}$, $\text{"floor"}$, $\text{"chair"}$, $\text{"table"}$, $\text{"roof"}$, $\text{"carpet"}$, $\text{"ground"}$, $\text{"bed"}$, $\text{"couch"}$ and $\text{"wall"}$. We eliminate $\text{"computer"}$, $\text{"quantum"}$ and 49,980 other tokens. The probabilities of the ten kept tokens are rescaled to sum to one, and we sample from this smaller set. This guarantees that we never select completely implausible tokens while still allowing some variety in generation. 

The value of $k$ creates a clear trade-off:
- **Small $k$** (5-10): Keeps generation focused on the most probable options, leading to safe, predictable text
- **Large $k$** (50-100): Allows more exploration and creativity but includes tokens that might be inappropriate or nonsensical
- **$k = 1$**: Equivalent to greedy decoding - we always pick the single most likely token
- **$k =$ vocabulary size**: Equivalent to sampling without any top-k restriction

One significant limitation of top-k sampling is that $k$ is fixed regardless of the model's confidence:
- **When the model is very certain**: After "To be or not to," the next token is almost certainly "be." In such cases, restricting to the top 10 tokens is unnecessary; even the top 3 would suffice.
- **When the model is uncertain**: At the beginning of a creative story, many different openings are equally plausible. Restricting to only 10 tokens might exclude many reasonable possibilities.

This inflexibility motivated the development of top-p sampling, which adapts to the model's confidence level.

## Top-P (Nucleus)

Top-p sampling, also known as **nucleus sampling** after the paper that introduced it in 2019, addresses the inflexibility of top-k by using a **cumulative probability threshold** rather than a fixed number of tokens. Instead of always keeping exactly $k$ tokens, we keep however many tokens are needed to reach a cumulative probability of $p$. 

The algorithm works as follows:
1. Sort all tokens by their probability in descending order
2. Start with the most probable token and keep adding tokens to our sampling pool, tracking the cumulative probability, until we reach or exceed $p$
3. All these tokens are kept, and everything else is excluded
4. Renormalize the kept tokens' probabilities to sum to one and sample from this set

For example, with $p=0.9$ for our example, we would start:
- Start with "mat" at probability 0.31
- Add "floor" bringing cumulative probability to 0.56
- Add "chair" for 0.74
- Add "table" for 0.84
- Add "roof" for 0.86
- Continue adding tokens until the cumulative probability exceeds 0.9

We might end up with six or seven tokens total. The key insight is that **this number varies depending on the probability distribution's shape**.

**Adaptive behavior based on model confidence**:

**When the model is very confident** (e.g., "To be or not to"):
- The token "be" might have probability 0.95
- With $p = 0.9$, we would include only "be" because it alone exceeds the threshold
- The sampling pool automatically becomes small when the model is certain

**When the model is uncertain** (e.g., at the beginning of a story):
- Probabilities might be spread across many tokens
- To reach cumulative probability 0.9, we might need to include 30 or 40 tokens
- The sampling pool automatically becomes larger when the model is uncertain

This adaptive behavior is top-p's greatest strength. The model gets to be conservative when it's confident and exploratory when it's not, which aligns with our intuitions about how uncertainty should affect generation. In practice, top-p has proven remarkably effective, becoming one of the most popular sampling methods. Typical values range from **0.9 to 0.95**, with higher values allowing more diversity and lower values producing more focused outputs.

Top-p is often combined with top-k as a "belt and suspenders" approach. You might specify both $k = 50$ and $p = 0.9$, meaning the model first restricts to the top 50 tokens and then further restricts to tokens needed to reach cumulative probability 0.9. This provides both an **absolute safety bound** (never consider more than 50 tokens, no matter how flat the distribution) and an **adaptive bound** that respects the model's confidence level.

## Beam Search

All the sampling methods we've discussed so far make decisions one token at a time. We generate probabilities for the next token, apply temperature and top-k or top-p filtering, sample a token and move on. The choice at step $t$ is made without considering how it will affect options at steps $t+1$, $t+2$ and beyond. This myopic approach sometimes leads to suboptimal sequences where a locally good choice creates globally poor outcomes.

Beam search addresses this by maintaining multiple candidate sequences simultaneously and selecting based on overall likelihood rather than greedy local decisions. Instead of tracking one partial generation, beam search tracks $b$ sequences, where $b$ is the beam width. At each step, it expands each sequence by considering all possible next tokens, evaluates the likelihood of all resulting sequences and keeps only the $b$ most likely overall.

The likelihood of a sequence is computed as the product of the probabilities of each token, or equivalently and more practically, as the sum of log probabilities to avoid numerical underflow. 

For example:
- Sequence "The cat sat on the mat" might have log probability: $-2.3 + (-1.8) + (-0.9) + (-1.2) + (-0.7) + (-1.5) = -8.4$
- Another sequence "The cat perched on the windowsill" might have log probability $-8.9$
- The first sequence has higher likelihood overall even though individual tokens in the second sequence might have been more probable in isolation

Beam search is particularly effective for tasks where there's a clear correct answer that we want to find by exploring multiple paths. **Machine translation** is the classic application. Given a sentence in French, there's typically one best translation to English, and beam search helps find it by considering multiple translation candidates and selecting the one with highest overall probability. This prevents the model from committing to an early word choice that makes good later translations impossible.

However, beam search has important limitations for open-ended generation:
- **Computationally expensive**: Effectively multiplying the cost of generation by the beam width
- **No streaming**: Can't be easily used for streaming generation where tokens are displayed as they're produced, since you don't know which beam was best until generation completes
- **Optimizes for likelihood, not quality**: Most importantly, beam search optimizes for high likelihood, but high likelihood doesn't always mean high quality for creative tasks

The most probable text according to the model is often generic and boring. "The cat sat on the mat" might have higher probability than "The cat perched precariously on the windowsill," but the second is more interesting and specific. For creative writing, conversation, or storytelling, sampling methods that inject randomness often produce better results than beam search, even though the sampled outputs have lower likelihood. Understanding when to use beam search versus sampling methods is crucial for effective deployment.

## Practical Configuration

In practice, generating high-quality text requires skillfully combining multiple sampling techniques and understanding how their parameters interact. The ideal configuration varies dramatically depending on the task, the model, and the desired output characteristics. 

Let's explore how to configure sampling for different real-world applications:

**Factual question answering** (accuracy is paramount):
- **Temperature**: 0.2
- **Top-p**: 0.9
- **Beam search**: No (streaming responsiveness is more important)
- **Why it works**: The very low temperature keeps the model focused on high-probability, likely-to-be-correct tokens. The top-p provides a safety mechanism, excluding the long tail of improbable tokens while adapting to the model's confidence. This configuration produces consistent, accurate responses without being completely mechanical.

**Creative writing** (variation and surprise):
- **Temperature**: 0.8-0.9
- **Top-p**: 0.95
- **Additional**: Frequency penalty to discourage repetition
- **Why it works**: The higher temperature encourages the model to explore less probable word choices, leading to more interesting and varied prose. The higher top-p allows consideration of a wider vocabulary, letting the model use less common words that might be more evocative. Frequency penalty helps prevent the model from falling into repetitive patterns that can make creative writing feel stilted.

**Code generation** (syntactic correctness is essential):
- **Temperature**: 0.2
- **Top-p**: 0.95
- **Top-k**: 40 (additional safety bound)
- **Why it works**: Code has strict grammatical rules - a missing bracket or incorrect indentation breaks everything - so you want the model to stick very close to high-probability, syntactically valid completions. Even small amounts of randomness can introduce bugs that make generated code unusable. The low temperature configuration minimizes this risk while still allowing some flexibility.

**Dialogue systems and chatbots** (depends on personality and goals):
- **Customer service bot**: Temperature 0.3 (consistency and reliability - users expect accurate, helpful responses, not creative interpretation)
- **Companion chatbot**: Temperature 0.7-0.8 with broader top-p (more varied and engaging conversation)
- **Therapeutic chatbot**: Temperature 0.5 (supportive and coherent without being completely predictable)

Some advanced Techniques include:

**Dynamic temperature adjustment**: Advanced practitioners sometimes adjust temperature dynamically during generation:
- Start with lower temperature to establish a coherent direction
- Increase temperature as generation continues to add variation and prevent monotony
- For code: Use lower temperature for function signatures and control structures where correctness is critical, then slightly higher temperature for comments or variable names where more flexibility is acceptable
- This allows the model to be conservative when precision matters and creative when variation is desirable

**Model scale considerations**: Another important consideration is how sampling parameters interact with model scale:
- **Larger models** (e.g., 175B parameters) generally produce better-calibrated probability distributions - their confidence estimates more accurately reflect true likelihoods
- Can be used effectively with higher temperatures without degrading into nonsense as quickly as smaller models
- Might produce coherent text at temperature 1.0
- **Smaller models** (e.g., 1B parameters) might need temperature 0.6 to maintain similar coherence
- Understanding your specific model's capabilities is essential for setting appropriate parameters

The field continues to evolve with new sampling techniques:
- **Contrastive decoding**: Explicitly avoids patterns seen in undesirable outputs
- **Classifier-guided generation**: Uses an auxiliary model to steer generation toward desired attributes like sentiment or topic
- **Mirostat**: Dynamically adjusts parameters to maintain a target level of randomness throughout generation

As our understanding deepens, we continue developing more sophisticated ways to extract high-quality outputs from these powerful but complex systems.

The fundamental insight underlying all sampling techniques is that **language generation is inherently probabilistic with genuine uncertainty**. Different sampling strategies represent different philosophies about navigating this uncertainty:
- **Greedy decoding and beam search**: Try to find the "best" output by maximizing likelihood
- **Temperature sampling**: Embraces controlled randomness
- **Top-k and top-p**: Filter implausible options while leaving room for variety

The art of working with language models lies in understanding these trade-offs and choosing the right balance for your particular application - balancing creativity with coherence, determinism with diversity, safety with surprise.


<div style="display: flex; justify-content: space-between;">
  <a href="Tokenization.md">Back</a>
</div>