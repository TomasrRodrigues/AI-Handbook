# Data Preparation

#### Table of Contents

1. [Introduction to Data Preparation for Pre-training](#introduction-to-data-preparation-for-pre-training)
2. [Data Sources and Collection](#data-sources-and-collection)
3. [Establishing Data Quality Standards](#establishing-data-quality-standards)
4. [Text Extraction and Preprocessing](#text-extraction-and-preprocessing)
5. [Data Cleaning and Filtering](#data-cleaning-and-filtering)
6. [Deduplication Strategies](#deduplication-strategies)
7. [Data Mixing and Composition](#data-mixing-and-composition)
8. [Advanced Preprocessing Techniques](#advanced-preprocessing-techniques)
9. [Tools, Frameworks and Infrastructure](#tools-frameworks-and-infrastructure)
10. [Best Practices and Emerging Trends](#best-practices-and-emerging-trends)


## Introduction to Data Preparation for Pre-training

The expression *"garbage in, garbage out"* has never been more relevant than in the context of large language models. While architectural innovations and training techniques receive significant attention in the research community, the foundation upon which all large language models are built is their training data. The **quality, diversity and scale** of this data directly determines the capabilities and limitations of the resulting model. A model trained on poorly prepared data will struggle with basic tasks, hallucinate incorrect information, and exhibit biases that reflect the flaws in its training corpus rather than genuine understanding of language.

Data preparation for pre-training is fundamentally different from preparing data for fine-tuning or other machine learning tasks. Pre-training aims to give the model broad knowledge about language, world facts, reasoning patterns and diverse domains. This requires processing enormous quantities of text - often measured in **trillions of tokens** - while maintaining sufficient quality that the model learns meaningful patterns rather than memorizing noise. The challenge lies in finding the sweet spot between scale and quality, between comprehensive coverage and focused relevance.

> Research from IBM suggests that nearly **80% of the time** spent on artificial intelligence projects goes toward data preparation.  
> No amount of computational power or algorithmic sophistication can compensate for fundamentally flawed training data.

The consequences of poor data preparation extend far beyond simple performance metrics. Models trained on unfiltered data can reproduce and amplify harmful biases present in their training corpus. They may learn to generate toxic content, violate privacy by memorizing personally identifiable information or produce outputs that reflect the worst aspects of internet discourse rather than helpful, harmless and honest responses. The infamous example of GPT-4chan, trained specifically on data from a notoriously toxic online forum, demonstrated how a model trained on poor quality data becomes essentially unusable regardless of its technical capabilities.

**Contrast example:**  
The TinyGSM project used a carefully curated synthetic dataset of elementary math problems with Python solutions. Despite training a relatively small model (1.3 billion parameters) on this specialized, high-quality data, the resulting model achieved **81.5% accuracy** on mathematical reasoning benchmarks—matching GPT-3.5 and outperforming many much larger models. This demonstrates that thoughtful data preparation can enable smaller models to punch far above their weight class.


### The Scale of Modern Pre-training

To understand the challenges of data preparation for pre-training, we must first grasp the sheer scale of modern language model training.

- Early models like GPT-2 were trained on datasets measuring in the **tens of billions of tokens**.
- Today's state-of-the-art models consume training datasets measuring in the **trillions of tokens**.
- The Llama 3 models, for instance, were trained on over **15 trillion tokens**.
- The open-source FineWeb dataset alone contains **15 trillion tokens** of cleaned web text.

This scale creates unique challenges:
- Manual review of trillions of tokens is impossible.
- Data cannot be stored on a single machine.
- Processing requires distributed systems spanning hundreds or thousands of machines.
- Even a **1% error rate** affects hundreds of billions of tokens.

The computational cost of processing data at this scale is substantial. Filtering, deduplication and quality assessment of trillion-token datasets requires significant computational resources. Companies and research labs invest millions of dollars not just in model training, but in the data preparation infrastructure that precedes it. This is why most organizations don't train foundation models from scratch but instead fine-tune existing pretrained models - the data preparation and pretraining stages require resources that only major technology companies and well-funded research institutions can afford.

> **Scale alone is insufficient.**  
> The relationship between dataset size and model quality is not linear.

Research on scaling laws, particularly the Chinchilla work from DeepMind, suggests an optimal ratio of roughly **20 training tokens for every model parameter**. A one billion parameter model should see approximately twenty billion tokens during training. Feeding it significantly more data yields diminishing returns, while using less data leaves capabilities untapped. As models grow larger, the data requirements grow proportionally, but the data must also grow in quality and diversity to justify the increased computational cost.

**FineWeb vs. RedPajama:**  
RedPajama contains approximately **20 trillion tokens** with relatively light filtering, while FineWeb has **15 trillion tokens** but with substantially more aggressive quality curation. Models trained on the smaller but cleaner FineWeb dataset consistently outperformed those trained on the larger RedPajama dataset. Beyond a certain scale, additional low-quality data becomes counterproductive, slowing learning and potentially teaching the model undesirable patterns.

### Quality vs Quantity Trade-offs

The tension between **quality and quantity** pervades every decision in data preparation for pre-training.

- **Aggressive filtering** improves average data quality but reduces dataset size.
- **Lighter filtering** preserves more data but includes lower quality content.

Finding the right balance requires understanding what quality means in the context of pre-training and how different filtering strategies affect downstream model performance.

Quality in pre-training data encompasses multiple dimensions. **Dimensions of quality:**
- Coherent, grammatically correct, and semantically meaningful text
- Factual accuracy
- Diversity of topics, writing styles, and perspectives

Different use cases demand different quality-quantity trade-offs:
- **General-purpose foundation models:** Prioritize scale and diversity, even at the cost of some quality.
- **Domain-specific models (medicine, law):** Accept smaller datasets in exchange for higher quality, domain-relevant content.

Assessing quality at scale requires automated methods, which are imperfect:
- Classifiers may exclude certain writing styles or topics, introducing bias.
- Heuristic filters (word count, punctuation) can remove junk but also exclude legitimate content.

**Empirical guidance:**
- Removing exact duplicates always improves performance.
- Fuzzy deduplication helps, with diminishing returns as the threshold becomes stricter.
- Filtering out obvious low-quality content (spam, gibberish, repetitive text) clearly benefits model quality.
- Beyond basics, additional filtering is context-dependent and requires careful ablation studies.

**Data efficiency:**  
Measures how much a model learns from each token of training data. Training on higher quality data tends to improve data efficiency—the model learns faster and reaches better performance with fewer training steps. This efficiency can offset the smaller dataset size that results from aggressive filtering.

> A model that trains 10% faster on data that's 20% smaller due to quality filtering may actually reach better final performance than a model trained on the full, unfiltered dataset.



## Data Sources and Collection

Building a comprehensive pre-training dataset requires aggregating text from diverse sources, each contributing different strengths to the model's knowledge and capabilities. The primary categories of general-purpose data have become well-established through years of research and industrial practice. Understanding the characteristics, advantages and challenges of each source is essential for assembling an effective training corpus.

Web text forms the backbone of most large language model training sets, providing unprecedented scale and diversity. Projects like Common Crawl maintain ongoing snapshots of the internet, archiving billions of web pages spanning countless domains, languages and topics. Common Crawl releases new snapshots regularly, each containing petabytes of HTML, metadata and extracted text from websites across the globe. This data source provides extraordinary breadth - any topic with a web presence appears somewhere in the corpus. 

However, the web data quality varies dramatically. Common Crawl includes everything from carefully edited Wikipedia articles and professional journalism to spam, malware-hosting pages and auto generated gibberish. The ration of high-quality to low-quality content is not favorable - much of the web consists of thin content, advertisements, navigation elements and other material that provides little learning value. Successful use of Common Crawl requires sophisticated filtering pipelines that identify and preserve the valuable subset while discarding the noise.

Different organizations have produced curated versions of Common Crawl with varying filtering strategies. The C4 dataset applied relatively conservative filtering, removing pages with certain spam indicators while preserving most content. RefinedWeb used more agressive cleaning and deduplication, producing a smaller but higher-quality corpus. FineWeb pushed quality filtering even further, resulting in fifteen trillion tokens that demonstrably improve model perforrmance compared to less filtered alternatives. Each represents a different position on the quality-quantity spectrum.

Books provide a complementary source of training data with distinct characteristics from web text. Unlike the often fragmentary and informal nature of web content, books offer long-form, carefully edited text with coherent narrative structure. They demonstrate proper grammar, varied vocabulary, and complex sentence structures. Books also provide knowledge that may not appear extensively on the internet - historical analysis, in-depth technical treatises, literary fiction that explores human psychology and culture.

Datasets like Books3 and BookCorpus, both components of The Pile dataset, aggregate tens of thousands of books spanning fiction, non-fiction, academic texts, and more. The extended context that books provide helps models learn to track information and maintain coherence across long passages, a capability essential for tasks requiring understanding of multi-paragraph contexts. However, books present challenges around copyright - most modern books are protected intellectual property that cannot legally be included in publicly released datasets. This constraint means book corpora tend to emphasize older works in the public domain or works released under permissive licenses, potentially creating gaps in contemporary knowledge.

Conversational data from platforms like Reddit brings yet another dimension to pre-training corpora. Online discussions capture informal language, slang, and the back-and-forth dynamics of human conversation. They demonstrate how people ask questions, provide answers, express opinions, and engage in debate. Conversation data helps models understand dialogue structure and respond more naturally in interactive settings.

The PushShift.io Reddit corpus provides comprehensive archives of Reddit discussions, organized by subreddit and preserving the threaded structure of comments and replies. Processors must decide how to flatten these conversational trees into training sequences - do you follow individual conversation branches, or do you sample across the full discussion? The optimal approach depends on your training objectives. Conversational data also requires careful filtering. Many online discussions contain toxic content, misinformation, and low-effort posts that contribute little learning value while potentially teaching harmful patterns.

Scientific and academic sources represent critical knowledge that doesn't appear extensively in general web text or fiction. Datasets drawn from arXiv preprints, PubMed abstracts, academic books, and research databases provide technical vocabulary, mathematical reasoning, and domain expertise across scientific fields. This content helps models develop capabilities in technical domains and improves performance on tasks requiring specialized knowledge.

However, scientific text presents unique preprocessing challenges. Mathematical notation, chemical formulas, protein sequences, and other specialized symbols require special handling during tokenization. LaTeX markup from scientific papers must be parsed to extract the underlying content. Tables, figures, and references need appropriate representation. Despite these complications, the value of scientific data for improving model capabilities in technical reasoning and domain expertise justifies the additional preprocessing effort.

### Speciliazed Data Sources

Beyond general-purpose data, specialized sources target specific capabilities that general text may not adequately cover. Code represents perhaps the most important specialized data source, given the widespread use of language models for programming assistance. Code repositories like GitHub provide massive quantities of programming examples across dozens of languages, along with documentation, comments, and discussions about programming problems.

Models trained on code demonstrate several valuable capabilities beyond just program synthesis. They develop better logical reasoning skills, as code inherently requires precise logical thinking. They improve at following structured specifications, since code must adhere to formal syntax rules. They learn to work with symbolic representations and abstract patterns. The StarCoder and Codex models demonstrate that including substantial code in pre-training significantly improves performance on both programming and general reasoning tasks.

Processing code data requires specialized considerations. Unlike natural language, code has strict syntactic requirements - small errors can make code completely non-functional. Code quality varies enormously across repositories, from well-tested production code to student exercises to abandoned experiments. Effective code filtering considers metrics like whether the code compiles, static analysis quality scores, licensing information, and whether the code appears to be malware or exploit code. The Stack and StarCoder datasets provide carefully filtered code corpora suitable for model training.

Multillingual data enables models to work across language boundaries. While much research focuses on English given its dominant presence in training data, truly capable models need representation from many language. Wikipedia provides high-quality text in hundreds of languages, offering encyclopedic content that benefits from consistent editorial standards across languages. News sources, government documents and cultural materials provide additional multilingual coverage.

Language-specific considerations complicate multilingual training. Some languages have abundant web presence while others have limited digital text available. Languages with different scripts require tokenizers that handle their character sets efficiently. Balancing representation across languages involves trade-offs - should we allocate training capacity proportionally to available data or should we oversample lower-resource languages to improve their representation despite the smaller corpus size? Modern practices tend toward oversampling lower-resource languages to ensure the model develops capabilities across all target languages rather than treating non-English languages as afterthoughts.

Domain-specific data targeting particular industries or applications enables specialized models or improves performance in targeted areas. Medical literature, legal documents, financial reports, and technical manuals all provide domain expertise that general web text may touch only superficially. For continual pretraining or domain adaptation, focused domain data can efficiently specialize a general model without full retraining.

However, domain data often comes with restrictions. Medical records contain sensitive patient information requiring strict privacy protection. Legal documents may be privileged or confidential. Financial data may be proprietary. Even when domain data is accessible, it may be in formats requiring specialized extraction - medical notes in electronic health record systems, legal briefs in court databases, regulatory filings in government repositories. Successfully incorporating domain data requires addressing these access, privacy, and formatting challenges.


### Ethical and Legal Considerations in Data Collection

Data collection for language models operates in a complex and evolving legal landscape. Copyright law, privacy regulations, terms of service agreements, and ethical considerations all constrain what data can legitimately be collected and how it can be used. Navigating these constraints requires understanding the legal framework, adopting responsible collection practices, and maintaining transparency about data sources.

Copyright presents perhaps the most significant legal challenge. Simply because text appears publicly on the internet does not mean it can be freely used for commercial purposes. News articles, books, creative works, and many other categories of content are protected by copyright. Using copyrighted material for machine learning training occupies a legally uncertain space in many jurisdictions. Some argue that training constitutes fair use or falls under text and data mining exceptions. Others contend that it infringes on copyright holder's rights. Multiple lawsuits against companies training models on copyrighted content remain pending as of 2025, with potential implications for the entire field.

Responsible data collection requires respecting copyright and working within legal boundaries. This means prioritizing public domain content, openly licensed materials, and content from sources that explicitly permit machine learning use. It means being cautious about including copyrighted books, news articles, and creative works without permission. For commercial applications, it often means seeking licensing agreements with content creators and publishers. The legal uncertainty means practices must evolve as case law develops and regulations emerge.

Privacy and personally identifiable information represent another critical concern. Training data scraped from the internet inevitably contains personal information - names, addresses, phone numbers, email addresses, and more sensitive information. Models trained on data containing PII can memorize and reproduce that information, creating privacy violations and compliance issues. Regulations like GDPR in Europe and CCPA in California impose strict requirements on how personal data is collected and used.

Responsible practice requires identifying and removing or redacting PII from training data. This involves both automated detection using regular expressions and named entity recognition models, and filtering out data sources likely to contain sensitive information. Social media posts, forum discussions, and user-generated content require particularly careful handling as they frequently include personal information. Some organizations apply differential privacy techniques during training to provide mathematical guarantees that individual examples can't be extracted from the trained model.

Web scraping itself raises legal and ethical questions beyond copyright and privacy. Websites establish terms of service that may prohibit automated data collection. Robots.txt files signal which parts of a site web crawlers should avoid. Aggressive scraping can burden servers, degrading service for legitimate users. Responsible scraping requires respecting these boundaries, identifying your crawler to site operators, implementing rate limiting to avoid overwhelming servers, and honoring robots.txt directives.

The robots.txt standard provides machine-readable instructions about what automated agents may access. A typical robots.txt file might disallow certain directories or paths, specify crawl delays, or block specific user agents. While not legally binding in most jurisdictions, violating robots.txt is generally considered unethical and may constitute trespass or violate computer fraud laws. Responsible crawlers always check and obey robots.txt, implement the specified delays, and clearly identify themselves in user-agent strings.

Content quality and societal impact introduce ethical dimensions beyond legal compliance. Training data inevitably reflects biases, stereotypes, and problematic content present in human-generated text. Models trained on unfiltered web data absorb these patterns, potentially amplifying harmful biases or generating toxic content. While complete neutrality is impossible and attempting to remove all controversial content would create its own biases, some filtering of the most harmful content is necessary.

Toxicity filtering aims to reduce the most egregious harmful content without eliminating legitimate discussions of difficult topics. This requires sophisticated approaches that can distinguish between hate speech and discussions about hate speech, between promoting violence and reporting on violence. Modern toxicity classifiers use machine learning models trained on annotated examples to make these fine-grained distinctions. However, no filter is perfect, and the decision of where to set thresholds involves balancing multiple considerations.

Transparency about data sources and collection practices has emerged as a key principle. Publishing documentation about what data was collected, from what sources, with what filtering applied, allows others to understand model capabilities and limitations. It enables reproducibility of research results. It helps identify and address problems when they arise. The trend toward transparency is embodied in efforts like detailed dataset documentation, data statements, and model cards that describe training data characteristics.


## Establishing Data Quality Standards


## Text Extraction and Preprocessing


## Data Cleaning and Filtering


## Deduplication Strategies


## Data Mixing and Composition


## Advanced Preprocessing Techniques


## Tools, Frameworks and Infrastructure


## Best Practices and Emerging Trends

