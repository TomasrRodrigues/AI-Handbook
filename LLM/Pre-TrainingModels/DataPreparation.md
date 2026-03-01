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
The TinyGSM project used a carefully curated synthetic dataset of elementary math problems with Python solutions. Despite training a relatively small model (1.3 billion parameters) on this specialized, high-quality data, the resulting model achieved **81.5% accuracy** on mathematical reasoning benchmarks-matching GPT-3.5 and outperforming many much larger models. This demonstrates that thoughtful data preparation can enable smaller models to punch far above their weight class.


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
Measures how much a model learns from each token of training data. Training on higher quality data tends to improve data efficiency-the model learns faster and reaches better performance with fewer training steps. This efficiency can offset the smaller dataset size that results from aggressive filtering.

> A model that trains 10% faster on data that's 20% smaller due to quality filtering may actually reach better final performance than a model trained on the full, unfiltered dataset.



## Data Sources and Collection

Building a comprehensive pre-training dataset requires aggregating text from diverse sources, each contributing different strengths to the model's knowledge and capabilities. The primary categories of general-purpose data have become well-established through years of research and industrial practice. Understanding the characteristics, advantages and challenges of each source is essential for assembling an effective training corpus.

### Web Text: The Backbone of Scale

Web text forms the backbone of most large language model training sets, providing unprecedented scale and diversity. Projects like **Common Crawl** maintain ongoing snapshots of the internet, archiving billions of web pages spanning countless domains, languages and topics. Common Crawl releases new snapshots regularly, each containing petabytes of HTML, metadata and extracted text from websites across the globe. This data source provides extraordinary breadth - any topic with a web presence appears somewhere in the corpus. 

However, the web data quality varies dramatically. Common Crawl includes everything from carefully edited Wikipedia articles and professional journalism to spam, malware-hosting pages, and auto-generated gibberish. The ratio of high-quality to low-quality content is not favorable-much of the web consists of **thin content, advertisements, and navigation elements** that provide little learning value. Successful use of Common Crawl requires sophisticated filtering pipelines that identify and preserve the valuable subset while discarding the noise.

**The Quality Spectrum of Web Datasets**:
- **C4**: Applied relatively conservative filtering, removing pages with certain spam indicators while preserving most content.
- **RefinedWeb**: Used more agressive cleaning and deduplication, producing a smaller but higher-quality corpus
- **FineWeb**: Pushed quality filtering even further, resulting in **15 trillion tokens** that demonstrably improve model performance compared to less filtered alternatives. 

### Books and Long-form Coherence

Books provide a complementary source of training data with distinct characteristics from web text. Unlike the often fragmentary and informal nature of web content, books offer **long-form, carefully edited text** with coherent narrative structure. They demonstrate proper grammar, varied vocabulary, and complex sentence structures. Books also provide knowledge that may not appear extensively on the internet - historical analysis, in-depth technical treatises, literary fiction that explores human psychology and culture.

Datasets like **Books3** and **BookCorpus** (both components of The Pile dataset) aggregate tens of thousands of books spanning fiction, non-fiction, academic texts, and more. The extended context that books provide helps models learn to track information and maintain coherence across long passages, a capability essential for tasks requiring understanding of multi-paragraph contexts. 

However, books present challenges around **Copyright Constraints** - most modern books are protected intellectual property that cannot legally be included in publicly released datasets. This constraint means book corpora tend to emphasize older works in the **public domain** or works released under permissive licenses, potentially creating gaps in contemporary knowledge.

Conversational data from platforms like **Reddit** brings yet another dimension to pre-training corpora. Online discussions capture informal language, slang, and the back-and-forth dynamics of human conversation. They demonstrate how people ask questions, provide answers, express opinions, and engage in debate. Conversation data helps models understand dialogue structure and respond more naturally in interactive settings.

The risk is that many online discussions contain toxic content, misinformation and low-effort posts that contribute little learning while potentially teaching harmful patterns.

Scientific and academic sources represent critical knowledge that doesn't appear extensively in general web text or fiction. Datasets drawn from **arXiv preprints**, **PubMed abstracts**, **academic books**, and **research databases** provide technical vocabulary, mathematical reasoning, and domain expertise across scientific fields. This content helps models develop capabilities in technical domains and improves performance on tasks requiring specialized knowledge.

However, scientific text presents unique preprocessing challenges:
- **Notation**: Mathematical notation and chemical formulas require special handling during tokenization
- **Formatting**: LaTeX markup must be parsed to extract underlying content while maintaining meaning.
- **Structure**: Tables, figures and references need appropriate representation to be useful.

Despite these complications, the value of scientific data for improving model capabilities in **technical reasoning** and domain expertise justifies the additional preprocessing effort.

### Speciliazed Data Sources

Beyond general-purpose data, specialized sources target specific capabilities that general text may not adequately cover: 

#### 1. Source Code

Code represents perhaps the most important specialized data source, given the widespread use of language models for programming assistance. Code repositories like GitHub provide massive quantities of programming examples across dozens of languages, along with documentation, comments, and discussions about programming problems.

Code inherently requires precise logical thinking, which improves the model's general reasoning as it helps models to learn to follow formal syntax rules and abstract patterns.

The **StarCoder** and **Codex** models demonstrate that including substantial code in pre-training significantly improves performance on both programming and general reasoning tasks.

Processing code data requires specialized considerations. Unlike natural language, code has strict syntactic requirements - small errors can make code completely non-functional. Code quality varies enormously across repositories, from well-tested production code to student exercises to abandoned experiments. Effective code filtering considers metrics like whether the code compiles, static analysis quality scores, licensing information, and whether the code appears to be malware or exploit code. The Stack and StarCoder datasets provide carefully filtered code corpora suitable for model training.


#### 2. Multilingual Data

Multillingual data enables models to work across language boundaries. While much research focuses on English given its dominant presence in training data, truly capable models need representation from many language. Wikipedia provides high-quality text in hundreds of languages, offering encyclopedic content that benefits from consistent editorial standards across languages.

**Oversampling**: Modern practices tend toward oversampling lower-resource languages to ensure the model develops capabilities across all target languages rather than treating non-English languages as afterthoughts.

#### 3. Domain-Specific Data

Domain-specific data targeting particular industries or applications enables specialized models or improves performance in targeted areas. Medical literature, legal documents, financial reports, and technical manuals all provide domain expertise that general web text may touch only superficially. For continual pretraining or domain adaptation, focused domain data can efficiently specialize a general model without full retraining.

However, domain data often comes with **access restrictions**. Medical records contain sensitive patient information requiring strict privacy protection. Legal documents may be privileged or confidential. Financial data may be proprietary. Even when domain data is accessible, it may be in formats requiring specialized extraction - medical notes in electronic health record systems, legal briefs in court databases, regulatory filings in government repositories. Successfully incorporating domain data requires addressing these access, privacy, and formatting challenges.


### Ethical and Legal Considerations in Data Collection

Data collection for language models operates in a complex and evolving landscape. Copyright law, privacy regulations, and ethical considerations all constrain what data can legitimately be collected.

#### Copyright and Fair Use

Copyright presents perhaps the most significant legal challenge. Using copyrighted material for machine learning training occupies a **legally uncertain space**. Some argue that training constitutes "fair use", while others content it infringes on rights. Multiple lawsuits remain pending as of 2025, with potential implications for the entire field.

Responsible data collection requires respecting copyright and working within legal boundaries. This means prioritizing public domain content, openly licensed materials, and content from sources that explicitly permit machine learning use. It means being cautious about including copyrighted books, news articles, and creative works without permission. For commercial applications, it often means seeking licensing agreements with content creators and publishers.

#### Privacy and PII

Privacy and personally identifiable information represent another critical concern. Training data scraped from the internet inevitably contains **Personally Identifiable Information (PII)** (like names, addresses, phone numbers, email addresses, and more sensitive information). Models trained on data containing PII can memorize and reproduce that information, creating privacy violations and compliance issues. 
For this reason, regulations like GDPR in Europe and CCPA in California impose strict requirements on how personal data is collected and used.

Responsible practice requires identifying and removing or redacting PII from training data. This involves both automated detection using regular expressions and **Named Entity Recognition (NER)** models, and filtering out data sources likely to redact sensitive info.

#### Web Scraping Ethics: robots.txt

Web scraping itself raises legal and ethical questions beyond copyright and privacy. Websites establish terms of service that may prohibit automated data collection. **robots.txt** files signal which parts of a site web crawlers should avoid. 

Aggressive scraping can burden servers, degrading service for legitimate users. Responsible scraping requires respecting these boundaries, identifying your crawler to site operators, implementing rate limiting to avoid overwhelming servers, and honoring robots.txt directives.

Crawlers should clearly identify themselves in user-agent strings. While not legally binding in all jurisdictions, violating robots.txt is generally considered unethical.

#### Toxicity and Transparency

Content quality and societal impact introduce ethical dimensions beyond legal compliance. Training data inevitably reflects **biases**, **stereotypes**, and **problematic content present in human-generated text**. 

***Models trained on unfiltered web data absorb these patterns, potentially amplifying harmful biases or generating toxic content.***

While complete neutrality is impossible and attempting to remove all controversial content would create its own biases, some filtering of the most harmful content is necessary:
- **Toxicity filtering** aims to reduce harmful content without eliminating legitimate discussions of difficult topics. This requires sophisticated approaches to distinguish between hate speech and discussions about hate speech. 
Modern toxicity classifiers use **machine learning** models trained on annotated examples to make these distinctions. However, **no filter is perfect**, and the decision of where to set thresholds involves balancing multiple considerations.
- **Transparency** has emerged as a key principle. Publishing documentation (such as **Model Cards** or **Data Statements**) allows others to understand model limitations and enables the reproducibility of research results.


## Establishing Data Quality Standards

Before collecting and filtering data, it's essential to establish clear standards for what constitutes high-quality training data. These standards guide every subsequent decision in the data preparation pipeline. Research has identified **thirteen key characteristics** of high-quality training datasets for large language models: 
- Reliability & Relevance
- Accuracy & Compliance
- Accessibility & Privacy Protection
- Documentation & Large-scale data
- Diversity & Knowledge content
- Wide range of sources
- Absence of low-quality documents
- Absence of toxic data

These characteristics can be organized into **four main categories** that provide a framework for thinking about data quality: 
1. **Data Reliability and Trustworthiness**: Ensuring the corpus represents real-world facts from credible sources.
2. **Data Scope and Coverage**: Providing both breadth across topics and depth in important domains.
3. **Data Cleanliness**: Removing harmful, toxic, or meaningless content.
4. **Data Governance and Compliance**: Ensuring collection follows legal and ethical guidelines.

### Reliability and Trustworthiness

Reliability and trustworthiness begin with **source selection**. Not all sources are equally valuable for training language models. Carefully curated corpora like Wikipedia, peer-reviewed academic papers, and professional journalism provide higher signal-to-noise ratios than unmoderated user-generated content. 

The reliability of a source affects how much the model should learn from it. A medical textbook should carry **more weight** than an anonymous health blog. A government statistical report should be trusted more than a random claim on social media.

Assessing source reliability at scale requires both automated methods and human judgment. Cross-verification against multiple sources helps identify likely misinformation. Accuracy concerns not just the readability of sources but the **factual correctness of individual claims**. Training data containing misinformation or outdated information teaches models to reproduce these errors. 

Practical approaches to imrpoving accuracy combine source reliability assessment with specific error detection: 
- Filtering out known misinformation domains
- Detecting inconsistencies within documents 
- Removing content that contradicts verified facts 

These help reduce but not eliminate inaccuraccies. Some inaccuracy is inevitable in very large corpora and models must develop robustness to occasional incorrect information. The goal is ensuring the overall signal of accurate information dominates the noise of errors.

### Data Scope and Coverage

The scope and coverage of training data determines what knowledge and capabilities the model can develop. Narrow, homogeneous data produces narrow, specialized models.

#### Breadth

**Breadth** of coverage means including text from diverse domains, genres, topics and perspectives. The training corpus should reflect the full scope of human knowledge and language use. This diversity **helps the model generalize across contexts**, preventing **overfitting** to particular patterns or topics.

Measuring diversity presents challenges at trillion-token scale: 
- **Topic modeling** can identify the distribution of subjects in the corpus. 
- **Vocabulary analysis** can assess linguistic range. 
- **Source attribution** can track how balanced the dataset is across different origins. 

These metrics provide rough indicators, though they can't fully capture the nuanced concept of true diversity.

#### Depth

***Depth of coverage ensures sufficient examples in important domains***. 

While breadth **prevents the model from being too narrow**, depth **prevents it from being too shallow**. Having ten thousand high-quality examples of mathematical reasoning provides better learning signal than having a single example of ten thousand different math problems. Depth allows the model to learn patterns and develop real understanding rather than just superficial familiarity. 

Balancing breadth and depth requires prioritization. **Not all topics deserve equal representation**. Scientific knowledge, programming, mathematics and other areas requiring specialized understanding benefit from substantial depth. Meanwhile, extremely obscure topics may not warrant significant corpus allocation. The **optimal balance depends on intended use cases** - a general-purpose model needs broad coverage with moderate depth in many areas, while a specialized model might sacrifice breadth for deep coverage of its domain.

#### Temporal

**Temporal coverage** adds another dimension. Language evolves, facts change, and new knowledge emerges. Training data that over-represents older content may teach outdated information or archaic language patterns. Conversely, focusing only on very recent data sacrifices historical knowledge and long-term context. 

***Temporal Coverage balances historical knowledge with current events and evolving language.***

The optimal temporal distribution depends on use cases - models for current events need fresh data, while models for historical analysis benefit from older content.

Data from different time periods also helps models understand change and context. Training on documents from the 1990s through 2020s allows the model to learn how technology terms evolved, how social norms shifted and how language usage changed. This temporal breadth improves the model's ability to interpret content from different eras and understand references to historical events.

### Data Cleanliness and Quality Control

Even "good" sources contains numerous types of noise and problematic content that must be addressed. **Data cleaning removes this noise while preserving valuable signal**. The challenge lies in **identifying what constitutes noise versus legitimate content** and implementing cleaning methods that scale to trillions of tokens.

#### Identifying Low-Quality Documents

Low-quality documents represent perhaps the most straightforward category to filter. **Gibberish**, **spam**, **ads**, **cookie notices**, **navigation menus** and **other web artifacts** provide no learning value while consuming computational resources. 

- **Heuristic filters** based on simple metrics can **identify** many obviously poor documents. Documents that are too short to convey meaningful content, have extremely low word counts or consist primarily of numbers or special characters are likely noise.
- **Repetition patterns** also indicate low quality. Documents that repeat the same phrase, sentence, or paragraph multiple times are often auto-generated spam or template-based content. **N-gram repetition** filters comput the fraction of a document consisting of repeated sequences. High repetition rates suggest low information density. However, thresholds must be set carefully - some legitimate content like legal documents or instruction manuals contains necessary repetition.

#### Toxic and Harmful Content

**Toxic and harmful content requires removal** both for ethical reasons and model quality. Hate speech, graphic violence, extreme profanity and content promoting illegal activities should generally be filtered. However, defining toxicity precisely is challenging - **context matters enormously**. Discussing hate speech to condemn it differs from using hate speech. Effective toxicity filtering requires nuanced models that can make these distinctions.

Modern toxicity classifiers (like **Detoxify Library**) use transformer-based models trained on human-annotated examples to identify harmful content across multiple categories. They produce probability scores for different types of problematic content, allowing configurable thresholds based on severity and use case. However, no classifier is perfect and agressive filtering risks removing legitimate content, particularly discussions of difficult topics or content from marginalized communities.

#### Personally Identifiable Information (PII)

**Personally identifiable information (PII)** creates both legal and ethical imperatives for removal. Names, addresses, phone numbers, email addresses, social security numbers and other PII appear frequently in web text. Models that memorize and reproduce this information create privacy violations.

PII detection combines **Regular Expression patterns** for structured information like phone numbers and social security numbers with **Named Entity Recognition (NER)** models for unstructured information like names and locations. Different types of PII may warrant different handling - common names might be retained while rare names are removed, public figures might be retained while private individuals are removed. The appropriate strategy depends on legal requirements and intended use.

#### Malformed Content

**Malformed content** from encoding errors, extraction problems or file corruption must be identified and removed. Unicode errors create garbled text that provides no learning signal. HTML extraction failures leave markup tags mixed with content text. PDF parsing errors produce scrambled characters or missing text. Detection requires checking for excessive special characters, invalid UTF-8 sequences, unexpected character patterns and other indicators of malformed text.

#### Language Filtering

**Language filtering** enables monolingual or controlled multillingual datasets. While some applications benefit from multillingual training data, others need specific language subsets. Language identification models can classify text into languages with high accuracy. Subsequent filtering removes content in unwanted languages. For highly multillingual corpora, language identification also enables language-specific processing steps.

## Text Extraction and Preprocessing

Training data comes in myriad formats beyond plain text. Web pages arrive as HTML, documents as PDF or Office formats, scientific papers as LaTeX or PDF, books as EPUB or physical scans. Converting these formats into clean, consistent text suitable for model training requires specialized extraction techniques for each format type.

### Format-Specific Extraction

#### HTML

Raw HTML contains not just content but extensive markup, navigation elements, scripts, styling, ads and boilerplate that must be stripped away. Simple tag-stripping works poorly - it retains navigation text, footer content and ads while sometimes damaging legitimate content.

Sophisticated extractors like **trafilatura** use heuristics to identify main content regions while filtering boilerplate. They recognize patterns indicating navigation menus, sidebars, headers, footers and ads, handle special elements like lists, tables and code blocks, and manage encoding issues to produce clean text focused on the primary content of the page. Different libraries make different trade-offs between **precision** (extracting only high-confidence content) and **recall** (extracting liberally but including more boilerplate).

#### PDF

PDFs encode **visual layout** rather than semantic structure - the positioning of text and images on pages. This makes reading order recovery non-trivial, especially for multi-column layouts, text boxes, sidebars, and tables embedded in prose. Libraries like **pdfplumber**, **pypdf**, and **pdfminer** provide programmatic extraction with varying capabilities, offering bounding box information to help separate headers, footers and main content.

> **Scanned or image-based PDFs** require OCR before text can be extracted. Services like **Amazon Textract** apply ML models to recognize text, though accuracy varies with image quality, font clarity and layout complexity. OCR errors introduce noise that must be managed downstream.

#### Office Documents

Formats like DOCX, PPTX and XLSX embed text in complex structures. Parsing libraries (python-docx, python-pptx, openpyxl) extract not just text but also metadata, formatting and structure. Key decisions include whether to preserve heading hierarchy or flatten to plain text, and whether to include comments, revision histories and metadata.

#### Scientific Papers

Papers present the hardest extraction challenge, combining PDF difficulties with specialized content like mathematical notation, chemical formulas, and figures and tables that need decisions about representation. Papers in **LaTeX source format** avoid PDF issues but still require markup parsing to extract clean text.

#### Books (EPUB)

EPUB is structurally closer to HTML, offering cleaner extraction than PDFs. Still, books require careful handling of chapter structure, front/back matter, footnotes, and specialized formatting for poetry, dialogue or technical content.


### Unicode and Encoding Normalization

Many data quality problems stem from mishandled text encoding. Text originally in **Latin-1** or **Windows-1252** that gets misinterpreted as UTF-8 produces garbled characters - for example, `"café"` appearing as `"cafÃ©"`. Smart quotes and special punctuation may also appear as strange character sequences.

**ftfy** (fix text for you) recognizes these patterns of *mojibake* and attempts to reverse the incorrect transformations. Beyond that, **NFC normalization** converts text to a canonical form ensuring that textually identical strings have identical byte representations, while control character removal strips zero-width spaces, direction marks and other invisible characters that cause downstream issues.

Special character handling requires decisions based on the target use case - whether to preserve currency symbols, math operators, emoji or decorative Unicode characters.


### Language Identification and Separation

For monolingual datasets or language-specific processing, identifying text language is a crucial step. Libraries like **langdetect** and **fastText** support hundreds of languages and output a predicted language alongside a confidence score. Detection can be applied at document, paragraph or sentence level depending on the expected granularity of multilingual content.

**Common challenges:**
- Very short texts provide insufficient signal for accurate classification
- Proper nouns and technical terms may trigger false language identifications
- Code-switched text - speakers alternating languages within a document - is particularly difficult to handle

Setting appropriate confidence thresholds and validating results helps manage false positives. After detection, texts are routed to language-specific pipelines or filtered based on target language requirements.



## Data Cleaning and Filtering

Once text has been extracted and normalized, filtering removes low-quality content. The pipeline typically combines two complementary approaches: **heuristic filters** (fast, transparent, rule-based) and **model-based filters** (more nuanced but computationally expensive).

### Heuristic Filtering

Heuristic filters trade sophisticated understanding for computational efficiency - they can process billions of documents quickly using simple metrics, and their behavior is easy to understand and debug.

**Common heuristics include:**
- **Length filters** - remove documents below ~25 words or above ~100,000 words
- **Character-level statistics** - low alphabetic ratios indicate data tables or non-linguistic material; excessive punctuation or digit density suggests spam or malformed content
- **Word-level statistics** - average word length outliers may indicate encoding issues; very low type-token ratios suggest repetitive or auto-generated content
- **Symbol density** - excessive bullet points, arrows, hashtags or capitalization signal spam or poorly formatted data
- **Stopword ratios** - common function words form a consistent fraction of natural language; deviations suggest keyword-stuffed spam or synthetic text
- **Boilerplate detection** - cookie notices, navigation menus and footer text appear across millions of pages and waste training capacity if not removed

**Cascading filters** apply multiple heuristics in sequence - early stages use conservative filters with minimal false positives, while later stages apply more aggressive ones to surviving content. This staged approach balances effectiveness against the risk of removing valuable content.


### Model-Based Quality Filtering

Beyond heuristic rules, ML models assess text quality using learned patterns from annotated examples. This provides more nuanced assessment at the cost of computational expense and potential bias.

**Classifier-based filtering** trains models to distinguish high-quality from low-quality text. A notable example is the **FineWeb-Edu classifier**, trained to identify educational web content, which produced the FineWeb-Edu dataset with measurably improved model performance. These classifiers require human-annotated training data rating documents on clarity, informativeness, accuracy and writing quality.

Classifier architectures range from lightweight fastText n-gram models (very fast, suitable for large-scale initial filtering) to BERT-style transformers (slower but capturing subtler patterns) to full large language models (used sparingly for final quality assessment on small subsets).

**Perplexity-based filtering** uses a language model to assess text naturalness - lower perplexity indicates text matching expected linguistic patterns. High perplexity often signals encoding problems or nonsensical content. However, it can bias against uncommon topics or unusual writing styles and must be applied carefully.

> **Key risk:** If quality classifiers are trained primarily on formal, professional writing, they may systematically exclude informal or creative content. Careful attention to training data composition and evaluation across diverse examples is essential to avoid bias amplification.


### PII Detection and Redaction

PII includes obvious identifiers like names, addresses, phone numbers and emails, as well as less obvious ones like IP addresses, device identifiers and quasi-identifiers that could identify someone in combination.

**Detection relies on two complementary methods:**
- **Regular expressions** - handle structured PII with predictable formats (phone numbers, emails, SSNs, credit card numbers)
- **Named Entity Recognition (NER)** - transformer-based models identify person names, locations, organizations and other entity types in unstructured text

Redaction can take three forms: complete removal, token replacement with placeholders like `<PERSON>` or `<EMAIL>`, or synthetic replacement with plausible fictional alternatives that preserve sentence structure and format.

Contextual nuance matters - "Washington" may refer to a historical figure, a state or a city, not necessarily a private individual. Quasi-identifiers also present challenges: age + occupation + city might not be PII individually but could identify someone in combination, warranting more conservative filtering.


### Toxicity and Content Safety Filtering

Models trained on unfiltered internet data absorb patterns of hate speech, harassment and harmful content. Removing this improves both safety and model quality.

**Modern toxicity classifiers** like **Detoxify** produce probability scores across multiple categories - general toxicity, severe toxicity, obscenity, threats, insults and identity-based hate - allowing configurable thresholds per category.

The core challenge is **context sensitivity**. The same word can be emphasis in one sentence and hostility in another. Academic discussions *about* hate speech differ from hate speech itself. Violence in journalism differs from gratuitous content. Cultural variation further complicates detection, as offensive standards vary across languages and communities.

**False positive trade-offs:**
- *Conservative* filtering removes more harmful content but risks excluding legitimate speech, especially from marginalized communities discussing difficult topics
- *Permissive* filtering preserves more valid speech but allows more potentially harmful content through

Transparency about filtering decisions - publishing classifier details, thresholds and categories - provides accountability while helping users assess model fit for their needs.


## Deduplication Strategies


## Data Mixing and Composition


## Advanced Preprocessing Techniques


## Tools, Frameworks and Infrastructure


## Best Practices and Emerging Trends

