# Naive Bayes

<div style="text-align: center; margin: 20px 0;">
  <p style="font-size: 1.4em; margin-bottom: 8px;">
    <i>"All models are wrong, but some are useful."</i>
  </p>
  <p style="font-size: 0.9em; color: #777;">
    George Box
  </p>
</div>


## The Central Question

Every chapter in this repository has been, at some level, a story about assumptions, about what structure we impose on the world before we have seen any data and how those priors shape what we can learn. Logistic regression assumes the log-odds are linear. Neural networks assume that useful representations can be built by composing affine transformations and nonlinearities. Each assumption is a bet and the quality of the model is determined by how well that bet pays off.

Naive Bayes makes what is, on its face, a breathtakingly implausible bet: that every feature in our dataset is statistically independent of every other feature, given the class label. In a spam filter, this means assuming that the presence of the word "prince" in an email tells you nothing about whether the word "bank" will also appear. In a medical diagnosis system, it means assuming that a patient's blood pressure reveals nothing about their cholesterol. Anyone who has spent five minutes with real data knows these assumptions are almost never true. And yet, Naive Bayes works - often embarrassingly well. It beats more sophisticated classifiers on text classification benchmarks. It has defended inboxes from spam for decades. A 1997 study by Domingos and Pazzani proved, rigorously, that the wrong assumption about independence can still yield the optimal decision, even when the probability estimates are entirely wrong.

This paradox (the gap between being a bad estimator and a good classifier) is the central mistery of Naive Bayes and resolving it will teach us something fundamental about the relationship between probability and prediction. The chapter is also the natural continuation of a story we began in the last chapter: Naive Bayes is the oldest and most transparent member of the generative model family, the counterpart to logistic regression's discriminative approach. Understanding both, and the precise conditions under which each dominates, is not optimal background material. It is the theoretical scaffolding of probabilistic machine learning.


## A Theorem Older Than the Computer

The algorithm begins not with a programmer but with a Presbyterian minister. Thomas Bayes, an 18th-century English clergyman and amateur mathematician, died in 1761 withour publishing the result that would bear his name. His friend Richard Price found the manuscript among his papers and submitted it to the Royal Society in 1763, framing it as a solution to the problem of **inverse probability**: given that we have observed certain outcomes, what can we infer about the processs that generated them? The question sounds abstract but is the exact question we ask every time we build a classifier - given that an email contains certain words, what can we infer about whether a human marked it as spam?

**Bayes' theorem** provides the formal machinery for updating a belief in light of evidence. Its statement is compact.  Let yy
y be a class label and x=(x1,x2,…,xn)\mathbf{x} = (x_1, x_2, \ldots, x_n)
x=(x1​,x2​,…,xn​) be a vector of observed features. The theorem says:

P(y∣x)=P(x∣y)⋅P(y)P(x).P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}.P(y∣x)=P(x)P(x∣y)⋅P(y)​.

Four quantities appear:
- The **prior $P(y)$** is our belief about how likely class $y$ is before we observe any features (the baseline rate of spam in all email, for example)
- The **likelihood $P(\mathbf{x} \mid y)$** is the probability of seeing this particular configuration of features given that the class is $y$ (how characteristic these words are of spam)
- The **evidence $P(\mathbf{x})$** is the overall probability of observing exactly these features, averaged across all possible classes, it acts as a normalizing constant.
- The **posterior $P(y \mid \mathbf{x})$** (the left-hand side) is what we actually want: the probability that the class is $y$ given the features we observed.

Bayes' theorem is fundamentally a formula for going from likelihood to posterior, from "how likely am I to see these symptoms given this disease?" to "how likely is this disease given these symptoms?" - an inversion that is immensely useful and surprisingly non-trivial.

The theorem itself requires no simplifying assumptions. The trouble is practical: to evaluate $P(\mathbf{x} \mid y)$ exactly, we would need to know the full joint provavility distribution over all $n$ features conditioned on each class. For any moderately sized feature space, this distribution has astronomically many parameters - an exponential explosion that makes direct estimation impossible. A spam filter might operate over a vocabulary of $50,000$ words. Estimating the joint distribution over even a hundred of them exactly would require more data than exists. Something has to give.


## The Independence Assumption

The insight that makes Naive Bayes tractable is a single, forceful simplification: assume that all features are **conditionally independent** given the class label. In formal terms, this means

$$P(x_1, x_2, \ldots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

This equation deserves to be read carefully. It says the probability of observing this entire feature vector, given class $y$, factors into the product of $n$ individual probabilities - one for each feature, each estimated independently. The joint distribution over $n$ features, which previously required an exponential number of parameters, now requires only $n \cdot K$ parameters, where $K$ is the number of classes. For $K=2$ and $n=50,000$, this collapses from an astronomically large table to a manageable set of $100,000$ numbers. That is the entire algebraic payoff of the naive assumption.

Substituting into Bayes' theorem, the posterior becomes

$$P(y \mid x_1, \ldots, x_n) = \frac{P(y) \cdot \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, \ldots, x_n)}$$

To make a classification decision, we compute this posterior for every class and pick the one with the highest value - the **Maximum A Posteriori (MAP)** estimate:

$$\hat{y} = \arg\max_y \, P(y) \cdot \prod_{i=1}^{n} P(x_i \mid y)$$

Notice that the denominator $P(x)$ has vanished. Since it is the same for every class, it cannot change which class has the highest posteriori. We are not computing the probability of the features; we are comparing the *relative palusibility* of different classes given those features and the common normalizing factor is irrelevant for that comparison. This is a subtle but important point: the model is never asked to assign a valid overall probability to $\hat{y}$; it is asked only to rank the classes. Dropping the denominator makes computation faster and introduces no error in the decision.

<figure_placeholder: naive_bayes_generative_structure.png - A graphical model diagram showing the generative structure of Naive Bayes. The class variable y is shown at the top as a large node, with directed edges pointing down to n feature nodes x₁, x₂, ..., xₙ arranged in a row. Each xᵢ is connected only to y, with no edges between feature nodes - visually representing conditional independence. A caption reads: "All features are generated independently from the class label. The absence of edges between features is the naive assumption made concrete." An inset shows how a Bayesian network with dependencies between features would look (a denser graph), contrasting with the sparse naive structure.>

## Learning the Parameters

Having established the form of the classifier, we need to estimate its two kinds of parameters from training data. Both are estimated by the principle of **Maximum Likelihood Estimation (MLE)** - choosing the parameters that make the observed training data as probable as possible.

*The prior* $P(y=c)$ for each class $c$ is straightforward: it is simply the proportion of training examples that belong to class $c$. If 60 or 100 emails are spam, then $\hat{P}(spam)=0.6$. No assumptions about feature distributions are involvedM this estimate is always available regardless of the variant of Naive Bayes being used.

*The class-conditional likelihoods* $P(x_i \mid y)$ depend on what kind of features we have. This is where the algorithm diverges into distinct variants, each choosing a distributional assumption appropriate to the feature type:
- **Gaussian Naive Bayes** assumes that continuous features follow a normal distribution within each class. The parameters $\mu_{y,i}$ (mean) and $\sigma^2_{y,i}$ (variance) are estimated separately for each feature $i$ in each class $y$, and the likelihood for a new value $x_i$ is evaluated using the Gaussian density $P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_{y,i}}} \exp\!\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma^2_{y,i}}\right)$. This variant is natural for sensor readings, biological measurements, and other continuous real-valued data.
- **Multinomial Naive Bayes** assumes that features represent counts - the number of times word $i$ appears in a document, for instance. The class-conditional likelihood for feature $i$ given class $y$ is $P(x_i \mid y) = \theta_{y,i}^{x_i}$, where $\theta_{y,i}$ is the probability of feature $i$ appearing in any one position within class $y$, estimated from the training corpus. This is the standard model for text classification.
- **Bernoulli Naive Bayes** treats each feature as binary - present or absent - rather than as a count. It assigns a probability $P(x_i = 1 \mid y) = p_{y,i}$ for each feature, and crucially, it *explicitly penalizes* the *absence* of a feature that is strongly indicative of a class. If "urgent" is a strong spam indicator, a Bernoulli model penalizes its absence in a candidate spam email; a Multinomial model simply ignores it. 

These three variants are not interchangeable. Multinomial and Bernoulli models will produce different predictions on the same text data, because Bernoulli treats a word mentioned ten times identically to a word mentioned once. The right choice is determined by the data-generating process you believe underlies your features.

<figure_placeholder: three_naive_bayes_variants.png - Three side-by-side panels showing the distribution of a single feature for two classes (class 0 in blue, class 1 in red). Left panel (Gaussian): two overlapping bell curves with different means and variances. Middle panel (Multinomial): two bar charts showing word count distributions with different frequency profiles. Right panel (Bernoulli): two pairs of bars at x=0 and x=1 with different probabilities of presence. Each panel is labeled with its variant name and typical use case (continuous data, word counts, binary features).>


## The Zero-Frequency Problem and the Bayesian Remedy

Consider an email containing the phrase "cryptocurrency arbitrage". Suppose no email in the training set contains the word "arbitrage" in the spam class. Under the Multinomial model, the MLE for $P("arbitrage" \mid spam)$ is exactly zero. And because the classifier multiplies all feature likelihoods together, a single zero entry annihilates the entire posterior for the spam class: the model assigns spam probability zero to any email containing a word it never saw in spam training examples, regardless of all other evidence. This pathology is called the **zero-frequency problem**.

The standard remedy is **Laplace smoothing** - add a pseudocount of $\alpha > 0$ to every feature count before computing the likelihood estimate. For Multinomial Naive Bayes, the smoothed estimate for feature $i$ in class $y$ is:

$$\hat{P}(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha \cdot V},$$

where: 
- $N_{yi}$ is the count of feature $i$ in training documents of class $y$
- $N_y$ is the total feature count for class $y$ 
- $V$ is the vocabulary size.

The setting $\alpha = 1$ is called *Laplace smoothing*; any $0<\alpha<1$ is called *Lidstone smoothing*. Both prevent zero probabilities and provide a graceful fallback for unseen features.

What looks like an ad hoc engineering fix turns out, on closer inspection, to be a principled Bayesian statement. Laplace smoothing is exactly equivalent to placing a **symmetric Dirichet prior** with concentration parameter $\alpha$ on the parameter vector **$\theta_y$**. A Dirichlet prior is natural conjugate prior for a multinomial distribution - meaning that when you combine a Dirichlet prior with multinomial-distribution data, the posterior is itself a Dirichlet, and the posterior mean has precisely the smoothed form above. Laplace smoothing is not a hack; it is Bayesian inference with a uniform prior over feature probabilities, recovering the MLE as $\alpha \rightarrow 0$ and pulling estimates toward uniformity as $\alpha$ increases. This connection matters because it suggests a principled way to set $\alpha$: treat it as a hyperparameter, select it via cross-validation and understand that we are simultaneously choosing the strength of our prior belief that all features are roughly equally likely.


## Working in Log-Space

There is a numerical subtlety worth making explicit. The MAP classifier computes a product of potentially thousands of small probabilities - one for each word in a document. In floating-point arithmetic, such products underflow to zero long before the computation is complete, making the model numerically useless. The remedy is the same one we used in logistic regression: take logarithms, turning the product into a sum.
 
Since the logarithm is a monotonically increasing function, the $\arg\max$ is unchanged, and the classification rule becomes
 
$$\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y) \right].$$
 
This is numerically stable for any feature vector of any length, requires only additions rather than multiplications, and is computationally efficient. There is also a conceptual elegance to this form: the class is chosen by accumulating a *sum of independent log-evidence contributions*, one from each feature. Each feature votes for each class with a weight equal to its log-likelihood ratio, and the prior acts as an initial "starting score" before any evidence is observed. The model is, in this view, a linear combination of log-likelihoods - a structure that will become important when we examine the relationship to logistic regression below.

## The Paradox: A Bad Estimator, A Good Classifier
 
Here is the most interesting theoretical result in the entire Naive Bayes story. Domingos and Pazzani showed in 1997 that although the Bayesian classifier's probability estimates are only optimal under quadratic loss if the independence assumption holds, the classifier itself can be optimal under zero-one loss - misclassification rate - even when this assumption is violated by a wide margin.
 
The key is the distinction between *probability estimation* and *classification*. What the model must get right for classification is not the *magnitude* of the posterior probabilities - how close $\hat{P}(\text{spam} \mid \mathbf{x})$ is to the true $P(\text{spam} \mid \mathbf{x})$ - but only their *ordering*. Specifically, the model must assign the correct class the highest score, which requires only that the *ratio* of posteriors preserves the correct direction. The probability estimates can be wildly off in absolute terms, catastrophically overconfident or underconfident, and the classification decision can still be perfect.
 
Two attributes may depend on each other, but the dependence may distribute evenly in each class. In this case, the conditional independence assumption is violated, but Naive Bayes is still the optimal classifier. More generally, what matters is not the *existence* of feature dependencies but their *asymmetry*: if the dependencies between features are roughly symmetric across classes - if they tend to cancel out when you take the log-odds difference - their effect on the decision boundary is neutral. The naive model's score is wrong in magnitude but right in direction.
 
This explains a pattern that puzzled researchers for years: Naive Bayes maintains high classification accuracy on datasets with clear feature dependencies, while its probability outputs are badly calibrated. Although Naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from `predict_proba` are not to be taken too seriously. The practical implication is sharp: use Naive Bayes to make decisions, not to report probabilities to stakeholders. A medical system that uses Naive Bayes to triage patients should be understood as a ranking tool, not as a device producing reliable probability estimates of disease. When calibrated probabilities matter - and in high-stakes settings they very often do - post-processing with Platt scaling or isotonic regression is required, or a better-calibrated model (such as logistic regression) should be used directly.
 

 
## Complement Naive Bayes: Fixing Asymmetry in Text
 
Multinomial Naive Bayes has a systematic weakness when classes are imbalanced or when training documents differ substantially in length. Because it estimates each class's model from only the documents *in that class*, classes with fewer training examples are poorly estimated, and those with more examples can dominate. Naive Bayes is often used as a baseline in text classification because it is fast and easy to implement. Its severe assumptions make such efficiency possible but also adversely affect the quality of its results.
 
**Complement Naive Bayes (CNB)**, introduced by Rennie et al. in 2003, addresses this by flipping the estimation logic. Rather than estimating $P(x_i \mid y)$ - the likelihood of feature $i$ given class $y$ - CNB estimates the likelihood of feature $i$ given *all other classes combined*, the complement of $y$. The classification rule assigns the document to the class for which it is the *worst fit in all other classes*: a document is spam if it least resembles non-spam. This complement approach produces more stable parameter estimates because each class's complement pool is larger and more representative than its positive pool. The Complement Naive Bayes classifier was designed to correct the "severe assumptions" made by the standard Multinomial Naive Bayes classifier. It is particularly suited for imbalanced data sets. In practice, CNB regularly outperforms standard Multinomial Naive Bayes on text classification tasks - a clean example of how understanding a model's failure mode can suggest a simple, principled fix.
 

 
## The Generative–Discriminative Duality, Revisited
 
In the previous chapter, we introduced the conceptual distinction between discriminative models like logistic regression, which model $P(y \mid \mathbf{x})$ directly, and generative models like Naive Bayes, which model $P(\mathbf{x}, y) = P(\mathbf{x} \mid y) P(y)$ and derive the posterior through Bayes' theorem. Here we can make this duality mathematically precise, because the two models are more deeply connected than they appear.
 
Consider binary Naive Bayes with Gaussian class-conditional distributions of equal variance ($\sigma^2_{0,i} = \sigma^2_{1,i} = \sigma^2_i$ for all features $i$). Computing the log-odds of the Naive Bayes posterior:
 
$$\log \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \log \frac{P(y=1)}{P(y=0)} + \sum_{i=1}^{n} \log \frac{P(x_i \mid y=1)}{P(x_i \mid y=0)}.$$
 
Substituting the Gaussian densities and simplifying, the Gaussian terms produce a linear function of $\mathbf{x}$ - specifically, a weighted sum of the features with weights determined by the class means and shared variances. The posterior is therefore the sigmoid of a linear combination:
 
$$P(y=1 \mid \mathbf{x}) = \sigma\!\left(\mathbf{w}^T \mathbf{x} + b\right),$$
 
which is exactly logistic regression. Under equal-variance Gaussian class-conditionals, **Naive Bayes and logistic regression make identical functional predictions** - they differ only in how the weights $\mathbf{w}$ are estimated. Logistic regression estimates them by maximizing $P(y \mid \mathbf{x})$ directly on the labeled training data. Naive Bayes estimates the class means and variances separately and derives $\mathbf{w}$ from those estimates, using the Gaussian modeling assumption as a structural constraint. This equivalence is not limited to Gaussians: it holds whenever the class-conditional distributions belong to the exponential family with shared dispersion - a broad class that includes Gaussians, Poisson, Bernoulli, and Multinomial distributions.
 
What the equivalence reveals is that Naive Bayes is *not* fundamentally a different model from logistic regression. It is the same decision function, reached by a different path. The path taken by Naive Bayes - going through an explicit generative model - requires extra assumptions (the specific form of $P(x_i \mid y)$), but it requires *less labeled data* to fit. When data is scarce, these structural assumptions act as strong prior knowledge that can prevent overfitting. When data is abundant, logistic regression's direct optimization of $P(y \mid \mathbf{x})$ produces a more accurate decision boundary precisely because it does not waste model capacity on accurately representing the feature space. This is the bias-variance trade-off expressed in the language of model families rather than hyperparameters.
 

 
## Looking Forward
 
Naive Bayes occupies a peculiar but valuable position in the landscape of machine learning: it is simultaneously a limiting approximation and a sophisticated probabilistic framework. Its structure - a product of independent class-conditional terms, combined with a prior - is the atom from which more expressive Bayesian classifiers are built. **Tree Augmented Naive Bayes (TAN)**, introduced by Friedman, Geiger, and Goldszmidt in 1997, relaxes the independence assumption by allowing each feature to depend on at most one other feature (forming a tree-structured Bayesian network), while retaining much of Naive Bayes' computational tractability. More generally, Bayesian networks replace the naive independence structure with a directed acyclic graph encoding the actual dependencies among features, recovering Naive Bayes as the special case where the graph has no edges between features.
 
The same generative spirit lives on in modern deep learning, in a different guise. The Variational Autoencoder - which we study in Chapter 7 - is a generative model that explicitly learns $P(\mathbf{x} \mid \mathbf{z})$ and $P(\mathbf{z})$, where $\mathbf{z}$ is a latent variable playing the role of a continuous, learned class structure. Generative Adversarial Networks learn to model $P(\mathbf{x})$ through a game-theoretic training procedure. In both cases, the ambition is the same as Naive Bayes': to understand not just which class a data point belongs to, but the process by which data of that class is generated. Naive Bayes is naive about features but not about goals. Its goal - learning the joint distribution, reasoning backward from observations to causes - is the most ambitious goal a learning algorithm can have, and the sophisticated models that follow it are best understood as attempts to achieve that same goal with fewer simplifying assumptions.
 

 
## Key Takeaways
 
Naive Bayes derives from Bayes' theorem by assuming **conditional independence** of all features given the class label - an assumption that reduces the number of parameters from exponential in the number of features to linear, making the model tractable even in very high-dimensional spaces like text; this independence assumption is almost always violated in practice, yet the model performs surprisingly well because classification requires only that the correct class receives the highest score, not that the posterior probabilities are accurate in magnitude.
 
The MAP decision rule $\hat{y} = \arg\max_y [P(y) \prod_{i} P(x_i \mid y)]$ is computed in log-space as $\hat{y} = \arg\max_y [\log P(y) + \sum_i \log P(x_i \mid y)]$ to avoid floating-point underflow; the evidence denominator $P(\mathbf{x})$ is dropped because it is constant across all classes and cannot affect the ranking.
 
**Laplace smoothing** - adding a pseudocount $\alpha$ to every feature count before estimation - prevents the zero-frequency pathology where a single unseen feature zeroes out the entire posterior; this is not a heuristic but exact Bayesian inference with a symmetric Dirichlet prior, and the smoothing parameter $\alpha$ controls the strength of that prior.
 
Domingos and Pazzani (1997) proved that Naive Bayes can be the optimal zero-one-loss classifier even when the independence assumption is badly violated, because what matters for classification is the *asymmetry* of feature dependencies across classes rather than their presence; as a consequence, Naive Bayes is a reliable classifier but a poor probability estimator, and its raw `predict_proba` outputs should not be interpreted as calibrated beliefs without post-processing.
 
When the class-conditional distributions are Gaussian with equal variances, the Naive Bayes posterior reduces exactly to the logistic regression sigmoid - both models produce the same linear decision boundary, but Naive Bayes estimates its weights through the generative path of class-conditional statistics while logistic regression estimates them by directly maximizing the conditional likelihood; this equivalence reveals the two models as instances of a single deeper framework, and their practical trade-off - Naive Bayes when data is scarce, logistic regression when data is abundant - follows directly from the bias-variance decomposition applied at the level of model families.
 

 
## Further Reading
 
- Bayes, T. (1763). An essay towards solving a problem in the doctrine of chances. *Philosophical Transactions of the Royal Society of London*, 53, 370–418. (Price's posthumous presentation of Bayes' theorem; a remarkable historical document to read alongside modern applications. Remarkably accessible given its age.)
- Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29(2–3), 103–130. (The paper that rigorously established when Naive Bayes can be optimal despite violated independence; the central theoretical result in this chapter. Essential reading for anyone who wants to understand *why* the model works.)
- Zhang, H. (2004). The optimality of naive Bayes. *Proceedings of the Seventeenth International Florida Artificial Intelligence Research Society Conference*, 562–567. (Extends Domingos & Pazzani by characterizing the role of *dependency distribution* - showing that dependencies cancel out when they are symmetric across classes, providing a cleaner sufficient condition for optimality.)
- Lewis, D. D. (1998). Naive (Bayes) at forty: The independence assumption in information retrieval. In *ECML-98: Machine Learning*, Lecture Notes in Computer Science 1398. Springer. (A thoughtful retrospective on the model's empirical successes in text retrieval; useful for understanding the gap between theory and practice in natural language tasks.)
- McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. *AAAI/ICML-98 Workshop on Learning for Text Categorization*, 41–48. (The canonical comparison of the Multinomial and Bernoulli variants for text classification; explains precisely when each model is appropriate and provides empirical evidence on real datasets.)
- Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of Naive Bayes text classifiers. *Proceedings of the Twentieth International Conference on Machine Learning (ICML-03)*, 616–623. (Introduces Complement Naive Bayes and proposes practical transforms - TF-IDF-like normalization, log-frequency scaling - that bring Naive Bayes to competitive performance with SVMs on text tasks.)
- Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. *Advances in Neural Information Processing Systems*, 14. (Proves the asymptotic equivalence of logistic regression and Gaussian Naive Bayes under the latter's distributional assumptions; establishes the famous empirical result that generative models win with small data and discriminative models win with large data.)
- Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian network classifiers. *Machine Learning*, 29(2–3), 131–163. (Introduces Tree Augmented Naive Bayes (TAN) as a principled relaxation of the independence assumption; the natural next step after understanding Naive Bayes, and a gateway to learning in Bayesian networks more generally.)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4 (sections 4.1–4.2). Springer. (Provides the rigorous generative/discriminative framework within which Naive Bayes and logistic regression are both instances; the section on Gaussian generative models makes the shared decision-boundary structure explicit.)
- Hand, D. J., & Yu, K. (2001). Idiot's Bayes - not so stupid after all? *International Statistical Review*, 69(3), 385–398. (A survey of empirical evidence for why Naive Bayes performs well; discusses calibration, conditions of optimality, and connections to other classifiers. Recommended for readers who want a data-centric perspective on the theoretical results.)