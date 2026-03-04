# Decision Trees

#### Table of Contents

1. [Introduction](#introduction)
2. [How Decision Trees Work](#how-decision-trees-work)
3. [Types of Decision Trees](#types-of-decision-trees)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Splitting Criteria](#splitting-criteria)
6. [Handling Numerical Features](#handling-numerical-features)
[Tree Construction Algorithms]
[The Feature Selection Problem]
[Pruning Strategies]
[Rule Extraction]
[Advantages and Disadvantages]
[Real-World Applications]
[Implementation Guide]
[Summary]

---
## Introduction

Decision Trees are supervised machine learning algorithms that can be used for both classification and regression tasks. They represent one of the most interpretable and intuitive approaches in machine learning, mimicking human decision-making processes through a series of "if-then" rules.

The foundations of decision tree learning can be traced back to the 1960s with Hunt's Algorithm, which was originally developed to model human learning in psychology. This pioneering work laid the groundwork for many modern decision tree algorithms.

Unlike black-box models such as neural networks, decision trees are white-box models that provide complete transparency in their decision-making process. Every decision can be traced back through a clear path of logical rules, making them particularly valuable in domains where explainability is crucial (such as healthcare, finance and legal systems).

The fundamental approach is divide-and-conquer: the algorithm recursively splits the data into smaller, more homogeneous subsets until each subset is sufficiently pure or until a stopping criterion is met.

#### The Decision-Making Analogy

The best way to understand how decision trees work is through the classic game "Guess Who?". In this game, we try to identify your opponent's secret character by asking yes/no questions.

When playing this game, we don't ask questions like "Does the person have a mole on their left cheek?" as this question is too specific and eliminates very few candidates. We ask questions like "Is your character male or female?" or "Do they wear glasses?" as this are broader questions that divide the pool roughly in half. 

The decision tree algorithm follows the exact same principle: it searches for the question (feature) that best separates the data into distinct groups. The algorithm quantifies "best" using mathematical measures of impurity or information gain.





---
## How Decision Trees Work

To properly analyze how decision trees work, we must understand its anatomy.

#### Anatomy of a Decision Tree

A decision tree is a directed acyclic graph (DAG) composed of three types of nodes:

**1. Root Node**: This is the starting point of the tree containing 100% of the training data. The root node performs the most important split in the entire tree. The feature chosen here has the highest discriminative power across the entire dataset. Mathematically, this node contains the highest impurity before any splits occur. 

**2. Internal Nodes (Decision Nodes)**: These are the intermediate nodes that pose questions about features. Each internal node tests a specific attribute and splits the data into child nodes based on the outcome. 

In binary trees, each internal node has exactly two children. In multi-way trees a node can have multiple children corresponding to different attribute values.

**3. Leaf Nodes**: Nodes with no children that provide final prediction:
- For Classification: The leaf contains the majority class label from the training samples that reached this node
- For Regression: The leaf contains the average (mean) of the target values of samples that reached this node.

Ideally, leaf nodes should be pure (containing samples from only one class), though this is rarely achieved in practice and can lead to overfitting.

#### Decision Trees Functionality

Decision trees employ a greedy, top-down, recursive partitioning strategy to build the model. The construction process has 5 steps:

**1. Start with the Entire Dataset**: Begin at the root node containin all training example and calculate the impurity of this node.

**2. Evaluate all Possible Splits**: For each feature:
- If categorical: Consider splits based on each value
- If numerical: Consider threshold splits ($feature \leq t$ vs $feature > t$)
Calculate the impurity reduction (information gain) for each possible split

**3. Choose the Best Split**: Select the feature and split point that produces the largest information gain (or largest impurity reduction). This create child nodes based on this split.

**4. Recursion**: Repeat steps 2-3 for each child node. Continue until stopping criteria are met:
- All samples in a node belong to the same class (pure node)
- Maximum depth reached
- Minimum samples per node threshold
- No further information gain possible

**5. Assign Predictions to Leaves**:
- For classification: Assign the majority class
- For regression: Assign the mean value


Decision trees use a **greedy algorithm**, meaning they make locally optimal choices at each step without considering future consequences. This has important implications:
- Advantages:
    - Computationally efficient (doesn't need to evaluate all possible trees)
    - Fast training even on large datasets
- Disadvantages:
    - May not find the globally optimal tree
    - Susceptible to the "horizon effect" (making a suboptimal split now might enable a better split later)

---
## Types of Decision Trees

Decision trees can be categorized based on the type of target variable they predict.

#### Classification Trees

The purpose is to predict categorical outcomes (discrete class labels). And outputs a class label from a finite set. Examples:
- Email classification: Spam or Not Spam
- Medical diagnosis: Disease or Healthy
- Customer churn: Will leave or Will Stay
- Image recognition: Cat, Dog or Bird

The prediction on the leaf node is done by calculating the majority class among samples reaching that leaf. It uses probability distributions and impurity measures (Gini, Entropy).

#### Regression Trees

The purpose is to predict continuous numerical outcomes. The output is a real number. Examples:
- House price prediction
- Temperature forecasting
- Stock price estimation
- Age prediction from images

The prediction on the leaf node is done by calculating the mean (average) of target values among samples reaching that leaf. It uses variance reduction as the splitting criterion.

**Splitting criterion for regression**: Instead of Gini or Entropy, regression trees minimize:
- Mean Squared Error (MSE): The average squared difference between predicted and actual values
- Mean Absolute Error (MAE): The average absolute difference


#### Types of Classification Problems

It's important to distinguish between different classification scenarios:

**Binary Classification**: Two mutually exclusive classes

$$
y \in \{0,1\} \quad \text{or} \quad y \in \{-1,+1\}
$$

**Multiclass Classification**: Multiple classes, but only one correct answer per instance. We can compare it to a multiple-choice test question. 

$$
y \in \{A,B,C...\}
$$

**Multi-label Classification**: Multiple labels can apply to a single instance simultaneously. We can compare it to checkboxes on a form (select all that apply). For this, every label must have independent probabilities.

$$
y \subseteq \{A,B,C,...\}
$$

(Note: Standard decision trees handle binary and multiclass classification naturally. Multi-label problems typically require special adaptations or ensemble methods.)

---
## Mathematical Foundations

The power of decision trees lies in their ability to quantify the "quality" of a split mathematically. This section explores the key metrics used.

**Impurity measures** quantify how mixed or disordered a set of samples is. The algorithm seeks to minimize impurity in child nodes after each split:
- **Perfect purity**: $\text{Impurity} = 0$ (all samples belong to one class)
- **Maximum impurity**: Impurity is maximized when classes are perfectly balanced

#### Gini Impurity
Gini Impurity is the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the class distribution in the node.

Formula:

$$
Gini(S) = 1 - \sum_{i=1}^{C} p_i^2
$$

Where:
- $S$ is the set of samples in the node
- $C$ is the number of classes
- $p_i$ is the proportion of samples belonging to class i

Property:
- Range: $[0,0.5]$ for binary classification, $[0,1-1/C]$ for C classes
- $Gini=0$: Perfect purity (all samples in one class)
- $Gini = 0.5$: Maximum impurity for binary (50/50 split)

Example calculation:

Dataset with 100 samples: 60 Class A, 40 Class B

$$
Gini = 1 - [(60/100)^2 + (40/100)^2]
$$

$$
Gini = 1 - [0.36 + 0.16] = 1 - 0.52 = 0.48
$$

Gini brings with it a computational advantage Gini impurity doesn't require logarithms, making it faster to compute than entropy. This is why it's the default in scikit-learn's implementation.

#### Entropy

A measure from information theory quantifying the average amount of information (or uncertainty) in the data.

Formula:

$$
H(S) = - \sum_{i=1}^{C} p_i log_2(p_i)
$$

Where:
- $H(S)$ is the entropy of set $S$
- $p_i$ is the proportion of samples belonging to class i
- By convention: $0 log(0)=0$

Properties:
- Range: $[0,log_2(C)]$ where C is the number of classes
- $H = 0$: Perfect purity (no uncertainty)
- $H = 1$: Maximum uncertainty for binary (50/50)
- $H = log_2(C)$: Maximum uncertainty for $C$ balanced classes

Example Calculation:

Same dataset: 60 Class A, 40 Class B

$$
H=-[(60/100) log_2(60/100) + (40/100) log_2(40/100)]
$$

$$
H=-[0.6 \times (-0.737) + 0.4 \times (-1.322)]
$$

$$
H=-[-0.442-0.529]=0.971 \text{ bits}
$$

On average, we need about 0.97 bits of information to specify the class of a random sample.

#### Information Gain

Te reduction in entropy (or impurity) achieved by splitting on a particular feature. 

$$
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
$$

Where:
- $S$ is the parent node dataset
- $A$ is the attribute (feature) being considered
- $S_v$ is the subset of $S$ where attribute has value $v$
- $|S|$ denotes the number of samples in $S$

Information gain measuers how much knowing the value of attribute A reduces our uncertainty about the class label. The algorithm selects the attribute with the highest Information Gain for splitting.

**Example**: Parent node entropy: $H(Parent)=0.971$.

After split on $Age < 30$:
- Left child (60 samples): $H(Left) = 0.5$
- Right child (40 samples): $H(Right) = 0.3$

$$
IG = 0.971 - [(60/100)(0.5) + (40/100)(0.3)]
$$

$$
IG = 0.971 - [0.3+0.12] = 0.971 - 0.42 = 0.551
$$

This split reduces entropy by 0.551 bits

#### Gini Gain (Gini Reduction)

For Gini impurity, the equivalent concept is Gini Gain:

$$
\Delta Gini = Gini(parent) - \sum \frac{|S_v|}{|S|} Gini (S_v)
$$

The algorithm maximizes Gini Gain (or minimizes the weighted Gini of Children).

---
## Splitting Criteria

The algorithm must decide not just which feature to split on, but also how to split it. The approach differs for categorical and numerical features.

#### Categorical Features

For categorical (discrete) features, the split is straightforward:
- Binary Trees (CART): Create split like $Color = Red$? (Yes/No) and one child gets samples where $Color = Red$ and the other child gets all other colors
- Multi-way trees (ID3, C4.5): Create one branch for each categorical value. If $Color \in {Red, Blue, Green}$, create 3 branches

The advantage is that it is natural and interpretable but multi-way splits can lead to data fragmentation (many small groups).

#### Numerical Features

Numerical (continuous) features require a threshold-based split. The challenge is we can't create a branch for every possible value (would create millions of branched and severe overfitting).

The solution is to convert to binary decision using a threshold t:

$$
Feature \leq t ?   \rightarrow \text{Yes/No}
$$


#### The Missing Value Problem

As we know, real-world data is not perfect and often contains missing values (usually, solved in data pre-processing but in this section we will consider that no pre-processing was done). The solutions for this case are:
1. **Most common value**: Assign the most frequent value in that feature
2. **Surrogate splits**: Find alternative features that correlate with the primary split
3. **Separate branch**: Create a third branch specifically for missing values.
4. **Ignore during split**: Don't consider samples with missing values when calculating split quantity

**CART approach**: Uses surrogate splits - if the primary split feature is missing, it uses the next best feature that tends to produce similar splits

---
## Handling Numerical Features