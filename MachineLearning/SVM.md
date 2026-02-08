# Support Vector Machines (SVM)

#### Table of Contents
1. [Introduction](#introduction)
2. [The Separation Problem](#the-separation-problem)
3. [The Concept of Margin](#the-concept-of-margin)
4. [Mathematical Optimization](#mathematical-foundations)
5. [Support Vectors](#support-vectors)
6. [The Kernel Trick](#the-kernel-trick)
7. [Non-Linear SVM](#non-linear-svm)
8. [Types of Kernel](#types-of-kernels)
9. [The Dual Problem and Lagrange Multipliers](#the-dual-problem-and-lagrange-multipliers)
10. [Support Vector Regression (SVR)]
11. [Hyperparameter Tuning]
12. [SVM vs Other Classifiers]
13. [Real-World Applications]
14. [Advantages and Use Cases]
15. [Summary]

---

## Introduction
Support Vector Machines (SVM) are supervised machine learning algorithms used for classification and regression tasks. They represent one of the most theoretically sound and practically effective approaches in the machine learning toolkit.

Unlike other classification methods that simply seek to minimize classification errors, SVMs optimize for both accuracy and generalization capability. The fundamental philosophy is based on the principle that the best separator is not just any line that correctly classifies the training data, but rather the one that maintains the maximum possible distance from data points of all classes. This approach, rooted in structural risk minimization, leads to better generalization of unseen data.

---
## The Separation Problem
When dealing with binary classification problems we often face a fundamental question: while there may be infinitely many lines that can perfectly separate the training data with 100% accuracy, which one should we choose.

Let's consider a simple scenario where we have colored valls of two types scattered on a table. We can place a stick (representing a line in 2D) to separate them. However, we'll quickly notice that there are countless ways to place it. The question then becomes which position is optimal?

The answer lies in thinking about what happens when we add new balls to the table. If our stick is to close too some balls it can lead to misclassification. On the other hand, if we place it as in a position whehre the distance from balls on both sides is maximized, we can create more tolerance for variation.

This is exactly what Maximum Margin tells us - the best separating line is the one that passes as far as possible from the points of both froups. This intuition is rooted in several principles:
1. Robustness: A separator that barely squeezes between the classes might work perfectly on training data but could fail on new, slightly different data points
2. Generalization: By maximizing the distance to all training points, we're implicitly preparing for the variablity we'll see in test data
3. Statistical Learning Theory: Vapnik's work showed that maximizing the margin provides theoretical guarantees on generalization error bounds.

---
## The Concept of Margin

The margin is the width of the "street" or "corridor" that separates the classes. It represents the buffer zone between the decision boundary and the nearest data points from each class. Mathematically, the margin is the perpendicular distance from the decision boundary to the nearest training examples.

The SVM objetice is not merely to classify correctly (minimize error) but to maximize the margin. This dual objective has several important implications:
1. Safety and Confidence: A larger margin means more confidence in predictions. Points can vary slightly from their training positions and still be correctly classified. This is crucial in real-world applications where measurement noise and data variability are inevitable.
2. Better Generalization: The model is less likely to overfit the training data. By keeping maximum distance from all training points, the model doesn't adapt too closely to the peculiarities of the training set. This is supported by VC theory, which shows that the generalization error is bounded by a function of the margin.
3. Robustness to Noise: The classifier becomes more robust to noise and small perturbations in the data. Outliers that are far from the margin have minimal impact on the decision boundary.
4. Structural Risk Minimization: Maximizing the margin is equivalent to minimizing a specific measure of model complexity, which aligns with the principle of structural risk minimization

---
## Mathematical Foundations
Understanding the mathematics behind SVM is crucial for appreciating its elegance and power. This section builds from basic definitions to the complete optimization problem.

#### HyperPlane Definition
In mathematical terms, the decision boundary (hyperplane) in SVM is defined by the equation:
$$
w \cdot x+b=0
$$

Where:
- w is the weight vector (normal vector to the hyperplane)
- x is the feature vector (data point)
- b is the bias term (intercept or offset)
- $\cdot$ denotes the dot product

In 2D space, this represents a line. In 3D, it's a plane. In higher dimensions, it's called a hyperplane. The vector w determines the orientation of the hyperplane, while b determines its position relative to the origin.

The perpendicular distance from a data point $x_i$ to the decision boundary is given by:
$$
d_i = |w \cdot x_i + b| / \left\| w \right\|
$$

Where:
- $\left\| w \right\|$ represents the Euclidean norm of te weight vector
- The absolute value ensures we get a positive distance

This formula is fundamental because the margin is defined in terms of these distances. Specifically, the margin equals the minimum distance from any training point to the hyperplane, multiplied by 2 (accounting for both sides)

#### The Margin Formula

The margin is given by:
$$
Margin = 2 / \left\| w \right\|
$$

This means that to maximize the margin, we need to minimize $\left\| w \right\|$. This is the core insight that transforms the intuitive goal into a concrete mathematical optimization problem.

Once we have the hyperplane, classifying a new point x is straightforward:
$$
\hat y = w \cdot x + b
$$

- If $w \cdot x + b \geq 0$, predict class +1
- If $w \cdot x + b < 0$, predict class -1

##### Hard Margin Classification
Hard Margin SVM is used when the data is perfeclty linearly separable - meaning there exists a hyperplane that can separate the classes withou any errors. 

This means all points must be:
1. On the correct side of the decision boundary
2. At least at unit distance from the decision boundary (when measured in the normalized space)

The points that satisfy $y_i(w \cdot x_i + b) = 1$ exactly ar ethe support vectors.

This is called "hard margin" because it enforces strict separation: no points are allowed within the margin or on the wrong side. This works perfectly for linearly separable data but fails when:
- Data contains outliers
- Classes naturally overlap
- Data is not linearly separable

In practice, perfectly linearly separable data is rare, which is why soft margin SVM is more commonly used.

##### Soft Margin Classification
Real-world data is rarely perfectly separable. There may be outliers, noise or natural overlap between classes. Soft margin SVM allows for some misclassification or margin violations to achieve better generalization and handle imperfect data.

Slack variables $ξ_i$ are introduced to measure the degree of misclassification or margin violation for each data point:
- $ξ_i = 0$: Point is correctly classified and outside the margin
- $0<ξ_i<1$: Point is correctly classified but inside the margin
- $ξ_i \geq 1$: Point is misclassified

#### Modified Optimization Problem
TODO: Explain this and the C value and the rest of maths foundation

---
## Support Vectors

One of the most elegant properties of SVMs is that most data points don't matter for defining the decision boundary. Only a small, carefully selected subset of points actually determines where the optimal hyperplane lies. This sparse representation is both computationally efficient and conceptually insightful.

**Support Vectors** are the data points that lie exactly on or within the margin boundaries - the points that "touch" or violate the edges of the street. More formally, they are training examples for which:

$$
y_i(w \cdot x_i + b) \leq 1
$$

This includes
- Points exactly on the margin ($y_i(w \cdot x_i + b)=1$)
- Points inside the margin or misclassified ($y_i(w \cdot x_i+b)<1$, only in soft margin SVM)

Properties of Support Vectors:
1. Minimal Representation: Only the support vectors are needed to define the classifier. The decision boundary is determined entirely by these critical points. If we removed all other points from the training set, the decision boundary would remain exactly the same.
2. Sparse Solution: In many practical problems, the number of support vectors is much smaller than the total number of training points. This leads to:
    - Memory-efficient models (only support vectors need to be stored)
    - Faster prediction times
    - Better interpretability (you can examine which training examples were most influential)
3. Critical Points: Support vectors are the most informative points in the dataset. They represent the cases that are hardest to distinguish between classes - the boundary cases where classification is most uncertain.
4. Robustness to Non-Support Vectors: The fact that the solution depends only on support vector makes SVM robust to:
    - Outliers that are far from the decision boundary
    - Changes in the distribution of points that are not near the boundary
    - Redundant or repeated training examples away from the boundary
5. Direct Influence: Support vectors directly determine the position and orientation of the hyperplane. In the mathematical solution, the weight vector w is expressed as a linear combination of only the support vectors

The decision function can be mathematically written in terms of support vectors:
$$
f(x) = \sum (a_i \cdot y_i \cdot K(x_i, x)) + b
$$
Where the sum is only over support vectors (points where $a_i>0$), and:
- $a_i$ are the Lagrange multipliers (learned during training)
- $y_i$ are the class labels of support vectors
- $K(x_i,x)$ is the kernel function (dot product in linear SVM)

This shows that only support vectors contribute to making prediction. This matters because the support vector property explains several important characteristics of SVMs:
1. Efficiency: We can discard all non-support vectors after training without affecting the model
2. Why SVMs work with small datasets: What matters is having good representation of the boundary cases, no necessarily having massive amounts of data. A small dataset with informative boundary examples can produce an excellent SVM
3. Interpretability: We can examine the support vectors to understand which training examples were most critical for defining the decision boundary.
4. Robustness: Adding more training data far from the boundary doesn't change the model, making SVMs stable and less prone to overfitting from redundant data.


---
## The Kernel Trick

The kernel trick is one of the most ingenious innovations in machine learning, allowing SVMs to handle complex, non-linear decision boundaries effectively. 

In the real world, data is rarely linearly separable. In many cases a straight line (or flat hyperplane) simply cannot separate these classes, no matter how we position it. 

The solution is to do feature mapping. The fundamental idea is revolutionary yet simple:
>If we can't separate the data in its current dimension, we project it to a higher dimension where separation becomes possible.

Given an original feature space with features $x \in \mathbb{R}^d$, we define a mapping function:

$$
φ: \mathbb{R}^d -> \mathbb{R}^D
$$

Where D >> d. This function $φ(x)$ transforms each data point into a higher-dimensional representation. In this new space, what appeared non-linearly separable in the original space may become linearly separable.

Let's imagine we have blue points in the center of a flat piece of paper and red points around the edgesIt is impossible to draw a straight line separating center from edges. Any linear boundary will misclassify points. But if we physically lift the center of the paper upward, we add a third dimension. Nowe we can pass a flat plate horizontally that separates the raised center from the flat edges. When we project this flat separator back down to the 2D paper, it appears as a circle - but in 3D, it was simply a flat plane.

This is exactly what feature mapping does mathematically - it finds a flat separator in a higher-dimensional space, which corresponds to a complex, non-linear boundary in the original space.

But naively applying feature mapping would be computationally prohibitive. The computational cost of explicitly computing $φ(x)$ and working in high-dimensional space would make SVMs impractical for most real problems.

The key insight is that the SVM algorithm only ever needs to compute dot products between data points. We never actually need the transformed features themselves, we only need to know what the dot product would be in the transformed space.

A kernel function $K(x_i, x_j)$ directly computes this dot product in the transformed space without ever explicitly performing the transformation:

$$
K(x_i, x_j) = φ(x_i) \cdot φ(x_j)
$$

We compute the kernel function in the original space, but it gives us the same result as if we had:
1. Transformed both points to the high-dimensional space
2. Computed their dot product here

The kernel trick provides:
1. Computational Efficiency
2. Memory Efficiency
3. Mathematical Elegance
4. Infinite Dimensions
5. Flexibility


But not every function can serve as a calid kernel. A function $K(x,y)$ is a valid kernel if and only if it satisfies Mercer's theorem, which essentially requires that:
1. The kernel is symmetric: $K(x,y)=K(y,x)$
2. The kernel is positive semi-definite

This ensures that K actually corresponds to a dot product in some Hilbert space (possibly infinite-dimensional).


---
## Non-Linear SVM

Non-linear SVMs extend the power of linear SVMs to handle complex, real-world data that cannot be separated by straight lines or flat hyperplanes.

Real-world data rarely echibits linear separability. Non-linear SVMs address this through a two-step conceptual process:
1. Feature Mapping - Transform the original input space to a higher-dimensional feature space where the data becomes linearly separable.
2. Linaer Separation in New Space - Apply linear SVM in this higher-dimensional space to find the optimal hyperplane.

When projected back to the original space, this linear hyperplane appears as a complex, non-linear decision boundary.

The challenge is what transformation $φ(x)$ should we use. This is where **kernels** come in. Different kernels correspond to different implicit feature mappings, allowing us to experiment with different types of non-linear boundaries without explicitly constructing the high-dimensional spaces.

The elegance of non-linear SVM lies in its duality:
- Conceptually: We're finding a simple linear separator in a complex high-dimensional space
- Computationally: We're working efficiently with kernels in the original space.
- Practically: We get complex, flexible decision boundaries that can fit real-world data

---
## Types of Kernels
Choosing the right kernel is like choosing the right tool for a specific job. Different kernels create different types of decision boundaries and implicit feature spaces. Here's a comprehensive guide to the most common kernel functions.

#### Linear Kernel
The linear kernel is simply the dot product in the original space - no transformation is applied.

$$
K(x_i, x_j) = x_i \cdot x_j
$$

Characteristics:
- No projection to higher dimensions
- The "rigid ruler" of kernels
- Create straight lines or flat hyperplanes
- Equivalent to standard linear SVM
- Simplest and fastest kernel

Mathematical properties:
- Correspond to identity mapping: $φ(x) = x$
- Decision boundary: $w \cdot x + b = 0$ directly in original space
- Number of parameters: equals the number of features

This is used for Text classification, Problems with many features, When data is already linearly separable, When interpretability is important and Baseline Model.

The advantages of a linear kernel is that it is fast in training and prediction, no hyperparameters to tune (beyond C), it is less prone to overfitting and easier to interpret. On the other hand, it cannot capture non-linear relationships and has a poor performance on inherently non-linear problems

#### Polynomial Kernel

The polynomial kernel function is:
$$
K(x_i, x_j) = (\gamma(x_i \cdot x_j) + r)^d
$$

Where:
- $d$ is the degree of the polynomial
- $\gamma$ is a scaling parameter
- $r$ is the coefficient term

This creates smooth polynomial curves and surfaces. The degree d controls the complexity and flexibity - as higher the degree the more flexible the boundaries are. This can model polynomial interaction between features.

This is mainly used for Image Processing, Pattern Recognition and Problems Requiring smooth, curved boundaries.

Hyperparameters:
- Degree ($d$):
    - $d=2$: Quadratic boundaries
    - $d=3$: Cubic boundaries
    - $d \geq 4$: Very flexible but computationally expensive and prone to overfitting (rarely used)
- Coef0 ($r$):
    - Trades off between higher-order and lower-order terms
    - $r=0: Homogeneous polynomial (all terms have degree exactly d)
    - $r>0$: Includes lower-degree terms

Advantages:
- Captures feature interactions naturally
- Smooth, interpretable boundaries
- Good for problems with known polynomial structure

Limitations: 
- High-degree polynomials can be computationally expensive
- Risk of overfitting with high degrees
- Sensitive to feature scalling 
- Kernel matrix can become numerically unstable with high degrees

#### RBF Kernel (Radial Basis Function)

RBF kernel is also known as the Gaussian kernel because it has a Gaussian (bell-curve) shape.

$$
K(x_i, x_j) = exp(-\gamma ||x_i-x_j||^2)
$$

This is the most powerful and popular kernel. It creates "islands" os complex countours around data points. This is based on the concept of similarity through proximity. It can create circular, eliptical or arbitrary-shaped boundaries and it maps to an infinite-dimensional feature space.

The RBF kernel measures similarity based on distance:
- Points close together $\rightarrow$ High kernel value $\rightarrow$ High similarity $\rightarrow$ Likely different classes
- Points far appart $\rightarrow$ Low kernel value $\rightarrow$ Low similarity $\rightarrow$ Likely different classes

The function value decreases exponentially with distance, creating a localized influence around each point.

##### The $\gamma$ Parameter:
Gamma controls the "reach" or "influence" of each training point. High $\gamma$ values make each training point have very local influence. This creates complex wiggly boundaries that closely follow training data which makes the model really prone to overfit as it creates many small "islands" around individual points.

On the other hand, low values of gamma make each training point has broad influence. This creates smoother, more generalized boundaries which can make the model prone to overfitting as it is too simple and misses important patterns. It behaves like a linear classifier.

There is no universal solution for optimal values of $\gamma$. It is typically found through cross-validation. As a rule of thumb usually it starts with $\gamma = 1/n_{features}$. It is adjusted based on validation performance.

This is the default choice for most non-linear problems such as image recognition, pattern classification and for when the relationship between features and labels is unknown (simply, any problem where linear and polynomial kernels fall).

Advantages:
- Extremely flexible—can approximate any continuous function
- Works well in practice for many problems
- Only one main hyperparameter ($\gamma$) to tune
- Robust across different types of data
- Less prone to numerical instability than polynomial kernels

Limitations:
- Can easily overfit if $\gamma$ is too high
- Less interpretable than linear or polynomial kernels
- More computationally expensive than linear kernel
- Requires careful hyperparameter tuning
- Sensitive to feature scaling


#### Sigmoid Kernel

This is similar to the activation function in neural networks. Sometimes it is called the "neural network kernel". It creates S-shaped boundaries.
$$
K(x_i, x_j) = tanh(\gamma(x_i \cdot x_j) + r)
$$

Where:
- $\gamma$ is a scaling parameter
- r is the coefficient term

This is raerely used in practice. This doesn't always satisfy Mercer's theorem and usually is outperformed by RBF in practice. It is also difficult to tune properly.


#### How to Choose the Kernel

Decision Tree for Kernel Selection
1. Start with Linear Kernel. If we have many features ($N_{features} \geq N_{samples}$) or we need fast training/prediction or we need interpretability we should use it. For text classification or NLP tasks we should also use it.
2. Try RBF Kernel if linear kernel underperforms, we have non-linear relationships, we don't know the structure of our data, we don't have enough computational resources or we cannot tune hyperparameters properly
3. Consider Polynomial Kernel if we have domain knowledge suggesting polynomial relationships, image processing applications or we need smooth boundaries


#### Important Considerations

Feature scaling is critical for RBF and polynomial kernels (not as much for linear kernel but still recommended). We should use functions such as *StandardScaler* or *MinMaxScaler* before training.

For hyperparameter Tuning we should use grid search with cross-validation:
- For RBF: Try $\gamma$ in range $[10^{-4}, 10^1]$
- For polynomial: Try degree in range $[2,4]$
- Always tune C parameter regardess of kernel


Computational cost:
1. Linear: $O(n \times d)$ for prediction
2. Polynomial: $O(n \times d^2)$
3. RBF: $O(n \times d)$ but with more expensive kernel evaluations
4. Polynomial (high degree): Very expensive

It is also possible to combine kernels. However it is rarely done in practice as it adds complexity without guaranteed improvement.

---
## The Dual Problem and Lagrange Multipliers

The dual formulation of SVM is crucial for understanding how the kernel trick works and why SVMs are computationally efficient. This section explores the transformation from the primal problem to the dual problem.

The optimization problem we've discussed so far is called the primal problem. However, solving it directly in high-dimensional feature spaces is computationally challenging. The dual problem offers several advantages:
1. Enables the Kernel trick
2. Computationally efficient
3. Reveals support vectors
4. Handles infinite dimensions

**Primal Problem** *(what we want to solve)*:
