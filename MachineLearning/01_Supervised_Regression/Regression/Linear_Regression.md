# Linear Regression: Learning to Draw a Line Through the World

<div style="text-align: center; margin: 20px 0;">
  <p style="font-size: 1.4em; margin-bottom: 8px;">
    <i>"Of all the principles which can be proposed for this purpose, I think there is none more general, more exact, and more easy of application, than that which consists of rendering the sum of the squares of the errors a minimum."</i>
  </p>
  <p style="font-size: 0.9em; color: #777;">
    Adrien-Marie Legendre, <em>Nouvelles Méthodes pour la Détermination des Orbites des Comètes</em>, 1805
  </p>
</div>



## The Central Question

Before a single equation is written, it is worth asking where the word *regression* even comes from. In 1886, the Victorian polymath Francis Galton published a paper titled "Regression Towards Mediocrity in Hereditary Stature." He had spent years collecting the heights of parents and their adult children, and when he plotted them, he noticed something surprising: the children of unusually tall parents were, on average, *shorter* than their parents, while the children of unusually short parents were, on average, *taller*. Extreme heights did not persist across generations - they regressed toward the population mean. Galton called the line that best described this parent-child relationship the "regression line," and the name stuck. The word now survives as a historical fossil: every time we train a linear regression model to predict house prices or exam scores, we are invoking an observation Galton made about heredity in Victorian England.

The mathematical machinery behind regression is older still, and its origin is one of the most famous priority disputes in the history of science. In 1805, the French mathematician Adrien-Marie Legendre published a nine-page appendix to a treatise on comet orbits in which he described, for the first time in print, the method of minimizing the sum of squared errors - what we now call *least squares*. Within ten years, the method had been adopted as a standard tool in astronomy and geodesy across Europe. Then in 1809, Carl Friedrich Gauss published his own celestial mechanics treatise, casually claiming he had been using the method since 1795 - fourteen years before Legendre's publication. The ensuing dispute between two mathematical giants over who deserved credit was, second only to the Newton–Leibniz controversy over calculus, the most famous priority dispute in the history of mathematics. Gauss ultimately received more of the credit, and for good reason: where Legendre presented least squares as a pure computational procedure, Gauss connected it to probability theory and the normal distribution, transforming a fitting algorithm into a statistical framework with deep theoretical guarantees.

That connection - between fitting a line and making probabilistic statements about the world - is the deep thread of this chapter. Linear regression is not merely a way to draw a best-fit line; it is a coherent statistical model with assumptions, theoretical guarantees, and inferential machinery. Understanding it at this level reveals why the same mathematical ideas appear at the core of logistic regression, neural networks, and almost every supervised learning algorithm that follows.



## The Model and What It Assumes

The simplest version of the problem is this: we have $n$ training examples, each consisting of an input $x_i$ and an output $y_i$, and we want to find a straight line that best describes the relationship between them. A straight line is described by two numbers: its **slope** $\beta_1$, which controls how steeply it rises or falls, and its **intercept** $\beta_0$, which determines where it crosses the vertical axis when $x = 0$. The model's prediction for any input $x$ is

$$\hat{y} = \beta_0 + \beta_1 x.$$

The hat on $\hat{y}$ is a notation convention meaning "our estimate of $y$." It is a prediction, not an observed value. The observed value is $y_i$; the prediction is $\hat{y}_i = \beta_0 + \beta_1 x_i$. If the slope is positive, the predicted output increases as the input grows. If the slope is zero, the input has no effect on the prediction. The intercept sets the baseline level of the prediction when all inputs are zero.

Most real problems involve more than one input. **Multiple linear regression** extends the model to $d$ input features, simply by adding more terms - one coefficient per feature:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_d x_d.$$

Here, $x_1, x_2, \ldots, x_d$ are the $d$ feature values for a given example, and $\beta_1, \beta_2, \ldots, \beta_d$ are the corresponding coefficients. Each coefficient $\beta_j$ answers the question: *how much does the prediction change when feature $j$ increases by one unit, holding all other features constant?* That last phrase - "holding all other features constant" - is easy to overlook but crucial. In a house-price model, the coefficient on square footage tells you the price increase per additional square foot *after accounting for the number of bedrooms and location*. This is different from the raw correlation between size and price.

The model's one non-negotiable assumption is **linearity**: the relationship between the inputs and the output is captured by a straight line (or flat hyperplane, in higher dimensions). Non-linear relationships - exponential growth, seasonal cycles, diminishing returns - are not handled by this formula. This is a genuine limitation, but it is less restrictive than it sounds. The *inputs* can be non-linear transformations of the original data: including $x^2$ as an additional feature alongside $x$, or taking the logarithm of income before using it as a predictor, allows linear regression to fit many curved relationships while keeping its theoretical guarantees intact. What must remain linear are the *coefficients*, not the features themselves.

<figure_placeholder: linear_regression_intro.png - Two side-by-side scatter plots. Left: simple linear regression - x on the horizontal axis, y on the vertical axis, a straight line through the data, with vertical red lines connecting each point to the line labeled "residuals." The line equation ŷ = β₀ + β₁x is annotated with arrows indicating the intercept (where the line crosses the y-axis) and the slope (rise/run). Right: a 3D scatter plot with two features x₁ and x₂ on the horizontal axes and y on the vertical axis. The regression is now a flat plane rather than a line, illustrating the extension to multiple features.>



## Why Squared Errors? The Least Squares Criterion

Before we can find the best-fit line, we need to define what "best" means. A natural definition is: the line whose predictions are closest to the observed values. The difference between the observed value $y_i$ and the model's prediction $\hat{y}_i$ for the same input is called the **residual**:

$$e_i = y_i - \hat{y}_i.$$

A positive residual means the model underpredicted; a negative residual means it overpredicted. We want these residuals to be small. The question is how to combine $n$ individual residuals into a single number to minimize.

Summing them directly fails immediately - positive and negative residuals cancel, and a line that wildly overshoots half the data and undershoots the other half can score a sum of zero. We could sum their absolute values, which avoids cancellation. Legendre's proposal - the one that won - was to square each residual before summing. The **Mean Squared Error**, which we will write as MSE, averages these squared residuals over all training examples:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2.$$

Reading this formula: for each of the $n$ training examples, compute the difference between the true value $y_i$ and the prediction $\hat{y}_i$, square that difference, then average all the squares. The result is a single number - our measure of how badly the current line fits the data. The smaller the MSE, the better the fit.

Why squaring rather than absolute values? Two reasons. The practical reason is that squaring produces a smooth, differentiable function - it has no sharp corners - which allows us to find the minimum using calculus and to derive elegant closed-form solutions. The absolute value function, by contrast, is not differentiable at zero, which complicates optimization. The deeper reason is probabilistic: squaring is the right penalty when you believe measurement errors are distributed like a bell curve, symmetric around zero, with large deviations being rare. Under that belief, minimizing the MSE is equivalent to finding the most likely coefficients given the data - a connection we will develop shortly.

Squaring also penalizes large errors disproportionately: a residual of 10 contributes 100 to the sum, while a residual of 1 contributes only 1. This is both a strength and a weakness. It means the model fights hardest against its worst predictions, which is often what we want. But a single anomalous data point can exert enormous influence, pulling the fitted line far from where it would otherwise sit. When outliers are a concern, robust alternatives - such as the Huber loss, which behaves like squared error near zero but grows more slowly for large residuals - are available and worth considering.

<figure_placeholder: residuals_and_mse.png - A scatter plot of data points with the fitted regression line. For three selected points, vertical red lines connect each point to the line, labeled e₁, e₂, e₃. A shaded square is drawn at each point whose side length equals the absolute residual, illustrating that squaring penalizes large residuals much more than small ones. A formula box shows MSE = (1/n)Σeᵢ². The total shaded area represents the MSE value.>



## Finding the Optimal Coefficients: Two Routes

We have a precisely defined objective: find the values of $\beta_0, \beta_1, \ldots, \beta_d$ that minimize the MSE. There are two paths to the answer, each revealing something different about the structure of the problem.

### The Normal Equation: The Closed-Form Solution

Because the MSE is a smooth, bowl-shaped function of the coefficients, it has a single lowest point that can be found by calculus. We ask: at what coefficient values does the slope of the MSE become zero (i.e., where are we at the bottom of the bowl)?

To express this solution compactly, it helps to stack all the data into matrix form. Imagine arranging all $n$ training examples as rows in a matrix $\mathbf{X}$, where each row is one training example and each column is one feature (with an extra column of all 1s to handle the intercept). Separately, stack all $n$ target values into a single column vector $\mathbf{y}$. Setting the derivative of the MSE to zero and solving yields the **Normal Equation**:

$$\hat{\boldsymbol{\beta}} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}.$$

What this formula says in plain language: given the data matrix $\mathbf{X}$ and the target vector $\mathbf{y}$, the optimal coefficients $\hat{\boldsymbol{\beta}}$ can be computed directly - no iteration, no guessing, no learning rate. The result is exact. The superscript $T$ denotes the *transpose* of a matrix (rows and columns are flipped), and the $-1$ superscript denotes the *matrix inverse* (the matrix equivalent of division). For a researcher comfortable with linear algebra, this formula is a complete and beautiful statement. For a beginner, the key takeaway is simply: there exists a direct formula that computes the best-fit coefficients in a single step.

The Normal Equation has two important limitations. First, it requires computing the matrix inverse of $\mathbf{X}^T\mathbf{X}$. If two features are perfectly correlated - like "temperature in Celsius" and "temperature in Fahrenheit" - this matrix has no inverse and the formula breaks down. Even near-perfect correlation makes the result numerically unstable. Second, when the number of features $d$ is very large (hundreds of thousands, as in text or genomics data), forming and inverting $\mathbf{X}^T\mathbf{X}$ becomes computationally impractical. This is where the second route becomes essential.

### Gradient Descent: The Iterative Solution

Gradient descent is the workhorse of modern machine learning. The idea is intuitive: imagine standing on a hilly landscape where altitude represents the MSE. You want to find the lowest valley. You cannot see the whole landscape at once, but you can always tell which direction is downhill from where you are currently standing. So you take a small step downhill, re-evaluate, and repeat until you stop descending.

In practice, "downhill" is determined by the **gradient** of the MSE - a vector that points in the direction of steepest increase. Moving in the *negative* gradient direction (downhill) decreases the MSE. Starting from an initial guess for the coefficients (often all zeros), the algorithm iterates:

1. Compute the model's predictions with the current coefficients.
2. Compute the residuals (predictions minus true values).
3. Compute the gradient of the MSE - how the error changes as each coefficient changes.
4. Update each coefficient by moving a small step in the direction that reduces the error.
5. Repeat until the error stops improving.

The **learning rate** $\alpha$ controls the size of the step. Too large a step and the algorithm overshoots the minimum and diverges; too small a step and progress is painfully slow. For a concrete formula: if the current estimate of the coefficients is $\boldsymbol{\beta}^{(t)}$, the update rule is

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \cdot \nabla \text{MSE}(\boldsymbol{\beta}^{(t)}),$$

where $\nabla \text{MSE}$ is the gradient vector of the MSE. For linear regression specifically, this gradient has a clean form - it is proportional to the residuals weighted by the feature values - and Chapter 3 on Optimizers derives it in full detail.

For linear regression, the MSE surface is convex - it has exactly one minimum, like a perfect bowl, with no false valleys. This means gradient descent is guaranteed to find the global optimum regardless of where it starts. This convexity guarantee is shared with the SVM's quadratic program. It is *not* shared by neural networks, where the loss surface can have many local minima, and managing that non-convexity becomes a central challenge.

When the training set is very large, computing the gradient over all $n$ examples at every step is expensive. **Stochastic Gradient Descent (SGD)** approximates the gradient using a randomly selected subset (a **mini-batch**) of training examples at each step. The estimate is noisy but, on average, points in the right direction. Interestingly, this noise has a beneficial regularizing effect - a topic explored in depth in Chapter 3.

<figure_placeholder: gradient_descent_linear.png - Two panels. Left: a bird's-eye view of the MSE loss surface as a function of β₀ and β₁, shown as nested elliptical contours. The global minimum is marked with a star at the center. The path of gradient descent is shown as a sequence of arrows spiraling inward from an initial starting point toward the minimum. Right: a plot of training MSE versus iteration number, showing rapid initial decrease then slower convergence. The learning rate α is annotated, and the MSE achieved by the Normal Equation is shown as a dashed line at the bottom.>



## The Probabilistic Interpretation: Why MSE Is the Right Loss

So far, minimizing the MSE has been motivated geometrically - it measures how far the predictions are from the data. But there is a deeper justification that connects linear regression to the broader framework of probabilistic modeling used throughout this book.

The key insight is to think of each observation not as an exact measurement of the truth, but as the truth plus some random noise. If we assume that the noise on each observation is drawn independently from a bell-curve (Gaussian) distribution centered at zero, then we are saying: the true underlying relationship is linear, and each measurement adds a random error that is more likely to be small than large, and equally likely to be positive or negative. Formally, we write this as:

$$y_i = (\beta_0 + \beta_1 x_{i1} + \cdots + \beta_d x_{id}) + \varepsilon_i,$$

where $\varepsilon_i$ is the random noise term for example $i$, which we assume follows a Gaussian distribution with mean zero and some fixed variance $\sigma^2$.

Under this probabilistic model, the Maximum Likelihood Estimator - the coefficient values that make the observed data most probable - turns out to be exactly the least squares solution. Minimizing the MSE is identical to maximizing the likelihood of the data under a Gaussian noise model. This is not a coincidence; it is the same principle we saw in the Logistic Regression chapter, where minimizing cross-entropy is equivalent to maximizing a Bernoulli likelihood. The loss function is not an engineering choice - it is the inference procedure that follows from a specific belief about how noise enters the data.

This probabilistic framing also reveals something important: the coefficients $\hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_d$ are themselves random quantities. They depend on the training data, and the training data contains noise. If we collected a different training sample - same process, different random noise - we would get slightly different coefficient estimates. The spread of those hypothetical estimates is called the **standard error** of each coefficient. Standard errors are what allow us to ask statistical questions like "is this coefficient reliably different from zero, or might it be noise?" We return to this in the next section.



## Statistical Inference: Are the Coefficients Real?

In scientific applications, fitting a regression model is not just about prediction. It is also about understanding which features genuinely influence the outcome. A fitted coefficient of 0.3 might mean "this feature really does affect the outcome" or it might mean "this is noise; a different dataset would give a different sign." How do we tell the difference?

The answer comes from comparing each coefficient to its standard error. If the standard error is small relative to the coefficient's magnitude, we have consistent evidence across hypothetical resamples that the coefficient is truly non-zero. If the standard error is large, the coefficient could easily be noise. The ratio of coefficient to standard error - called the **t-statistic** - is the basic unit of evidence. A large t-statistic (say, above 2 in absolute value) is conventionally taken as evidence that the coefficient is real. The corresponding **p-value** measures: if the true coefficient were zero, how likely would we be to observe a t-statistic this large by chance? A p-value below 0.05 is the conventional threshold for declaring significance.

The **F-test** zooms out and asks a related question: does the model as a whole explain anything? It compares the variance explained by all the features together to the variance in a null model that simply predicts the mean for every observation. A significant F-statistic means the features collectively provide useful information; a non-significant F-statistic means the model is not reliably better than guessing the average.

For a researcher connecting this to the Logistic Regression chapter: the machinery is identical. The same Hessian matrix that governed optimization in logistic regression provides the variance-covariance matrix of the estimated coefficients, enabling inference. For linear regression under Gaussian noise, this is particularly clean: the variance-covariance matrix of the coefficients is $\sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}$, the same matrix that appears in the Normal Equation, scaled by the noise variance. Optimization and inference share the same algebraic object - the **Gauss-Markov theorem** formalizes this, proving that the least squares estimator has the minimum variance among all linear unbiased estimators, regardless of whether the noise is exactly Gaussian.



## Evaluation Metrics

A trained model needs to be assessed on held-out test data that was never seen during training. Several metrics measure different aspects of predictive quality.

**Mean Squared Error (MSE)** is the same quantity we minimized during training. On test data, it measures the average squared gap between predictions and true values. Its units are the square of the output's units, which can make interpretation awkward - predicting house prices in dollars, the MSE is in dollars-squared, which is hard to interpret intuitively.

**Root Mean Squared Error (RMSE)** fixes this by taking the square root of the MSE. An RMSE of \$10,000 in a house-price model means predictions are, on average, about \$10,000 off. RMSE is in the same units as the output, making it directly interpretable.

**Mean Absolute Error (MAE)** is the average of the absolute residuals - it measures how far off the predictions are on average, without squaring. MAE is less sensitive to large individual errors (outliers) than RMSE, because it does not square the residuals. Use MAE when large individual errors are not disproportionately bad; use RMSE when they are.

The metric that answers "what fraction of the variation in the outcome does the model explain?" is the **coefficient of determination**, written $R^2$. Its formula is

$$R^2 = 1 - \frac{\displaystyle\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\displaystyle\sum_{i=1}^n (y_i - \bar{y})^2}.$$

The numerator is the total squared error of the model - how much variance remains after the model makes its predictions. The denominator is the total squared deviation of the labels from their mean $\bar{y}$ - how much variance existed in the data before the model. Their ratio is the fraction of variance the model *failed* to explain, and subtracting it from 1 gives the fraction the model *did* explain. An $R^2$ of 0.81, for example, means the model explains 81% of the variance in the target. An $R^2$ of 0 means the model does no better than predicting the mean for every point. A negative $R^2$ is possible on test data and means the model is worse than that baseline - a sign of severe overfitting.

$R^2$ has one well-known trap: adding more features can only *increase* $R^2$ on training data, even if those features are pure noise. The model will always find some way to slightly reduce its training error by adding a coefficient. **Adjusted $R^2$** corrects for this by penalizing the model for each additional feature it uses:

$$R^2_\text{adj} = 1 - \left(1 - R^2\right) \cdot \frac{n - 1}{n - d - 1}.$$

Here, $n$ is the number of training examples and $d$ is the number of features. The penalty factor $(n-1)/(n-d-1)$ grows as we add more features ($d$ increases), applying increasingly harsh penalties. Adjusted $R^2$ can actually decrease when a new feature adds less explanatory power than the penalty costs, making it a more honest guide to model quality than raw $R^2$.



## Multiple Linear Regression and Multicollinearity

When many features are used simultaneously, two problems become especially important.

**Multicollinearity** occurs when some features are strongly correlated with each other - not just with the output, but *with each other*. Temperature in Celsius and temperature in Fahrenheit are a trivial example of perfect correlation. More realistic examples include "number of rooms" and "house area," or "years of education" and "age at first job." The problem is not in the predictions - a multicollinear model can still make accurate predictions. The problem is in the *coefficients*, which become numerically unstable. When two features are highly correlated, the model can achieve roughly the same fit by assigning a large positive weight to one and a large negative weight to the other, in almost any combination. Small changes in the training data cause the estimated coefficients to swing wildly, making them useless for interpretation. The **Variance Inflation Factor (VIF)** for each feature quantifies the severity: a VIF above 10 for feature $j$ is a conventional warning sign that its coefficient is being destabilized by its correlation with the other features.

**Overfitting** is the second concern. As the number of features $d$ grows toward the number of training examples $n$, the model gains the ability to fit the training data almost perfectly - but it does so by memorizing the noise, not the signal. A model that perfectly fits its training data may generalize poorly to new observations. The standard diagnostic is to compare training error to validation error: if training MSE is much lower than validation MSE, the model is overfitting. The remedy is regularization, described next.



## Regularization: When the Solution Needs Constraint

When multicollinearity or overfitting are present, the ordinary least squares solution can behave badly. The fix is to add a penalty term to the objective that discourages large coefficients - the model must now balance fitting the data against keeping its weights small.

**Ridge regression** adds a penalty proportional to the sum of all squared coefficients. The new objective is to minimize:

$$\frac{1}{n}\sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2 + \lambda \sum_{j=1}^d \beta_j^2.$$

The first term is the familiar MSE - fit the data well. The second term penalizes large coefficients - stay simple. The parameter $\lambda \geq 0$ controls the balance: $\lambda = 0$ recovers ordinary least squares, while large $\lambda$ forces all coefficients toward zero. Ridge shrinks all coefficients *proportionally* - strong ones remain nonzero but smaller, weak ones become very small. It particularly helps with multicollinearity: the instability that caused coefficients to swing wildly is damped by the penalty, and the Normal Equation always has a stable solution.

**Lasso regression** (Least Absolute Shrinkage and Selection Operator) uses a penalty on the absolute values of the coefficients rather than their squares:

$$\frac{1}{n}\sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2 + \lambda \sum_{j=1}^d |\beta_j|.$$

The difference seems minor, but the geometric consequence is striking: while Ridge shrinks all coefficients toward zero without ever making them exactly zero, Lasso can drive some coefficients to *exactly* zero, effectively removing those features from the model. Lasso performs automatic feature selection. A Lasso model trained on a hundred features may produce non-zero coefficients for only twenty of them, identifying the informative subset and discarding the rest. This sparsity is particularly valuable in genomics, text analysis, and financial modeling, where the number of candidate features far exceeds the number of training examples and most features are noise.

The Regularization chapter develops both methods in depth, explains the geometric reason for Lasso's sparsity, and introduces Elastic Net - a combination of both penalties that inherits Lasso's sparsity and Ridge's stability simultaneously.



## Assumptions and When They Fail

The theoretical guarantees of linear regression rest on several assumptions. Understanding them and knowing what to do when they are violated is what separates principled analysis from mechanical curve-fitting.

*Linearity* requires that the true relationship between the inputs and the output is, at least approximately, linear. Violation shows up as a systematic pattern in plots of residuals against fitted values - the model consistently overshoots in some regions and undershoots in others. The remedy is often to add polynomial features, include interaction terms between features, or transform right-skewed variables by taking their logarithm.

*Independence of errors* requires that the residual for one observation does not predict the residual for another. This is almost always violated in time-series data, where today's error tends to predict tomorrow's, a phenomenon called *autocorrelation*. It is also violated in spatial data, where nearby observations tend to share correlated errors. Independence violations do not bias the coefficient estimates but invalidate the standard errors and hypothesis tests - the reported p-values will be wrong.

*Homoscedasticity* - constant variance of the errors - is violated when the spread of residuals grows or shrinks as the predicted value increases. This is common in economic data (income uncertainty grows with income level) and measurement processes (uncertainty grows with the measured quantity). It does not bias coefficients but distorts their standard errors in different regions of the prediction range.

*Normality of errors* is assumed for the validity of t-tests and F-tests. In practice, by the Central Limit Theorem, coefficient estimates are approximately normally distributed for large $n$ even if individual errors are not, so regression inference is reasonably robust to moderate violations of this assumption.

*No multicollinearity* and *no omitted variable bias* have been discussed above. Of these, omitted variable bias - when a feature that truly affects the outcome is left out of the model - is the most consequential and the most commonly encountered in observational studies. If that omitted feature is correlated with the included features, the estimated coefficients become biased in ways that cannot be corrected without additional data or more sophisticated methods.

<figure_placeholder: regression_assumptions.png - A 2×2 grid of diagnostic plots. Top left: residuals plotted against fitted values with no pattern - horizontal scatter around zero, the ideal. Top right: same plot but with a curved band through the residuals, indicating a non-linear relationship missed by the model. Bottom left: residuals fanning out as fitted values increase, indicating heteroscedasticity. Bottom right: a Q-Q plot comparing the empirical distribution of residuals to the theoretical normal distribution - on a good model, points fall on the 45-degree diagonal; curvature in the tails signals non-normal errors.>



## Looking Forward

Linear regression is the simplest member of a unified family of models, and understanding it deeply is the most efficient foundation for everything that follows. Logistic regression keeps the linear combination of features but applies the sigmoid function, replacing the Gaussian noise assumption with a Bernoulli likelihood and the MSE loss with cross-entropy. Support Vector Regression keeps the concept of a loss function with a flat zero region but replaces the squared penalty with the $\varepsilon$-insensitive tube. Decision tree regression replaces the global linear model with a hierarchical partition of the feature space, predicting a constant within each region. And neural networks go furthest of all: they learn the features themselves, applying stacked layers of linear transformations and non-linear activations to discover representations from which the output is approximately linearly predictable - and then pass those representations through a linear regression head to produce the final prediction.

The gradient descent logic developed here - define a differentiable loss, compute its gradient, step downhill, repeat - is identical in form across all of these models. Only the loss and the architecture change. Chapter 2 on Backpropagation will show how gradients are computed efficiently through deep networks using the chain rule, and Chapter 3 on Optimizers will show how the plain gradient descent described here is extended to handle non-convex landscapes, adaptive step sizes, and mini-batch noise. But the underlying logic is already complete in fitting the simplest straight line.



## Key Takeaways

Linear regression fits the model $\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d$ by finding the coefficients $\beta_j$ that minimize the **Mean Squared Error** - the average squared difference between the predicted and observed values; squaring the errors is not arbitrary but is equivalent to Maximum Likelihood Estimation under the assumption that each observation equals its true linear prediction plus a random Gaussian noise term.

The **Normal Equation** computes the exact optimal coefficients in a single matrix calculation, but becomes numerically unstable under multicollinearity (highly correlated features) and computationally impractical when the number of features is very large; **gradient descent** provides the iterative alternative - starting from any initial coefficients and repeatedly stepping in the direction that reduces the error - and is the foundational optimization algorithm used throughout all subsequent models in this book.

The **Gauss-Markov theorem** proves that under uncorrelated equal-variance errors, the least squares estimator is the best linear unbiased estimator - the one with the smallest variance among all unbiased linear estimators; this theoretical guarantee justifies least squares even when the Gaussian noise assumption does not hold exactly, and the same algebraic structure that gives the optimal coefficients also provides the **standard errors** that enable hypothesis tests and confidence intervals.

**$R^2$** measures the fraction of variance in the output that the model explains, ranging from 1 (perfect fit) to 0 (no better than predicting the mean), and possibly below 0 on test data when the model overfits; **Adjusted $R^2$** corrects $R^2$ for the number of features used, penalizing model complexity so that adding noise features decreases rather than increases the score.

**Ridge regression** and **Lasso regression** extend linear regression by adding a penalty on the size of the coefficients: Ridge shrinks all coefficients proportionally and stabilizes multicollinear solutions, while Lasso drives some coefficients to exactly zero and performs automatic feature selection - both controlled by a regularization parameter $\lambda$ that is tuned via cross-validation.



## Further Reading

- Legendre, A. M. (1805). *Nouvelles Méthodes pour la Détermination des Orbites des Comètes*. Paris: Courcier. (The first publication of the least squares method, in a nine-page appendix. Worth reading as a historical document alongside Gauss's 1809 work.)

- Gauss, C. F. (1809). *Theoria Motus Corporum Coelestium in Sectionibus Conicis Solem Ambientum*. Hamburg: Perthes and Besser. (English translation published 1857.) (Where Gauss connected least squares to the normal distribution and maximum likelihood, laying the foundations of mathematical statistics.)

- Galton, F. (1886). Regression towards mediocrity in hereditary stature. *Journal of the Anthropological Institute of Great Britain and Ireland*, 15, 246–263. (The paper that coined "regression" in the statistical sense. Galton's observation that children of extreme parents regress toward the population mean is the historical origin of the term now used for all linear prediction models.)

- Stigler, S. M. (1981). Gauss and the invention of least squares. *Annals of Statistics*, 9(3), 465–474. (The definitive historical investigation of the Gauss-Legendre priority dispute, with documentary evidence arguing Gauss probably possessed the method first but failed to communicate it.)

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapter 3. Springer. (The rigorous graduate-level treatment of linear methods: OLS, Ridge, Lasso, and the geometric perspective. Chapter 3 derives everything from first principles and remains the standard reference for researchers.)

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267–288. (The original Lasso paper. Introduced L1 penalization for linear regression, producing sparse solutions with automatic feature selection - one of the most practically important regularization methods ever developed.)

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. (The original Ridge regression paper, introducing L2 penalization as a solution to instability from correlated features.)

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 3. Springer. (A rigorous Bayesian treatment: the probabilistic model, maximum likelihood, Bayesian inference over coefficients, and the connection between regularization and Gaussian or Laplace priors on the coefficients.)

- Senn, S. (2011). Francis Galton and regression to the mean. *Significance*, 8(3), 124–126. (A readable account of Galton's original discovery and the statistical phenomenon of regression to the mean - with entertaining examples of how it continues to mislead researchers today.)