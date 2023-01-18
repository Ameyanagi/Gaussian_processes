# Bayes theorem
## Fundamentals of the equation

Bayes theorem is a mathematical formula that describes the probability of an event occurring, based on prior knowledge of conditions that might be related to the event. It is used in statistics and probability theory to calculate the probability of an event occurring, given certain conditions. 

The formula is the following.

$$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$$

This can be understood more easily by as following.

$$P(A|B) = \frac{P(B|A) * P(A)}{P(B|A) * P(A) + P(B|A^c) * P(A^c)}$$

Where $A$ is the event we are interested in, and $B$ is the event that has occurred. $A^c$ is the complement of $A$, which is the event that $A$ does not occur.

You can play around with the following interactive graph that someone made [https://www.skobelevs.ie/BayesTheorem/](https://www.skobelevs.ie/BayesTheorem/)

![Geometrical Bayes](img/Geometrical_Bayes.png)

## Deriving Gaussian Processes from Bayes Theorem
### Integration of Posterior
First we think about integration of posterior. Let's keep in mind the relationship of conditional probability and integration.

$$P(a) = \int_{-\infty}^{\infty} P(a|b)P(b)db$$

We can obtain P(a) by integrating the probability distribution for all condition b.

### Maximum Likelihood Estimation (MLE) 
There is two method of learning a model $w$ from data $D$, which is Maximum Likelihood Estimation (MLE) and Maximum Aposteriori Estimation (MAP). $ y = w^Tx $ or $y \sim \mathcal{N}(w^Tx, \sigma^2)$.

MLE is maximizing the likelihood of the data given the model.

$$P(D|w) = \prod_{i=1}^N P(y_i| x_i, w)$$
$$ D = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$$

### Maximum a Posteriori Estimation (MAP)
MAP is maximizing the posterior probability of the model given the data.
$$P(w|D) \propto \frac{P(D|w)P(w)}{Z}$$
$$P(y_i|x_i, w) = \mathcal{N}(w^Tx_i, \sigma^2)$$

The ultimate goal of calculating $w$ is to predict $y$ given $x$ using $w$. Can we get rid of $w$ in the equation?

Instead of following flow,
$$D \rightarrow w \rightarrow y = w^Tx$$

Can we do the following?
$$D \rightarrow y = f(D, x)$$

## Instead of calculating w, why don't we calculate $P(y | x, D)$?
We can do this by maximizing $P(y | x, D)$, which is the probability of $y$ given $x$ and $D$.

$$ P(y | x, D) = \int P(y | x, D, w) dw$$

$w$ depend on $D$, but and $D$ and $y$ depend only on $x, w$. Therefore, we can write the following.

$$P(y | x, D) = \int_w P(y | x, w) P(w | D) dw$$

We know that $\int P(y | x, w)$ is gauss function.

Here, We assume that $P(w)$ is a gaussian distribution. (Be careful that this is a assumption!!)

$$P(w | D) = \frac{P(D | w) P(w)}{Z}$$

$$\begin{align*}
P(y | x, D) &= \int_w P(y | x, w) \frac{P(D | w) P(w)}{Z} dw
\end{align*}$$

$\int gaussian * gaussian * gaussian$ is a gaussian function.

This will give the full description about the distribution of probability of $y$ with given $x$ and $D$.

## Understanding the meaning of $P(y | x, D)$
From the fact that $P(y | x, D)$ is a gaussian function, we can understand the following.

$$P(y | x, D) \sim \mathcal{N}(\mu, \Sigma)$$

Which is the same as the following.

$$
P\left(\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \\ y \end{bmatrix}, [x_1, x_2, ..., x_n, x] \right) \sim \mathcal{N}(\mu, \Sigma)
$$

The question is, "How can we obtain the covariance matrix $\Sigma$?".

Property of $\Sigma$ is the following.
1. Covariance matrix $\Sigma$ has a positive semi-definite property. This means that the eigenvalues of $\Sigma$ are all positive. Therefore, $\Sigma$ is a kernel function.
2. Similar points need a large value in covariance matrix $\Sigma$.

One possible candidate is Radius Basis Function (RBF) kernel.

$$K(x, x') = \exp \left( -\gamma\frac{||x - x'||^2}{2\sigma^2} \right)$$

## Gaussian Process Regression (GPR)

$$P(y | x, D) = P\left(\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \\ y \end{bmatrix}, [x_1, x_2, ..., x_n, x] \right) \sim \mathcal{N}(0, \Sigma)$$

$$\Sigma = K, K_{ij} = K(x_i, x_j)$$

$$P(y | x, D) = \mathcal{N}(0, K)$$

This part needs to be reviewed later on.
$$P(y_+ | y_1, ..., y_n, x_+, x_1, ..., x_n, x_+) \sigma \mathcal{N}(K_*^TK^{-1}y, K_{**} - K_{*}^TK^{-1}K_{*})$$

$$\Sigma = \begin{bmatrix} K & K_* \\ K_*^T & K_{**} \end{bmatrix}$$

Now, $K_{*}^TK^{-1}y$ is the kernel regression and $K_{**} - K_{*}^TK^{-1}K_{*}$ is uncertainty.


## Application to Bayesian Optimization
We will talk about optimization that maximizes the function $f(x)$.

### Probability Inprovement (PI)
Probability Inprovement (PI)

If we set $x^+ = \argmax_{x_i \in x_{i:t}} f(x_i)$, then the probability of improving $f$ by adding new point $x$ is the following.

$$\begin{align*}
PI(x) &= P(f(x) > f(x^+)) \\
&= \int_{f(x^+))}^{\infty} \mathcal{N}(f(x)|\mu_x, \sigma_x^2) df(x) \\
&= \Phi \left( \frac{\mu_x - f(x^+)}{\sigma_x} \right)
\end{align*}$$

This method tries to find the point that has the highest probability of improving the current best solution. Wether the improvement is large or small is not considered.

Trade-off parameter is introduce to control the trade-off between exploration and exploitation. The larger the trade-off parameter, the more exploration is performed.

$$\begin{align*}
PI(x) &= P(f(x) > f(x^+) + \xi) \\
&= \int_{f(x^+) + \xi}^{\infty} \mathcal{N}(f(x)|\mu_x, \sigma_x^2) df(x) \\
&= \Phi \left( \frac{\mu_x - f(x^+) - \xi}{\sigma_x} \right)
\end{align*}$$


### Expected Improvement (EI)
Expected Improvement (EI) is a function that measures expetation of the improvement of the current best solution $x^+$.

$$EI(x) = \mathbb{E} \left[ \max(f(x) - f(x^+), 0) \right]$$

$f(x^+)$ is the current best solution and $x^+$ is the location of the current best solution. $$x^+ = \argmax_x f(x)$$

If we think of probability for the improvement $I$

$$ I(x) = \max\{f(x) - f(x^+), 0\} $$

$$ f(x) \sim \mathcal{N}(\mu_x, \sigma_x^2) $$

$$ I(x) \sim \mathcal{N}(\mu_x - f(x^+), \sigma_x^2) $$

$$P(I) = \frac{1}{\sqrt{2\pi}\sigma_x} \exp \left( -\frac{( I - \mu_x + f(x^+))^2}{2\sigma_x^2} \right)$$

$$EI(x) = \int_{0}^{\infty} \frac{1}{\sqrt{2\pi}\sigma_x} \exp \left( -\frac{( I - \mu_x + f(x^+))^2}{2\sigma_x^2} \right) I dI$$

If we set $t = \frac{I - \mu_x + f(x^+)}{\sigma_x}$,then at $I = 0$, $t = \frac{-\mu_x + f(x^+)}{\sigma_x}$, and at $I = \infty$, $t = \infty$.
$dt = \frac{1}{\sigma_x}dI$, $I = \sigma_x t + \mu_x - f(x^+)$

$$\begin{align*}
EI(I) &= \int_{t = \frac{-\mu_x + f(x^+)}{\sigma_x}}^{\infty} \frac{1}{\sqrt{2\pi}\sigma_x} \exp \left( -\frac{t^2}{2} \right) (\sigma_x t + \mu_x - f(x^+)) \sigma_x dt \\
&= \sigma_x \int_{t = \frac{-\mu_x + f(x^+)}{\sigma_x}}^{\infty} \frac{t}{\sqrt{2\pi}} \exp \left( -\frac{t^2}{2} \right) dt \\
& + (\mu_x - f(x^+) )\int_{t = \frac{-\mu_x + f(x^+)}{\sigma_x}}^{\infty} \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{t^2}{2} \right) dt \\
\end{align*}$$

If we set $t^2 = u$, $tdt = \frac{1}{2}du$

$$\begin{align*}
\sigma_x \int_{t = \frac{-\mu_x + f(x^+)}{\sigma_x}}^{\infty} \frac{t}{\sqrt{2\pi}} \exp \left( -\frac{t^2}{2} \right) dt
&= 
\sigma_x \int_{u = \frac{(-\mu_x + f(x^+))^2}{2\sigma_x^2}}^{u = \infty} \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{u}{2} \right) \frac{1}{2}du \\
&= \frac{\sigma_x}{\sqrt{2 \pi}} \left[ - \exp \left( -\frac{u}{2} \right) \right]_{u = t^2}^{u = \infty} \\
&= \frac{\sigma_x}{\sqrt{2 \pi}} \left[ - \exp \left( -\frac{(I -\mu_x + f(x^+))^2}{2\sigma_x^2} \right) \right]_{u = t^2}^{u =\infty} \\
&= \frac{\sigma_x}{\sqrt{2 \pi}} \left( 0 + \exp \left( -\frac{( -\mu_x + f(x^+))^2}{2\sigma_x^2} \right) \right)\\
&= \sigma_x \mathcal{N} \left( \frac{-\mu_x + f(x^+)}{\sigma_x}|0 , 1 \right) \\
&= \sigma_x \phi \left( \frac{-\mu_x + f(x^+)}{\sigma_x} \right) \\
&= \sigma_x \Phi \left( \frac{\mu_x - f(x^+)}{\sigma_x} \right) \\
\end{align*}$$

$$\begin{align*}
 (\mu_x - f(x^+) )\int_{t = \frac{-\mu_x + f(x^+)}{\sigma_x}}^{\infty} \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{t^2}{2} \right) dt &= (\mu_x - f(x^+) ) \left( 1 - \int_{t = -\infty}^{t = \frac{-\mu_x + f(x^+)}{\sigma_x}} \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{t^2}{2} \right) dt \right) \\
&= (\mu_x - f(x^+) ) \left( 1 - \Phi \left( \frac{-\mu_x + f(x^+)}{\sigma_x} \right) \right) \\
&= (\mu_x - f(x^+) ) \Phi \left( \frac{\mu_x - f(x^+)}{\sigma_x} \right)
\end{align*}$$


$$EI(x) = \left\{ \begin{array}{ll} (\mu(x) - f(x^+)) \Phi(Z) + \sigma(x) \phi(Z) & \sigma(x) > 0 \\ 0 & \sigma(x) = 0 \end{array} \right.$$

$$ Z = \left\{ \begin{array}{ll} \frac{\mu(x) - f(x^+)}{\sigma(x)} & \sigma(x) > 0 \\ 0 & \sigma(x) = 0 \end{array} \right.$$

where $\mu(x)$ and $\sigma(x)$ are the mean and the standard deviation of the GP posterior predictive at x. $\Phi(z)$ and $\phi(z)$ are the cumulative distribution function and the probability density function of the standard normal distribution.

The expected improvement with trade-off parameter $\xi$ is defined as following.

$$EI(x) = \left\{ \begin{array}{ll} (\mu(x) - f(x^+) - \xi) \Phi(Z) + \sigma(x) \phi(Z) & \sigma(x) > 0 \\ 0 & \sigma(x) = 0 \end{array} \right.$$

$$ Z = \left\{ \begin{array}{ll} \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)} & \sigma(x) > 0 \\ 0 & \sigma(x) = 0 \end{array} \right.$$

### Upper Confidence Bound (UCB)
UCB is maximizing the predicted value + constant*standard deviation.

$$UCB(x) = \mu(x) + \kappa \sigma(x)$$


## Appendix: Ordinary Least Squares (OLS)

OLS is a linear regression model. 

$y_i \in \mathbb{R}$, $x_i \in \mathbb{R}^d$. OLS assumes that the data of $y_i$ is in linear relationship with $x_i$.

Thus we can write the following. ($\epsilon_i$ comes from central limit theorem)
$$y_i = w^Tx_i + \epsilon_i$$
$$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

or we can write the following.

$$ y_i = \mathcal{N}(w^Tx_i, \sigma^2)$$


$$P(y_i | x_i, w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(y_i - w^Tx_i)^2}{2\sigma^2} \right)$$

## Maximum Likelihood Estimation (MLE) approach to solve the OLS
$$ \argmax_w \prod_{i=1}^N P(y_i | x_i, w)$$ 
$$ \argmax_w \sum_{i=1}^N \log P(y_i | x_i, w) = \argmax_w \sum_{i=1}^N \log \left(\frac{1}{\sqrt{2\pi\sigma^2}} \right) - \frac{(y_i - w^Tx_i)^2}{2\sigma^2}$$

$\log\left(\frac{1}{\sqrt{2\pi\sigma^2}} \right)$ and $\frac{1}{2\sigma^2}$ are constants, so we can ignore them when we are maximizing the function.

$$ \argmin_w \frac{1}{n}\sum_{i=1}^N (y_i - w^Tx_i)^2$$

## Maximum Aposteriori Estimation (MAP) approach to solve the OLS

$$P(w | y_1, ..., y_n)$$

If we denote $D = {y_1, ..., y_n}$, and combining the Bayes theorem, we can write the following.

$$P(w | D) \propto \frac{P(D | w) P(w)}{Z}$$

$P(w)$ is the prior distribution of $w$. $P(D | w)$ is the likelihood function. $Z$ is the normalization constant. 
$w$ is a random variable, and we assume that $w$ is a gaussian distribution.

$$ P(w) = \mathcal{N}(0, \tau^2)$$

$$ \begin{align*} 
\argmax_w P(w | D) &= \argmax_w \prod_{i=1}^N P(y_i | x_i, w) P(w) \\
&= \argmax_w \sum_{i=1}^N \log P(y_i | x_i, w) + \log P(w) \\
&= \argmax_w \sum_{i=1}^N \log \left(\frac{1}{\sqrt{2\pi\sigma^2}} \right) - \frac{(y_i - w^Tx_i)^2}{2\sigma^2} + \log \left(\frac{1}{\sqrt{2\pi\tau^2}} \right) - \frac{w^Tw}{2\tau^2} \\
&= \argmin_w \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - w^Tx_i)^2 + \frac{n}{2\tau^2} w^Tw \\
&= \argmin_w \frac{1}{n}\sum_{i=1}^N (y_i - w^Tx_i)^2 + \lambda ||w||^2_2
\end{align*}$$

where $\lambda = \frac{\sigma^2}{\tau^2}$


### Calculation of $w$
$$ \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^N (y_i - w^Tx_i)^2 ( + \lambda ||w||^2_2)$$

$X^T = [x_1, x_2, ..., x_n\ ]$, $Y = [y_1, y_2, ..., y_n\ ]$

$$ \begin{align*}
 (Xw - Y)^2 &= (Xw - Y)^T(Xw - Y) \\
    &= w^TX^TXw - 2w^TX^TY + Y^TY \\
 \end{align*}$$

If we take the derivative of $\mathcal{L}(w)$, we can get the following.

$$2X^TXw - 2X^TY = 0$$
$$ w = (X^TX)^{-1}X^TY$$





