# Bayes Neural Network

Notation
$\theta$ is the parameter of the neural network.
$x$ is the input of the neural network.
$y$ is the output of the neural network.

## Normal Neural Network
Normal Neural Network is a neural network try to learn $\theta$ by minimizing the loss function $\mathcal{L}(w) = (y - wx)^2 $ by using the gradient descent method.

## Bayesian Neural Network
Bayesian Neural Network is a neural network try to learn $\theta$ by maximizing the posterior distribution $P(y | x, \theta)$.

### Maximum Likelihood Estimation (MLE)
If we assume that $P(y | x, \theta)$ is a Gaussian distribution, then we can use the Maximum Likelihood Estimation (MLE) to learn $\theta$.
$$P(D | \theta) = \prod_{i=1}^N P(y_i | x_i, \theta)$$

$$P(y | x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y - wx)^2}{2\sigma^2}\right)$$




### Maximum a Posteriori Estimation (MAP)

$$P(\theta | D) \propto \frac{P(D | \theta)P(\theta)}{Z}$$

