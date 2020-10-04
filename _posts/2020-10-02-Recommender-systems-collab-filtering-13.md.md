---
layout: post
title: Matrix Factorization for Recommender Systems - Collaborative filtering with Python 13
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

In the [previous posting](https://buomsoo-kim.github.io/recommender%20systems/2020/09/25/Recommender-systems-collab-filtering-12.md/), we learned how vanilla matrix factorization (MF) models work for the rating prediction task. In this posting, let's see how different variants of MF are optimized for performance.


# Basic optimization scheme

As briefly discussed earlier, virtually every MF model essentially tries to minimize the difference between observed and predicted rating. And predicted rating is the dot product of corresponding user ($q_u$) and item  ($p_i$) latent factors. Therefore, we get the basic optimization scheme:

\begin{equation}
minimize \sum_{u, i} difference(r_{ui} - q_{u}p_{i})
\end{equation}


The difference can be calculated using various metrics such as mean squared error or mean absolute error. In the rest of this posting, let's see how this simple optimization scheme can be improved for a better prediction performance.

 
# SVD

*SVD*, or *FunkSVD* since it was first popularized by Simon Funk, is one of the pioneering models in MF for collaborative filtering. At first, many (including me) would be curious to see the relation to the [singular value decomposition (SVD) of a square matrix](https://en.wikipedia.org/wiki/Singular_value_decomposition) in linear algebra. However, the SVD model in MF does not bear much resemblance to SVD commonly known in linear algebra. In fact, this was what left me very much confused in the outset. I won't go into the details here, but please refer to [this posting](https://www.freecodecamp.org/news/singular-value-decomposition-vs-matrix-factorization-in-recommender-systems-b1e99bc73599/) at freecodecamp.com for more information. In short, if you don't know much about SVD, it doesn't matter, and if you are familar with SVD in linear algebra, don't expect to see the same thing here!


### Objective function

It has two distinct features from the vanilla MF that we have seen earlier - (1) user/item biases and (2) regularization terms. First, the baseline estimates for users and items are incorporated as user ($b_u$) and item biases ($b_i$). Thus, the estimated ratings is:


\begin{equation}
\hat{r_{ui}} = \mu + b_u + b_i + q_{u}p_{i}
\end{equation}

where $\mu$ is the overall average rating. Also, the squared error is regularized to prevent overfitting to a user and/or item. By adding the two to the optimization objective, SVD generally demonstrates a superior performance over validation test data. 

\begin{equation}
minimize \sum_{u, i} (r_{ui} - (\mu + b_u + b_i + q_{u}p_{i}))^2 + \lambda(b_u^2 + b_i^2 + {\parallel q_u \parallel}^2 + {\parallel p_i \parallel}^2)
\end{equation}


### Learning

A simple stochastic gradient descent (SGD) algorithm can be applied to update the parameters and learn the patterns. Since the objective is to minimize the difference between the predicted and actual ratings, parameters ($b_u, b_i, q_u, p_i$) are updated by moving towards *the opposite direction of the computed gradient*. This would be easier to understand if you grasp the intuition behind [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) in machine learning in general. Once the error term $e_{ui} = r_{ui} - \hat{r_{ui}}$ is computed, parameters are updated following the below scheme.


\begin{equation}
b_u \leftarrow b_u + \gamma (e_{ui} - \lambda b_u) \\
b_i \leftarrow b_i + \gamma (e_{ui} - \lambda b_i) \\
q_u \leftarrow q_u + \gamma (e_{ui} - \lambda q_u) \\
p_i \leftarrow p_i+ \gamma (e_{ui} - \lambda p_i) \\
\end{equation}


$\lambda$, just like $\gamma$ are hyperparameters that should be determined before training the model. As we have seen, a higher level of $\gamma$ will regularize the parameters and suppress overfitting. Whereas, $\lambda$ is used to control the size of each step in the parameter update. Both should be set to optimal values meticuously using tuning methods such as grid search. 


# SVD++

SVD is a great choice when we have reliable and sizable explicit feedback information from users such as ratings. Nevertheless, in many cases, we have much richer implicit information such as search history, page visits, likes and sharing, and bookmarks. Such information can be effectively used to improve the learning process whether or not explicit feedback is abundant or not. SVD++ is one of the methods that exploits such information with a slight modification in SVD. 

To model the implicit feedback, an additional term is added to $q_u$ to represent the user. The revised user factor is as below.

\begin{equation}
q_u + {|R(u)|}^{-1/2} + \sum_{j \in R(u)} y_j
\end{equation}
 
where $R(u)$ is the set of items rated by the user $u$ and $y_j$ is anothger factor vector to represent each item in $R(u)$. Hence, to represent a user, SVD++ combines the user factor learned from explicit ratings ($q_u$) and implicit information from items that the user has rated (or searched, shared, visited, etc.) previously ($R(u)$). In other words, it *combines the model-based and memory-based approaches in a single equation.* 


# References


- Funk, S. (2006) Netflix Update: Try This at Home. (https://sifter.org/~simon/journal/20061211.html)
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In Recommender systems handbook (pp. 1-35). Springer, Boston, MA.
