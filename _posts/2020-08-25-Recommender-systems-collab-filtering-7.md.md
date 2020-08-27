---
layout: post
title: Recommender systems with Python - (7) Memory-based collaborative filtering - 4 
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

So far, we have seen how users and items can be represented as vectors and users' feedback records on items as entries in the *user-item interaction matrix*. Furthermore, using the matrix, we can calculate how two users or items are similar to each other. In this posting, let's see how we can actually predict prospective ratings and recommend items by combining what we have reviewed so far.


# User-based recommendation vs. Item-based recommendation

In the [previous posting](https://buomsoo-kim.github.io/recommender%20systems/2020/08/18/Recommender-systems-collab-filtering-6.md/), we learned that users and items both can be represented as vectors and similarities can be calculated for each of user or item pairs. Therefore, item recommendation can be performed in two different ways, namely *user-based* and *item-based*. 

## User-based recommendation

For user-based recommendations, similarity calculation and neighborhood identification are performed among users. That is, we calculate similarity measures between users, i.e., user vectors in the interaction matrix, and identify $k$ nearest users. Then, we calculate the average rating among those $k$ users. 

Note that the list of $k$ nearest users can differ based on which item that we want to estimate ratings for. This is because some users could have not rated the item of interest. For instance, assume that we want to predict the rating on *The departed* by Krusty. Let's assume Willie is the most similar user to Krusty. However, Willie did not watch *The departed* and rate the movie as a result. Thus, we need to neglect Willie's record and find other users that is both *similar to Krusty* and *have watched and rated the movie*.

Once $k$-nearest users satisfying the both conditions are found, the rating can be estimated by simply averaging the neighbors' ratings that are given to the item of interest.

\begin{equation}
\hat{r_{ui}} = \frac{1}{|N_i(u)|} \displaystyle\sum_{v \in N_i(u)} (r_{vi})
\end{equation}

In some practical cases, simply averaging can be insufficient. Therefore, many improvements such as weighing the ratings based on similarity metrics and normalizing the ratings have been proposed. However, this is also contingent upon the context - it is important to understand the problem and data before choosing/improving your averaging scheme.


## Item-based recommendation

Instead of finding similar users and averaging ratings on the item of interest, item-based recommendation tries to find similar items to the item of interest. Then, we average the ratings to those items given by the user. Again, we cannot include the items that have not been previously rated by the user to the neighborhood. Once the $k$-nearest items are retrieved, we can average the ratings using the following equation.

\begin{equation}
\hat{r_{ui}} = \frac{1}{|N_u(i)|} \displaystyle\sum_{j \in N_u(i)} (r_{uj})
\end{equation}

Here, similar improvements such as weighting and normalization can be employed.


# User- or Item-based, which one is better?

Traditionally, user-based recommendation methods were employed in practice. However, they were reported to have severe scalability and sparsity issues, especially with very large datasets that are prevelant nowadays. Hence, large-scale web applications such as Amazon.com started to move onto item-based collaborative filtering (Sarwar et al. 2001, Linden et al. 2003). In my opinion, both are methodologically sound, strong baselines with appealing features such as convenient implementation and explainability of the results. Thus, it would be meaningful to employ both methods and compare the results with more sophisticated methods that we are going to have a look in the following postings.



# References

- Linden, G., Smith, B., & York, J. (2003). Amazon. com recommendations: Item-to-item collaborative filtering. IEEE Internet computing, 7(1), 76-80.
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In Recommender systems handbook (pp. 1-35). Springer, Boston, MA.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001, April). Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web (pp. 285-295).