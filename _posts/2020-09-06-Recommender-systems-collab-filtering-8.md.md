---
layout: post
title: Recommender systems with Python - (8) Memory-based collaborative filtering - 5 (k-NN with Surprise)
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

In previous postings, we have gone through core concepts in memory-based collaborative filtering, including the user-item interaction matrix, similarity measures, and user/item-based recommendation. In this posting, let's review those concepts while going through Python implementation using the [Surprise](http://surpriselib.com/) package.


# Preparation

Prior to implementing the models, we need to install the Surprise package (if not installed already) and import it. For more information on installing and importing the Surprise package, please refer to [this tutorial](https://buomsoo-kim.github.io/recommender%20systems/2020/07/18/Recommender-systems-collab-filtering-3.md/)


```python
from surprise import Dataset
from surprise import KNNBasic, BaselineOnly
from surprise.model_selection import cross_validate
```

# Load a built-in dataset

Let's start with loading a built-in dataset. Here, let's use the MovieLens dataset, which is one of the most widely used public datasets in the recommender systems field. The default setting for the ```load_builtin()``` function is downloading the MovieLens data. The data should be downloaded if not downloaded earlier. 

```python
dataset = Dataset.load_builtin()
```

Let's see how many user and items are present in the data. There are 100,000 rating instances, and this is why this data is called *ml-100k*. FYI, MovieLens provides much larger datasets - you can check them out [here](https://grouplens.org/datasets/movielens/).

```python
ratings = dataset.raw_ratings

print("Number of rating instances: ", len(ratings))
print("Number of unique users: ", len(set([x[0] for x in ratings])))
print("Number of unique items: ", len(set([x[1] for x in ratings])))
```
<div style="background-color:rgba(250, 202, 220,.30); padding-left: 15px; padding-top: 10px; padding-bottom: 10px; padding-right: 15px">
Number of rating instances:  100000
<br>Number of unique users:  943
<br>Number of unique items:  1682
</div>  

# Training & evaluation

## Baseline estimator

Let's start with the baseline algorithm that we had implemented in the [previous posting](https://buomsoo-kim.github.io/recommender%20systems/2020/07/18/Recommender-systems-collab-filtering-3.md/). It is always good to have a baseline for benchmark - this gives you a yardstick of how good (or bad) your algorithm is.

```python
clf = BaselineOnly()
cross_validate(clf, dataset, measures=['MAE'], cv=5, verbose=True)
```

It seems that the baseline algorithm shows the average mean absolute error of around 0.75 in a five-fold cross validation.

```python
Estimating biases using als...
Estimating biases using als...
Estimating biases using als...
Estimating biases using als...
Estimating biases using als...
Evaluating MAE of algorithm BaselineOnly on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
MAE (testset)     0.7456  0.7479  0.7462  0.7521  0.7494  0.7482  0.0023  
Fit time          0.42    0.47    0.32    0.30    0.30    0.36    0.07    
Test time         0.11    0.23    0.10    0.10    0.18    0.14    0.05    
{'fit_time': (0.4205029010772705,
  0.47090888023376465,
  0.3164703845977783,
  0.3028404712677002,
  0.2975282669067383),
 'test_mae': array([0.7456069 , 0.74790715, 0.74621253, 0.75207455, 0.74940825]),
 'test_time': (0.10664844512939453,
  0.2300584316253662,
  0.10011959075927734,
  0.10106658935546875,
  0.18408536911010742)}
```

## k-Nearest Neighbor (k-NN)

Let us move on to k-NN, which is a simple memory-based collaborative filtering algorithm. Now, you can implement your first memory-based recommender system! 


### Similarity options

An important parameter for k-NN-based algorithms in Surprise is ```sim_options```, which describes options for similarity calculation. Using ```sim_options```, you can set how similarity is calculated, such as similarity metrics. The default setting is using the Mean Squared Difference (MSD) to calculate pairwise similarity and user-based recommendations. 

```python
sim_options = {
    'name': 'MSD',
    'user_based': 'True'
}
```

You can change the similarity calculation method by converting the value mapped to the ```name``` key. Also, item-based recommender systems can be easily implemented by changing ```user_based``` value to ```False```.

```python
sim_options = {
    'name': 'pearson',
    'user_based': 'False'
}
```


### User-based k-NN

Let's try implementing a user-based k-NN model with the default similarity calculation method, i.e., MSD. 

```python
sim_options = {
    'name': 'MSD',
    'user_based': 'True'
}

clf = KNNBasic(sim_options = sim_options)
cross_validate(clf, dataset, measures=['MAE'], cv=5, verbose=True)
```

It seems that the average MAE is around 0.77 with a five-fold cross validation. This is actually worse than the baseline algorithm!

```python
Computing the msd similarity matrix...
Done computing similarity matrix.
Computing the msd similarity matrix...
Done computing similarity matrix.
Computing the msd similarity matrix...
Done computing similarity matrix.
Computing the msd similarity matrix...
Done computing similarity matrix.
Computing the msd similarity matrix...
Done computing similarity matrix.
Evaluating MAE of algorithm KNNBasic on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
MAE (testset)     0.7727  0.7709  0.7686  0.7761  0.7776  0.7732  0.0033  
Fit time          0.34    0.39    0.39    0.41    0.42    0.39    0.03    
Test time         3.71    3.81    3.82    3.93    3.74    3.80    0.08    
{'fit_time': (0.3350553512573242,
  0.3878347873687744,
  0.39234137535095215,
  0.40755629539489746,
  0.4163696765899658),
 'test_mae': array([0.77267657, 0.77093624, 0.76855595, 0.77607487, 0.77755156]),
 'test_time': (3.7132723331451416,
  3.8124566078186035,
  3.8213281631469727,
  3.9288363456726074,
  3.736668348312378)}
```


### Item-based k-NN (& changing the similarity metric)

Maybe the reason for a poor performance is wrong hyperparameters. Let's try a user-based system with the pearson coefficient as a similarity metric.

```python
sim_options = {
    'name': 'pearson',
    'user_based': 'False'
}

clf = KNNBasic(sim_options = sim_options)
cross_validate(clf, dataset, measures=['MAE'], cv=5, verbose=True)
```

However, to our frustration, the performance gets even worse with MAE over 0.80! Does this mean that k-NN performs actually worse than the baseline model, which is a dumb enough model? 

```python
Computing the pearson similarity matrix...
Done computing similarity matrix.
Computing the pearson similarity matrix...
Done computing similarity matrix.
Computing the pearson similarity matrix...
Done computing similarity matrix.
Computing the pearson similarity matrix...
Done computing similarity matrix.
Computing the pearson similarity matrix...
Done computing similarity matrix.
Evaluating MAE of algorithm KNNBasic on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
MAE (testset)     0.7984  0.8060  0.8006  0.8021  0.8081  0.8030  0.0036  
Fit time          1.46    1.39    1.57    1.51    1.52    1.49    0.06    
Test time         3.80    4.01    3.74    3.88    3.86    3.86    0.09    
{'fit_time': (1.4587411880493164,
  1.3949718475341797,
  1.5693511962890625,
  1.5131325721740723,
  1.5200302600860596),
 'test_mae': array([0.79836295, 0.80602847, 0.80057509, 0.80210933, 0.80808347]),
 'test_time': (3.799623966217041,
  4.005688190460205,
  3.7438101768493652,
  3.882983446121216,
  3.8566648960113525)}
```


### Other considerations

Nonetheless, don't be frustrated yet. Since k-NN is a relatively more sophisticated model than the baseline model, there are some other considerations that should be fully accounted for. In the following posting, let's see how we can yield optimal performance with k-NN. 



# References

- Koren, Y. (2010). Factor in the neighbors: Scalable and accurate collaborative filtering. ACM Transactions on Knowledge Discovery from Data (TKDD), 4(1), 1-24.
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In Recommender systems handbook (pp. 1-35). Springer, Boston, MA.