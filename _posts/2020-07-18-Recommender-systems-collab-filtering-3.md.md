---
layout: post
title: Recommender systems with Python - (3) Introduction to Surprise package in Python
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

In the [previous posting](https://buomsoo-kim.github.io/recommender%20systems/2020/07/14/Recommender-systems-collab-filtering-2.md/), we overviewed collaborative filtering (CF) and two types of CF methods - *memory-based* and *model-based* methods. In this posting, before going into the details of two CF methods, let's have a quick look at the __Surprise__ package in Python.


# What is Surprise!?

[Surprise](http://surpriselib.com/) is a Python scikit specialized for recommender systems. It provides built-in public datasets, ready-to-deploy CF  algorithms, and evaluation metrics.  


# Installing Surprise

Installing Surprise is straightforward like any other scikit libraries. You can conveniently install it using __pip__. In terminal console, run below command. 

```python
pip install surprise
```

<p align = "center">
<img src ="/data/images/2020-07-31/1.PNG" width = "600px" class="center">
</p>

If you are using Google colaboratory or Jupyter Notebook, run below code in any cell.

```python
!pip install surprise
```

<p align = "center">
<img src ="/data/images/2020-07-31/0.PNG" width = "600px" class="center">
</p>


After installation, let's import necessary submodules for this exercise.

```python
from surprise import Dataset
from surprise import BaselineOnly
from surprise.model_selection import cross_validate
```

# Built-in datasets

Surprise provides built-in datasets and tools to create custom data as well. The built-in datasets provided are from [MovieLens](https://grouplens.org/datasets/movielens/), a non-commercial movie recommendation system, and [Jester](http://eigentaste.berkeley.edu/dataset/), a joke recommender system. Here, let's use the Jester built-in dataset for demonstration. For more information on Jester dataset, please refer to [this page](http://eigentaste.berkeley.edu/dataset/).

## Load dataset

The built-in dataset can be loaded using ```load_builtin()``` function. Just type in the argument ```'jester'``` into the function.

```python
dataset = Dataset.load_builtin('jester')
```

If you haven't downloaded the dataset before, it will ask whether you want to download it. Type in "Y" and press Enter to download.

<p align = "center">
<img src ="/data/images/2020-07-31/2.PNG" width = "600px" class="center">
</p>

## Data exploration

You don't need to know the details of the dataset to build a prediction model for now, but let's briefly see how the data looks like. The raw data can be retrieved using ```raw_ratings``` attribute. Let's print out the first two instances. 

```python
ratings = dataset.raw_ratings

print(ratings[0])
print(ratings[1])
```

The first two elements in each instance refer to the **user ID** and **joke ID,** respectively. The third element shows ratings and I honestly don't know about the fourth element (let me know if you do!). Therefore, the first instance shows User #1's rating information for Joke #5. 

<div style="background-color:rgba(250, 202, 220,.30); padding-left: 15px; padding-top: 10px; padding-bottom: 10px; padding-right: 15px">
('1', '5', 0.219, None)
<br>('1', '7', -9.281, None)
</div>

Now let's see how many users, items, and rating records are in the dataset.

```python
print("Number of rating instances: ", len(ratings))
print("Number of unique users: ", len(set([x[0] for x in ratings])))
print("Number of unique items (jokes): ", len(set([x[1] for x in ratings])))
```

It seems that 59,132 users left 1,761,439 ratings on 140 jokes. That seems like a lot of ratings for jokes!

<div style="background-color:rgba(250, 202, 220,.30); padding-left: 15px; padding-top: 10px; padding-bottom: 10px; padding-right: 15px">
Number of rating instances:  1761439
<br>Number of unique users:  59132
<br>Number of unique items (jokes):  140
</div>


# Prediction and evaluation

There are a few prediction algorithms that can be readily used in Surprise. [The list](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html) includes widely-used CF methods such as [k-nearest neighbor](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic) and [probabilistic matrix factorization](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD). But since we haven't looked into the details of those methods, let's use the [```BaselineOnly``` algorithm](https://surprise.readthedocs.io/en/stable/basic_algorithms.html), which predicts "baseline estimates," i.e., calculating ratings using just bias terms of users and items. To put it simply, it does not take into account complex interaction patterns between users and items - considers only "averaged" preference patterns pertaining to users and items. 

```python
clf = BaselineOnly()
```

Let's try 3-fold cross validation, which partitions the dataset into three and use different test set for each round. 

```python
cross_validate(clf, dataset, measures=['MAE'], cv=3, verbose=True)
```

On average, the prediction shows a mean average error (MAE) of 3.42. Not that bad, considering that we used a very naive algorithm. In the following postings, let's see how the prediction performance will improve with more sophisticated prediction algorithms!

<p align = "center">
<img src ="/data/images/2020-07-31/3.PNG" width = "600px" class="center">
</p>




# References

- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In Recommender systems handbook (pp. 1-35). Springer, Boston, MA.