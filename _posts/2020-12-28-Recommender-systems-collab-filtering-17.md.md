---
layout: post
title: Neural collaborative filtering with fast.ai - Collaborative filtering with Python 17
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

In the [previous posting](https://buomsoo-kim.github.io/recommender%20systems/2020/11/27/Recommender-systems-collab-filtering-16.md/), we learned how to train and evaluate a matrix factorization (MF) model with the [fast.ai](https://www.fast.ai/) package. Nowadays, with sheer developments in relevant fields, neural extensions of MF such as NeuMF (He et al. 2017) and Deep MF (Xue et al. 2017) became very popular. In this posting, let's have a look at a very simple variant of MF using multilayer perceptron.



# Data Import & Preparation

These steps are identical to preparing for MF. If you haven't yet, please have a look at this [previous posting](https://buomsoo-kim.github.io/recommender%20systems/2020/11/27/Recommender-systems-collab-filtering-16.md/) for importing and preparing the data.


# Creating and training a neural collaborative filtering model

We use the same ```collab_learner()``` function that was used for implementing the MF model. Parameters that should be changed to implement a neural collaborative filtering model are ```use_nn``` and ```layers```. Setting ```use_nn``` to ```True``` implements a neural network. Recall that the MF model had only embedding layers for users and items. ```layers``` parameter lets us define the architecture of the neural network. In specific, we can designate the numbers of nodes in hidden layers. Here, let's set it to ```[30, 30]``` - by doing so, we are generating a neural network with two hidden layers having 30 nodes each.

```python
learn = collab_learner(databunch, n_factors=50, y_range=(0, 5), use_nn = True, layers = [30, 30])
learn.model
```

You can see that the resulting model has three additional ```Linear()``` layers. The final one is the output layer and the first two are the hidden layers that we have configured. Note that ```out_features``` for the two layers are set to 30.

```python
EmbeddingNN(
  (embeds): ModuleList(
    (0): Embedding(944, 74)
    (1): Embedding(1625, 101)
  )
  (emb_drop): Dropout(p=0.0, inplace=False)
  (bn_cont): BatchNorm1d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=175, out_features=30, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Linear(in_features=30, out_features=30, bias=True)
    (4): ReLU(inplace=True)
    (5): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Linear(in_features=30, out_features=1, bias=True)
  )
```

To train the model with the given data, we use ```fit()``` function. We train 5 epochs here.

```python
learn.fit(5)
```

```python
epoch train_loss  valid_loss  time
0 0.969993  0.921394  00:07
1 0.907009  0.894607  00:06
2 0.866429  0.886609  00:06
3 0.863333  0.877121  00:06
4 0.822304  0.874548  00:06
```

To evaluate the model on the test data, we can use ```get_preds()``` function to get model predictions and convert them into a NumPy array.

```python
from sklearn.metrics import *

y_pred = learn.get_preds(ds_type = DatasetType.Test)[0].numpy()
print(mean_absolute_error(test_df["rating"], y_pred))
```

The model shows slightly improved performance compared to MF. You can experiment on other configurations, e.g., making the model deeper by adding more layers or wider by adding nodes to the hidden layers. 

```python
0.7473991450190545
```


# References

- Collaborative filtering tutorial. (https://docs.fast.ai/tutorial.collab)
- Collaborative filtering using fastai. (https://towardsdatascience.com/collaborative-filtering-using-fastai-a2ec5a2a4049)
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).
- Xue, H. J., Dai, X., Zhang, J., Huang, S., & Chen, J. (2017, August). Deep Matrix Factorization Models for Recommender Systems. In IJCAI (Vol. 17, pp. 3203-3209).
