---
layout: post
title: Matrix Factorization with fast.ai - Collaborative filtering with Python 16
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

In this posting, let's start getting our hands dirty with fast.ai. [fast.ai](https://www.fast.ai/) is a Python package for deep learning that uses Pytorch as a backend. It provides modules and functions that can makes implementing many deep learning models very convinient. More information on fast.ai can be found at the [documentation](https://docs.fast.ai/quick_start.html). Here, we will be just implementing collaborative filtering models, but if you want to learn more about deep learning and fastai, I strongly recommend starting with the [Practical deep learning with coders course](https://course.fast.ai/) by Jeremy Howard. 



# Data Import

Let's start with importing the MovieLens 100k data that we used before with the Surprise package. You can use functions provided by fast.ai, but let us try doing it from scratch so that you can import any data later on. If you are manually downloading the data, please download the zip file by [clicking](http://files.grouplens.org/datasets/movielens/ml-100k.zip) and unzip it. If you are using Google Colab or Jupyter Notebook like me, use below command. For more information on downloading files from the Web in Colab, please refer to [this posting](https://buomsoo-kim.github.io/colab/2020/05/03/Colab-downloading-files-from-web.md/).

```python
!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip
```

```python
--2020-11-27 22:14:57--  http://files.grouplens.org/datasets/movielens/ml-100k.zip
Resolving files.grouplens.org (files.grouplens.org)... 128.101.65.152
Connecting to files.grouplens.org (files.grouplens.org)|128.101.65.152|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4924029 (4.7M) [application/zip]
Saving to: ‘ml-100k.zip’

ml-100k.zip         100%[===================>]   4.70M  12.2MB/s    in 0.4s    

2020-11-27 22:14:58 (12.2 MB/s) - ‘ml-100k.zip’ saved [4924029/4924029]

Archive:  ml-100k.zip
   creating: ml-100k/
  inflating: ml-100k/allbut.pl       
  inflating: ml-100k/mku.sh          
  inflating: ml-100k/README          
  inflating: ml-100k/u.data          
  inflating: ml-100k/u.genre         
  inflating: ml-100k/u.info          
  inflating: ml-100k/u.item          
  inflating: ml-100k/u.occupation    
  inflating: ml-100k/u.user          
  inflating: ml-100k/u1.base         
  inflating: ml-100k/u1.test         
  inflating: ml-100k/u2.base         
  inflating: ml-100k/u2.test         
  inflating: ml-100k/u3.base         
  inflating: ml-100k/u3.test         
  inflating: ml-100k/u4.base         
  inflating: ml-100k/u4.test         
  inflating: ml-100k/u5.base         
  inflating: ml-100k/u5.test         
  inflating: ml-100k/ua.base         
  inflating: ml-100k/ua.test         
  inflating: ml-100k/ub.base         
  inflating: ml-100k/ub.test  
```

Just to check if the file is downloaded and unzipped properly, run below command. 

```python
!ls
```

If you see a ```ml-100k``` folder, it is well done!

```python
ml-100k  ml-100k.zip  sample_data
```

Finally, we can import the downloaded data with ```read_csv``` function in Pandas. 

```python
import pandas as pd

data = pd.read_csv('ml-100k/u.data', sep="\t", header = None, 
            usecols = [0,1,2],
            names = ['user', 'item', 'rating', 'timestamp'])

print(data.shape)
data.head()
```


# Creating Data Bunch

The primary data structured used in fastai is data bunch, which utilizes data loader in Pytorch. For collaborative filtering tasks, fastai provides ```CollabDataBunch```, which makes our life much easier.

Let's start by dividing the data frame into training and testing data. We divide the data in a 7-3 ratio.

```python
train_df = data.iloc[:70000]
test_df = data.iloc[70000:]
```

Since we the data is in data frame format, we use ```from_df()``` function to create a ```CollabDataBunch```. Note that the test data is passed onto the ```test``` parameter. Other key parameters are ```valid_pct```, which is the proportion of valid dataset and ```bs```, which refers to the batch size.

```python
databunch = CollabDataBunch.from_df(train_df, test = test_df, valid_pct = 0.1, bs=128)
databunch.show_batch()
```


# Creating and training a matrix factorization model

Simple collaborative filtering models can be implemented with ```collab_learner()```. Note that we have to set ```y_range```, which shows possible range of values that the target variable, i.e., rating in this case, can take. 

```python
learn = collab_learner(databunch, n_factors=50, y_range=(0, 5))
learn.model
```

The basic ```collab_learner``` model is ```EmbeddingDotBias``` - this is identical to the SVD model that we have seen before. The model has four parameters - ```u_weight```, ```i_weight```, ```u_bias```, and ```i_bias```; we will later see what these parameters refer to . 

```python
EmbeddingDotBias(
  (u_weight): Embedding(944, 50)
  (i_weight): Embedding(1622, 50)
  (u_bias): Embedding(944, 1)
  (i_bias): Embedding(1622, 1)
)
```

To train the model with the given data, we use ```fit()``` function. We train 5 epochs here.

```python
learn.fit(5)
```

```python
epoch	train_loss	valid_loss	time
0	0.953882	0.913318	00:06
1	0.808686	0.853414	00:06
2	0.677575	0.839932	00:06
3	0.551616	0.858383	00:06
4	0.422990	0.894348	00:06
```

To evaluate the model on the test data, we can use ```get_preds()``` function to get model predictions and convert them into a NumPy array.

```python
from sklearn.metrics import *

y_pred = learn.get_preds(ds_type = DatasetType.Test)[0].numpy()
print(mean_absolute_error(test_df["rating"], y_pred))
```

The model shows test MAE of around 0.75. This seems to on a similar level with the [performance shown by MF using Surprise](https://buomsoo-kim.github.io/recommender%20systems/2020/10/22/Recommender-systems-collab-filtering-14.md/), although we did not run cross validation here.

```python
0.7538161431928476
```


In this posting, we have seen how to import data and implement a simple matrix factorization model using fastai. In following postings, let's see we can implement deep recommender models with fastai.


# References

- Collaborative filtering tutorial. (https://docs.fast.ai/tutorial.collab)
- Collaborative filtering using fastai. (https://towardsdatascience.com/collaborative-filtering-using-fastai-a2ec5a2a4049)

