---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (12) - 다양한 CNN 구조
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 합성곱 신경망 6 - 다양한 CNN 구조

Objective: 케라스로 다양한 CNN 모델을 만들어 본다.

[지난 포스팅](https://buomsoo-kim.github.io/keras/2018/05/05/Easy-deep-learning-with-Keras-11.md/)에서 케라스로 deep CNN 모델을 만들어 보고 mnist 데이터 셋에서의 검증 정확도를 99% 이상으로 끌어올리는 데 성공하였다.

CNN 모델은 mnist와 같이 이미지 형태의 데이터에 흔히 사용되는 것으로 알려져 있지만, 음성, 영상, 텍스트 형태의 비정형 데이터의 분석에도 자주 활용되고 있다.

이번 포스팅에는 텍스트 데이터인 imdb 데이터를 CNN 구조를 적용한 네트워크를 활용해 분석을 해보자.

## 문장 분류를 위한 합성곱 신경망(CNN for Sentence Classification)

[Kim 2014](http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf) 논문 이후로 NLP 과업을 위해 CNN이 자주 활용되고 있으며, 최근에는 [순환형 신경망(RNN)이 과도한 연산량 등으로 인해 비판을 받으면서 CNN이 더욱 떠오르고 있다](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)

Kim 2014가 제안한 모델의 대략적인 구조는 아래와 같다

<p align = "center"><br>
<img src ="/data/images/2018-05-16/1.png" width = "600px"/>
</p>

- 학습 데이터는 각 문장 내에 있는 단어의 embedding vector를 구해 만들어진다(행렬에서 하나의 단어가 하나의 행을 차지)
- 합성곱 연산은 단어 단위(행 단위)로 이루어진다
- 결과로는 각 문장을 긍정(1) 혹은 부정(0) 으로 분류한다.


<p align = "center"><br>
<img src ="/data/images/2018-05-16/2.png" width = "600px"/>
</p>

### IMDB 데이터 셋 불러오기

IMDB 영화 리뷰 감성분류 데이터 셋(IMDB Movie Revies Sentiment Classification Dataset)은 총 50,000 개의 긍정/혹은 부정으로 레이블링된 데이터 인스턴스(즉, 50,000개의 영화 리뷰)로 이루어져 있으며, 케라스 패키지 내에 processing이 다 끝난 형태로 포함되어 있다.

리뷰 데이터를 불러올 때 주요 파라미터는 아래와 같다

- ```num_features```: 빈도 순으로 상위 몇 개의 단어를 포함시킬 것인가를 결정
- ```sequence_length```: 각 문장의 최대 길이(특정 문장이 ```sequence_length```보다 길면 자르고, ```sequence_length```보다 짧으면 빈 부분을 0으로 채운다)를 결정
- ```embedding_dimension```: 각 단어를 표현하는 벡터 공간의 크기(즉, 각 단어의 vector representation의 dimensionality)


```python
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

num_features = 3000
sequence_length = 300
embedding_dimension = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_features)

X_train = pad_sequences(X_train, maxlen = sequence_length)
X_test = pad_sequences(X_test, maxlen = sequence_length)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

```python
(25000, 300), (25000, 300), (25000,), (25000,)
```

### 모델 생성하기

가장 기본적인 형태의 1-D convolution (temporal convolution)을 구현해 보자.
이는 2차원 형태(2-D)의 데이터에 적용 가능하다.


```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Embedding, Flatten
from keras import optimizers
def imdb_cnn():
    model = Sequential()

    # use Embedding layer to create vector representation of each word => it is fine-tuned every iteration
    model.add(Embedding(input_dim = 3000, output_dim = embedding_dimension, input_length = sequence_length))
    model.add(Conv1D(filters = 50, kernel_size = 5, strides = 1, padding = 'valid'))
    model.add(MaxPooling1D(2, padding = 'valid'))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr = 0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])

    return model

model = imdb_cnn()
```

보통 위와 같이 ```Conv1D``` 레이어에 ```MaxPooling1D``` 레이어를 이어 사용한다.
모델 학습은 이미지 데이터와 똑같다.


```python
history = model.fit(X_train, y_train, batch_size = 50, epochs = 100, validation_split = 0.2, verbose = 0)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()
```

모델의 학습은 잘 이루어져 학습 오차는 아래와 같이 0으로 수렴하는 것을 볼 수 있다. 하지만 validation error는 0.85 내외를 왔다갔다 하다가 60 에포크 이상에서 갑자기 떨어져 과적합(overfitting)의 신호를 보인다.

<p align = "center"><br>
<img src ="/data/images/2018-05-16/3.png" width = "600px"/>
</p>

검증 정확도로 모델을 검증해 보자.

```python
results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])
```

```python
Test accuracy:  0.8556
```

검증 정확도 85%로 간단한 모델을 활용하고 오래 학습시키지 않았음에도 불구하고 나쁘지 않은 결과를 보여준다. 이제 다음 포스팅에서는 이번 포스팅에서 적용해본 바닐라 모델에서 나아가, 조금 더 복잡한 CNN 모델을 NLP 과업에 적용해 보자.


# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/2.%20CNN/3-Advanced-CNN-2/3-advanced-cnn-2.ipynb)에서 열람하실 수 있습니다!
