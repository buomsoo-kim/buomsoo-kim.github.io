---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (20) - 순환형 신경망(RNN) 모델 만들기 3
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 순환형 신경망 6 - 순환형 신경망 모델 만들기

Objective: 케라스로 RNN 모델을 구현해 본다

이번 포스팅에서는 조금 더 다양한 RNN 모델을 케라스로 구현하고 학습해 보자.


### 데이터 셋 불러오기

RNN을 학습하기 위한 reuters 데이터 셋을 불러온다.

```python
import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# parameters for data load
num_words = 30000
maxlen = 50
test_split = 0.3

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = num_words, maxlen = maxlen, test_split = test_split)

# pad the sequences with zeros 
# padding parameter is set to 'post' => 0's are appended to end of sequences
X_train = pad_sequences(X_train, padding = 'post')
X_test = pad_sequences(X_test, padding = 'post')

X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

y_data = np.concatenate((y_train, y_test))
y_data = to_categorical(y_data)

y_train = y_data[:1395]
y_test = y_data[1395:]

# 데이터의 모양 출력하기
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

```python
(1395, 49, 1)
(599, 49, 1)
(1395, 46)
(599, 46)
```

### 심층 RNN 모델(Deep RNN)

여러 개의 LSTM 셀로 이루어진 layer를 여러 층 쌓아 심층 RNN 모델을 구현해 보자


```python
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

# 기본 RNN 모델을 구현하기 위한 함수
def deep_lstm():
    model = Sequential()
    model.add(LSTM(20, input_shape = (49,1), return_sequences = True))
    model.add(LSTM(20, return_sequences = True))
    model.add(LSTM(20, return_sequences = True))
    model.add(LSTM(20, return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model
```

KerasClassifier 함수로 RNN 모델을 생성한다(KerasClassifier 함수를 사용하지 않아도 만들 수 있다).

```python
model = KerasClassifier(build_fn = deep_lstm, epochs = 200, batch_size = 50, verbose = 1)
```

모델 학습 및 검증

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))
```

```python
0.816360601002
```

LSTM 레이어를 더 추가했지만 정확도는 기존 심층 LSTM에 비해 떨어지는 것을 볼 수 있다. 그렇다면 레이어를 추가하는 것 말고 RNN을 개선하는 어떠한 방법이 있을 지 알아보자.


### 양방향 RNN (Bidirectional RNN)

- 기존의 RNN은 한 방향(앞 -> 뒤)으로의 순서만 고려하였다고 볼 수도 있다. 반면, 양방향 RNN은 역방향으로의 순서도 고려하는 모델이다.
- 실제로는 두 개의 RNN 모델(순방향, 역방향)을 만들어 학습 결과를 합치는 인공신경망 모델에 가깝다.

<p align = "center"><br>
<img src ="http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/bidirectional-rnn.png" width = "600px"/>
</p>

```python
# 양방향 RNN을 구현하기 위한 함수
from keras.layers import Bidirectional

def bidirectional_lstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences = False), input_shape = (49,1)))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model
```

```python
model = KerasClassifier(build_fn = bidirectional_lstm, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))
```

```python
0.841402337229
```

더 적은 숫자의 레이어를 활용했음에도 불구하고 양방향 RNN이 심층 RNN에 비해 더 높은 성능을 자랑하는 것을 볼 수 있다.

### 심층 양방향 RNN (Deep Bidirectional RNN)

- 양방향 RNN 모델도 다른 RNN 모델과 같이 심층적으로 레이어를 쌓아 올릴 수 있다.

<p align = "center"><br>
<img src = "http://www.wildml.com/wp-content/uploads/2015/09/Screen-Shot-2015-09-16-at-2.21.51-PM-272x300.png" width = "600px"/>
</p>

```python
# 다층 양방향 LSTM을 구현하기 위한 함수
def deep_bidirectional_lstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(10, return_sequences = True), input_shape = (49,1)))
    model.add(Bidirectional(LSTM(10, return_sequences = True)))
    model.add(Bidirectional(LSTM(10, return_sequences = True)))
    model.add(Bidirectional(LSTM(10, return_sequences = False)))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model
```

```python
model = KerasClassifier(build_fn = deep_bidirectional_lstm, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))
```

```python
0.824707846411
```

이번에도 레이어를 여러개 추가하자 오히려 검증 결과가 떨어지는 것을 볼 수 있다. 복잡한 모델 선택으로 인해 학습이 제대로 되지 않았을 수도 있고, 학습 데이터가 과적합된 케이스일 수도 있다. 이렇듯 복잡한 구조를 갖는 인공신경망 모델을 학습시킬 때에는 여러 가지 경우의 수를 고려해 보아야 한다.

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/2-Advanced-RNN/2-advanced-rnn.ipynb)에서 열람하실 수 있습니다!
