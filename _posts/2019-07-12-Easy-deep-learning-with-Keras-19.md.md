---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (19) - 순환형 신경망(RNN) 모델 만들기 2
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 순환형 신경망 5 - 순환형 신경망 모델 만들기

Objective: 케라스로 RNN 모델을 구현해 본다

이번 포스팅에서는 조금 더 다양한 RNN 모델을 케라스로 구현하고 학습해 보자.

### 다중 RNN (Stacked RNN)

비슷한 MLP layer를 여러 층 쌓아 깊은 모델을 구현하듯이, RNN layer도 여러 겹 쌓아 다중 RNN을 구현할 수 있다.

<p align = "center"><br>
<img src ="https://lh6.googleusercontent.com/rC1DSgjlmobtRxMPFi14hkMdDqSkEkuOX7EW_QrLFSymjasIM95Za2Wf-VwSC1Tq1sjJlOPLJ92q7PTKJh2hjBoXQawM6MQC27east67GFDklTalljlt0cFLZnPMdhp8erzO" width = "600px"/>
</p>

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

### 다중 RNN 모델(Stacked Vanilla RNN)

SimpleRNN 셀로 이루어진 layer를 여러 층 쌓아 다중 RNN 모델을 구현해 보자


```python
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

# 기본 RNN 모델을 구현하기 위한 함수
def stacked_vanilla_rnn():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape = (49,1), return_sequences = True))   # return_sequences parameter has to be set True to stack
    model.add(SimpleRNN(50, return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model
```

KerasClassifier 함수로 RNN 모델을 생성한다(KerasClassifier 함수를 사용하지 않아도 만들 수 있다).

```python
model = KerasClassifier(build_fn = stacked_vanilla_rnn, epochs = 200, batch_size = 50, verbose = 1)
```

모델 학습 및 검증

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))
```

```python
0.746243739566
```

SimpleRNN 레이어를 여러 층 쌓아 다중 RNN을 구현해 보았지만, 정확도는 단층 RNN과 크게 달라지지 않는 것을 볼 수 있었다. 그렇다면 다른 RNN 셀을 활용한 RNN 모델을 구현하고 결과를 살펴보자.


### LSTM 모델

<p align = "center"><br>
<img src ="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" width = "600px"/>
</p>

```python
# 단층 LSTM을 구현하기 위한 함수
from keras.layers import LSTM

def lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape = (49,1), return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model
```

```python
model = KerasClassifier(build_fn = lstm, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))
```

```python
0.844741235392
```

기본 RNN 모델에 비해 LSTM 모델을 구현했을 때에는 정확도가 10% 가량 높아진 것을 볼 수 있다. 그렇다면 다중 LSTM모델의 결과는 어떠할지 한번 살펴보자.

```python
# 다층 LSTM을 구현하기 위한 함수
def stacked_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape = (49,1), return_sequences = True))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model
```

```python
model = KerasClassifier(build_fn = stacked_lstm, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))
```

```python
0.858096828047
```

다중 LSTM은 단층 LSTM에 비해 정확도가 다소 올라가는 것을 볼 수 있다. 또한 전반적으로 LSTM 모델이 기본 RNN 모델에 비해서 좋은 성능을 보이는 것을 확인해보았다.

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/1-Basic-RNN/1-basic-rnn.ipynb)에서 열람하실 수 있습니다!
