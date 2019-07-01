---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (18) - 순환형 신경망(RNN) 모델 만들기
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 순환형 신경망 4 - 순환형 신경망 모델 만들기

Objective: 케라스로 RNN 모델을 구현해 본다

[지난 포스팅](https://buomsoo-kim.github.io/keras/2019/06/26/Easy-deep-learning-with-Keras-17.md/)까지 RNN 셀에 대해서 알아보고 이를 케라스로 구현하는 방법에 대해 알아보았다. 이번 포스팅에서는 실제로 케라스를 활용하여 RNN 모델을 어떻게 구현하는지에 대해 알아보자.

### RNN

MLP나 CNN과 같은 Feedforward neural network는 universal approximator로 다양한 종류의 형태의 데이터에 우수한 성능을 자랑하지만, 순차형(sequential) 데이터를 학습하기 위해 최적화된 형태는 아니라고 할 수 있다. 다른 의미로는, CNN이나 MLP는 지난 입력값의 기억(memory)을 보존하고 있지 않다. 예를 들어, 말뭉치를 번역한다고 할 때, 문맥(context)을 충분히 고려하지 않을 수 있다는 단점이 있다.

이에 반면, RNN은 순환형 구조로 인해 지난 입력값에 대한 기억(memory)을 가지고 있다는 특성으로 인해 순차형 데이터를 처리하기 위해 최적화된 형태라고 할 수 있다.

<p align = "center"><br>
<img src ="http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg" width = "600px"/>
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

### 기본 RNN 모델(Vanilla RNN)

SimpleRNN 셀을 갖는 가장 기본적인 RNN 모델을 구현해 본다. 기본 RNN은 지난 포스팅에서 봤듯이 간단한 구조를 가지고 있어 long-term dependency를 효율적으로 처리할 수 없는 단점을 갖는다.

<p align = "center"><br>
<img src ="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png" width = "400px"/>
</p>

```python
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

# 기본 RNN 모델을 구현하기 위한 함수
def vanilla_rnn():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape = (49,1), return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model
```

KerasClassifier 함수로 RNN 모델을 생성한다(KerasClassifier 함수를 사용하지 않아도 만들 수 있다).

```python
model = KerasClassifier(build_fn = vanilla_rnn, epochs = 200, batch_size = 50, verbose = 1)
```

모델 학습 및 검증

```python
model = KerasClassifier(build_fn = vanilla_rnn, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))
```

```python
0.74958263773
```

이번 포스팅에서는 기본적인 RNN 모델을 생성하는 방법에 대해 알아보았다. 다음 포스팅에서는 조금 더 복잡한 구조인 LSTM 모델 구현에 대해 알아보자.

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/1-Basic-RNN/1-basic-rnn.ipynb)에서 열람하실 수 있습니다!
