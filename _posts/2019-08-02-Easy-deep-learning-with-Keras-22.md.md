---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (22) - 순환형 신경망(RNN) 모델 만들기 5
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 순환형 신경망 8 - CuDNNGRU & CuDNNLSTM

Objective: 케라스로 CuDNNLSTM과 CuDNNGRU 모델을 구현해 본다

이번 포스팅에서는 GPU를 활용하여 기존의 LSTM/GRU보다 더 빠르게 학습할 수 있는 CuDNNLSTM과 CuDNNGRU를 구현해 보자.


### 데이터 셋 불러오기

CNN-RNN 모델을 학습하기 위한 IMDB 데이터 셋을 불러온다.


```python
num_words = 30000
maxlen = 300

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_words)

# pad the sequences with zeros 
# padding parameter is set to 'post' => 0's are appended to end of sequences
X_train = pad_sequences(X_train, maxlen = maxlen, padding = 'post')
X_test = pad_sequences(X_test, maxlen = maxlen, padding = 'post')

X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

```python
(25000, 300, 1)
(25000, 300, 1)
(25000,)
(25000,)
```

### LSTM

CuDNN을 활용하지 않은 기존의 LSTM


```python
def lstm_model():
    model = Sequential()
    model.add(LSTM(50, input_shape = (300,1), return_sequences = True))
    model.add(LSTM(1, return_sequences = False))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

 model = lstm_model()

 %%time
model.fit(X_train, y_train, batch_size = 100, epochs = 10, verbose = 0)
```

```python
Wall time: 29min 40s
```

기존의 LSTM 모델은 epoch 10회를 학습하는 데 30분 가량 걸리는 것을 볼 수 있다.

### GRU

CuDNN을 활용하지 않은 기존의 GRU

```python
def gru_model():
    model = Sequential()
    model.add(GRU(50, input_shape = (300,1), return_sequences = True))
    model.add(GRU(1, return_sequences = False))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

model = gru_model()
%%time
model.fit(X_train, y_train, batch_size = 100, epochs = 10, verbose = 0)
```

```python
Wall time: 21min 46s
```

기존의 GRU 모델은 학습에 있어 20분 정도 걸리는 것을 볼 수 있다. GRU가 LSTM에 비해 셀 구조가 단순해 LSTM에 비해서 적은 학습 시간을 필요로 한다.

### CuDNN LSTM

CuDNN을 활용한 CuDNNLSTM

```python
def cudnn_lstm_model():
    model = Sequential()
    model.add(CuDNNLSTM(50, input_shape = (300,1), return_sequences = True))
    model.add(CuDNNLSTM(1, return_sequences = False))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

model = cudnn_lstm_model()

%%time
model.fit(X_train, y_train, batch_size = 100, epochs = 10, verbose = 0)
```

```python
Wall time: 2min 53s
```

CuDNN LSTM은 3분 이내의 학습 시간을 보이며 기존의 LSTM에 비해 10배 가량 빠른 학습 속도를 보여준다.

### CuDNN GRU

CuDNN을 활용한 GRU

```python
def cudnn_gru_model():
    model = Sequential()
    model.add(CuDNNGRU(50, input_shape = (300,1), return_sequences = True))
    model.add(CuDNNGRU(1, return_sequences = False))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

model = cudnn_gru_model()

%%time
model.fit(X_train, y_train, batch_size = 100, epochs = 10, verbose = 0)
```

```python
Wall time: 1min 54s
```

CuDNN GRU도 역시 기존의 GRU에 비해 10배 가량 빠른 학습 속도를 보여준다. 


# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/4-Advanced-RNN-3/4-advanced-rnn-3.ipynb)에서 열람하실 수 있습니다!
