---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (21) - 순환형 신경망(RNN) 모델 만들기 4
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 순환형 신경망 7 - CNN-RNN 모델  

Objective: 케라스로 RNN 모델을 구현해 본다

이번 포스팅에서는 서로 다른 형태의 인공신경망 구조인 CNN과 RNN을 합성한 CNN-RNN 모델을 구현하고 학습해 보자.


### 데이터 셋 불러오기

CNN-RNN 모델을 학습하기 위한 CIFAR-10 데이터 셋을 불러온다.

- source: https://www.cs.toronto.edu/~kriz/cifar.html

<p align = "center"><br>
<img src ="https://image.slidesharecdn.com/pycon2015-150913033231-lva1-app6892/95/pycon-2015-48-638.jpg?cb=1442115225" width = "600px"/>
</p>

```python
import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

```python
(50000, 32, 32, 3)
(10000, 32, 32, 3)
(50000, 10)
(10000, 10)
```

### CNN-RNN

- Convolution과 pooling 연산을 순차적으로 수행한 후 그 결과를 RNN 구조로 이어 학습한다.
- 이미지 캡셔닝(이미지 설명)에 활용되는 모형과 비슷한 구조라고 할 수 있다

<p align = "center"><br>
<img src ="https://cdn-images-1.medium.com/max/1600/1*vzFwXFJOrg6WRGNsYYT6qg.png" width = "600px"/>
</p>


```python
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Input, Activation, Reshape, concatenate
from keras import optimizers

model = Sequential()

model.add(Conv2D(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Reshape(target_shape = (16*16, 50)))
model.add(LSTM(50, return_sequences = False))

model.add(Dense(10))
model.add(Activation('softmax'))
adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

print(model.summary())
```

```python
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_8 (Conv2D)            (None, 32, 32, 50)        1400      
_________________________________________________________________
activation_18 (Activation)   (None, 32, 32, 50)        0         
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 16, 16, 50)        0         
_________________________________________________________________
reshape_6 (Reshape)          (None, 256, 50)           0         
_________________________________________________________________
lstm_13 (LSTM)               (None, 50)                20200     
_________________________________________________________________
dense_18 (Dense)             (None, 10)                510       
_________________________________________________________________
activation_19 (Activation)   (None, 10)                0         
=================================================================
Total params: 22,110
Trainable params: 22,110
Non-trainable params: 0
_________________________________________________________________
None
```

모델 학습 및 검증

```python
%%time
history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, verbose = 0)
results = model.evaluate(X_test, y_test)
print('Test Accuracy: ', results[1])
```

```python
Test Accuracy:  0.5927
```

정확도 59%로 괄목할 만한 성능은 아니지만 모델 학습을 개선한다면 더 나은 결과를 기대해볼 수 있을 것이다. 하이퍼파라미터와 optimizer, 모델 구조 등을 조금씩 바꾸어 가며 학습 성능을 각자 개선해 보자.


### CNN-RNN 2

- CNN과 RNN 연산을 독립적으로 수행하고 그 결과를 합치는 다른 CNN-RNN 모델 구조를 구현해 보자.
- 시각 질의응답(visual question answering)에 쓰이는 모델과 비슷한 구조라고 할 수 있다.

<p align = "center"><br>
<img src ="https://camo.githubusercontent.com/828817c970da406d2d83dc9a5c03fb120231e2a2/687474703a2f2f692e696d6775722e636f6d2f56627149525a7a2e706e67" width = "600px"/>
</p>

```python
input_layer = Input(shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]))
conv_layer = Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same')(input_layer)
activation_layer = Activation('relu')(conv_layer)
pooling_layer = MaxPooling2D(pool_size = (2,2), padding = 'same')(activation_layer)
flatten = Flatten()(pooling_layer)
dense_layer_1 = Dense(100)(flatten)

reshape = Reshape(target_shape = (X_train.shape[1]*X_train.shape[2], X_train.shape[3]))(input_layer)
lstm_layer = LSTM(50, return_sequences = False)(reshape)
dense_layer_2 = Dense(100)(lstm_layer)
merged_layer = concatenate([dense_layer_1, dense_layer_2])
output_layer = Dense(10, activation = 'softmax')(merged_layer)

model = Model(inputs = input_layer, outputs = output_layer)

adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

print(model.summary())
```

```python
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_4 (InputLayer)             (None, 32, 32, 3)     0                                            
____________________________________________________________________________________________________
conv2d_6 (Conv2D)                (None, 32, 32, 50)    1400        input_4[0][0]                    
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 32, 32, 50)    0           conv2d_6[0][0]                   
____________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)   (None, 16, 16, 50)    0           activation_8[0][0]               
____________________________________________________________________________________________________
reshape_4 (Reshape)              (None, 1024, 3)       0           input_4[0][0]                    
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 12800)         0           max_pooling2d_5[0][0]            
____________________________________________________________________________________________________
lstm_4 (LSTM)                    (None, 50)            10800       reshape_4[0][0]                  
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 100)           1280100     flatten_2[0][0]                  
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 100)           5100        lstm_4[0][0]                     
____________________________________________________________________________________________________
concatenate_2 (Concatenate)      (None, 200)           0           dense_6[0][0]                    
                                                                   dense_7[0][0]                    
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 10)            2010        concatenate_2[0][0]              
====================================================================================================
Total params: 1,299,410
Trainable params: 1,299,410
Non-trainable params: 0
____________________________________________________________________________________________________
```

모델 학습 및 검증

```python
%%time
history = model.fit(X_train, y_train, epochs = 10, batch_size = 100, verbose = 0)
results = model.evaluate(X_test, y_test)
print('Test Accuracy: ', results[1])
```

```python
Test Accuracy:  0.10000001
```

0.1 정도의 정확도로 새로운 CNN-RNN 모형은 학습이 거의 이루어지지 않는 것을 볼 수 있다. 복잡한 모델일수록 하이퍼파라미터 등이 여러 가지 경우의 수가 있고 local optima에 빠질 가능성이 높아 학습이 어려운 경향을 보인다는 것을 다시 확인해볼 수 있다.

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/3-Advanced-RNN-2/3-advanced-rnn-2.ipynb)에서 열람하실 수 있습니다!
