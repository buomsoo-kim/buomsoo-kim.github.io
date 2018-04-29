---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (8) - 간단한 합성곱 신경망(CNN) 모델 만들기
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 합성곱 신경망 2 - 간단한 합성곱 신경망(CNN) 모델 만들기 (Basic CNN)

Objective: 케라스로 간단한 합성곱 신경망 모델을 만들어 본다.

[지난 포스팅](https://buomsoo-kim.github.io/keras/2018/04/25/Easy-deep-learning-with-Keras-6.md/)에서 뉴럴 네트워크의 학습 과정을 개선하기 위한 7가지 방법을 모두 적용한 MLP 모델을 생성해 보았다. 그 7가지 방법은 아래와 같다.

- 가중치 초기화(Weight Initialization)
- 활성함수(Activation Function)
- 최적화(Optimization)
- 배치 정규화(Batch Normalization)
- 드랍아웃(Dropout)
- 앙상블(Model Ensemble)
- 학습 데이터 추가(More training data)

최종적으로 개선된 MLP 모델은 MNIST 데이터셋에서 98%가 넘는 정확도를 보여주었다.

이번 포스팅부터는 이미지 데이터를 인식하는데 흔히 쓰이는 합성곱 신경망(CNN) 모델에 대해서 알아보자.

## 합성곱 신경망

CNN은 MLP에 합성곱 레이어(convolution layer)와 풀링 레이어(pooling layer)라는 고유의 구조를 더한 뉴럴 네트워크라고 할 수 있다.

- 합성곱 레이어: 필터(filter), 혹은 커널(kernel)이라고 하는 작은 수용 영역(receptive field)을 통해 데이터를 인식한다.
- 풀링 레이어: 특정 영역에서 최대값만 추출하거나, 평균값을 추출하여 차원을 축소하는 역할을 한다.

<p align = "center"><br>
<img src ="/data/images/2018-04-26/cnn.jpeg" width = "600px"/>
</p>

CNN은 MLP나 뒤에서 나올 순환형 신경망(RNN)에 비해 학습해야 할 파라미터의 개수가 상대적으로 적어 학습이 빠르다는 장점이 있다.

2013년에 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)이 제안되어 ImageNet 대회에서 획기적인 성적을 낸 이후로 CNN에 대한 연구가 활발히 되어 이제는 이미지 인식 뿐 아니라 [자연어처리](http://www.aclweb.org/anthology/D14-1181)에도 흔히 쓰이며, [CNN의 학습 과정을 해석]((https://arxiv.org/abs/1311.2901))하고 [시각화](https://distill.pub/2017/feature-visualization/)하려는 시도도 자주 등장하고 있다.

p align = "center"><br>
<img src ="/data/images/2018-04-26/convnet.jpeg" width = "600px"/>
</p>

### Digits 데이터 셋 불러오기

이번에는 MNIST와 비슷한 형태지만 데이터셋의 크기가 작은 ```scikit-learn```의 digits 데이터 셋을 활용해 보자.

- [Doc](http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html)

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
```

```python
data = datasets.load_digits()
plt.imshow(data.images[0])    # show first number in the dataset
plt.show()
print('label: ', data.target[0])    
```

<p align = "center"><br>
<img src ="/data/images/2018-04-26/1.png" width = "400px"/>
</p>

```python
label:  0   # 첫 번째 데이터 인스턴스의 라벨(클래스)는 0이다
```

Digits 데이터 인스턴스의 개수는 총 1797 개이며, 데이터의 모양은 8 X 8이다.

```python
# shape of data
print(X_data.shape)    # (8 X 8) format
print(y_data.shape)
```

```python
(1797, 8, 8)
(1797,)
```

### 데이터 셋 전처리

데이터의 모양을 바꾼다. X 데이터는 (3차원으로 바꿔) 차원을 하나 늘리고, Y 데이터는 one-hot 인코딩을 해준다.

```python
# reshape X_data into 3-D format
# note that this follows image format of Tensorflow backend
X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))

# one-hot encoding of y_data
y_data = to_categorical(y_data)
```

전체 데이터를 학습/검증 데이터 셋으로 나눈다

```python
# partition data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 777)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

```python
(1257, 8, 8, 1)
(540, 8, 8, 1)
(1257, 10)
(540, 10)
```

1257개의 학습 데이터를 가지고 모델을 학습시키고, 540개의 검증 데이터로 이를 평가해본다.

### 모델 생성하기

MLP 모델을 생성하는데 사용하였던 ```Sequential()```로 모델을 생성한다.

```python
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

model = Sequential()
```

### 합성곱 레이어

이미지 처리를 위해서는 일반적으로 2D convolution layer (```Conv2D```)를 사용한다. 사용자가 지정해주어야 하는 ```Conv2D```의 주요 파라미터는 아래와 같다.

- 필터의 사이즈(```kernel_size```): 합성곱 연산을 진행할 필터(커널)의 사이즈를 의미한다. 구체적으로, 수용 영역의 너비(width)와 높이(height)를 설정해 준다.
- 필터의 개수(```filters```): 서로 다른 합성곱 연산을 수행하는 필터의 개수를 의미한다. 필터의 개수는 다음 레이어의 깊이(depth)를 결정한다.
- 스텝 사이즈(```strides```): 필터가 이미지 위를 움직이며 합성곱 연산을 수행하는데, 한 번에 움직이는 정도(가로, 세로)를 의미한다.
- 패딩(```padding```): 이미지 크기가 작은 경우 이미지 주위에 0으로 이루어진 패딩을 추가해 차원을 유지할 수 있다.

<p align = "center"><br>
<img src ="/data/images/2018-04-26/1.jpg" width = "400px"/>
</p>


```python
model.add(Conv2D(input_shape = (X_data.shape[1], X_data.shape[2], X_data.shape[3]), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
```

### 활성함수

MLP와 동일하게 ReLU 활성함수를 사용한다

```python
model.add(Activation('relu'))
```

### 풀링 레이어

일반적으로 이미지 인식을 위해서는 맥스 풀링(max pooling), 혹은 애버리지 풀링(average pooling)이 사용되며 특정 영역을 묘사하는 대표값을 뽑아 파라미터의 수를 줄여주는 역할을 한다

<p align = "center"><br>
<img src ="/data/images/2018-04-26/2.jpg" width = "400px"/>
</p>

```python
model.add(MaxPooling2D(pool_size = (2,2)))
```

### 완전 연결 레이어(Dense 혹은 fully-connected layer)

CNN의 마지막 단에 MLP와 동일한 완전 연결 레이어를 넣을 수도 있고, 넣지 않을 수도 있다.

MLP로 연결하기 전에 3차원의 데이터의 차원을 줄이기 위해 ```Flatten()``` 을 추가해 주는것에 유의하자.

```python
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))     ### 이미지를 분류하기 위한 마지막 레이어
```

### 모델 컴파일 및 학습

모델을 컴파일하고 학습을 진행시킨다.

```python
adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 50, validation_split = 0.2, epochs = 100, verbose = 0)
```

모델 학습 과정을 시각화해본다. 빠르게 정확도가 올라가는 것으로 보아 학습이 잘 되는 것 같다.

```python
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()
```

<p align = "center"><br>
<img src ="/data/images/2018-04-26/2.png" width = "300px"/>
</p>

### 모델 평가

검증 데이터로 모델을 평가해본다

```python
results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])
```

```python
Test accuracy:  0.972222222222
```

최종 결과로 97%가 넘는 높은 정확도를 기록하였다. 가장 간단한 형태의 CNN을 구현했음에도 불구하고 상당히 정확히 숫자를 분류해내는 것을 알 수 있다. 이제 다음 포스팅부터는 CNN에 대해서 더 자세히 알아보자.

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/2.%20CNN/1-Basic-CNN/1-basic-cnn.ipynb)에서 열람하실 수 있습니다!
