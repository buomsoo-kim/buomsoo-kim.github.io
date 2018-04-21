---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (1) - 다층 퍼셉트론 1 (Regression with MLP)
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 다층 퍼셉트론 1 (Regression with MLP)

Objective: 케라스로 다층 퍼셉트론 모델을 만들고, 이를 회귀(regression) 문제에 적용해 본다

## 다층 퍼셉트론이란?

가장 기본적인 형태의 인공신경망(Artificial Neural Networks) 구조이며, 하나의 입력층(input layer), 하나 이상의 은닉층(hidden layer), 그리고 하나의 출력층(output layer)로 구성된다.

물론, 각 층에서 뉴런(neuron)의 개수에는 제약이 없다.

<p align = "center">
<img src ="/data/images/2018-04-21/1.jpeg" width = "800px"/>
은닉층이 하나인 다층 퍼셉트론 구조
</p>

위의 MLP 네트워크에서 뉴런의 개수는 다음과 같다
- 입력층의 뉴런 개수: 3
- 은닉층의 뉴런 개수: 4
- 출력층의 뉴런 개수: 2

<p align = "center">
<img src ="/data/images/2018-04-21/2.jpeg" width = "800px"/>
은닉층이 두 개인 다층 퍼셉트론 구조
</p>

위의 MLP 네트워크에서 뉴런의 개수는 다음과 같다
- 입력층의 뉴런 개수: 3
- 첫 번째 은닉층의 뉴런 개수: 4
- 두 번째 은닉층의 뉴런 개수: 4
- 출력층의 뉴런 개수: 1

## 다층 퍼셉트론의 회귀 과업 적용

- 회귀 과업(regression task)은 머신러닝에서 예측하고자 하는 변수(y)가 실수 값을 가질 때(continuous)를 일컫는다.
- 사람의 키(height), 지능(IQ), 연봉(salary) 등을 예측하는 과업을 예로 들 수 있다.

<p align = "center">
<img src ="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png" width = "700px"/>
</p>

- 손실 함수(loss function)과 평가 지표(evaluation metric)을 위해서는 예측치와 실제 값을 뺀 후에 제곱하여 평균한 평균 제곱 오차(MSE; Mean Squared Error)가 흔히 활용된다.

<p align = "center">
<img src ="https://i.stack.imgur.com/iSWyZ.png" width = "400px"/>
</p>

<br>
### Boston housing 데이터 셋 가져오기

- 총 506개의 데이터 인스턴스(학습 데이터 404개, 검증 데이터 102개)를 포함
- 13개의 피쳐(feature)를 통해 특정 위치에 있는 집들의 중앙값("the median values of the houses at a location")을 예측
- [documentation](https://keras.io/datasets/)

```python
from keras.datasets import boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

```python
(404, 13), (102, 13), (404,), (102,)
```

### 모델 만들기

- Sequential Model API를 통해 레이어가 하나하나 순서대로 추가되는 모델을 만들 수 있다
- ```Sequential()```로 모델을 생성하며, 생성한 직후에는 아무런 레이어가 추가되지 않은 '빈 모델'이다. (```add()```함수로 레이어를 추가해야 함)
- [documentation](https://keras.io/getting-started/sequential-model-guide/)

```python
from keras.models import Sequential
model = Sequential()      # 현재 이 모델은 레이어가 하나도 추가되어 있지 않음
```

### 레이어 추가하기

- 생성된 모델에 레이어를 하나하나 추가한다.
- ```add()``` 함수를 활용하여 레고 블럭을 쌓듯이 하나하나 추가해 나간다.

```python
from keras.layers import Activation, Dense
model.add(Dense(10, input_shape = (13,)))    # 입력층 => input_shape이 명시되어야 함
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # 은닉층 1
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # 은닉층 2
model.add(Activation('sigmoid'))
model.add(Dense(1))                          # 출력층 => output dimension == 1 (regression problem)
```

혹은 레이어 추가를 아래와 같이 더 간단하게 실행할 수 있다(결과는 위와 같음).

```python
model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1))
```

### 모델 컴파일

- 케라스 모델은 학습 이전에 컴파일되어야 하며, 이 과정에서 손실 함수(loss function)와 최적화 방법(optimizer)가 구체화외더야 한다.
- [documentation (optmizers)](https://keras.io/optimizers/)
- [documentation (losses)](https://keras.io/losses/)

```python
from keras import optimizers
sgd = optimizers.SGD(lr = 0.01)    # stochastic gradient descent optimizer
model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])   
```

### 모델 서머리

- ```summary()``` 함수로 자신이 생성한 모델의 레이어, 출력 모양, 파라미터 개수 등을 체크할 수 있다.

```python
model.summary()
```

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_9 (Dense)              (None, 10)                140       
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_10 (Dense)             (None, 10)                110       
_________________________________________________________________
activation_5 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                110       
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 11        
=================================================================
Total params: 371
Trainable params: 371
Non-trainable params: 0
_________________________________________________________________
```

### 모델 학습

- ```fit()``` 함수를 통해 학습 데이터와 기타 파라미터를 명시하고 모델 학습을 진행할 수 있다.
  - ```batch_size```: 한 번에 몇 개의 데이터를 학습할 것인가
  - ```epochs```: 모델 학습 횟수
  - ```verbose```: 모델 학습 과정을 표시할 것인가(0인 경우 표시 안함, 1인 경우 표시함)

```python
model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)
```

### 모델 평가

- ```evaluate()``` 함수를 활용해 모델을 평가할 수 있다.
   - 파라미터로 학습 데이터(```X_train```)과 학습 레이블(```y_train```)을 넣어준다.
   - 결과는 리스트([손실, 오차])로 반환한다.

```python
results = model.evaluate(X_test, y_test)

print(model.metrics_names)     # 모델의 평가 지표 이름
print(results)                 # 모델 평가 지표의 결과값

print('loss: ', results[0])
print('mse: ', results[1])
```

```python
['loss', 'mean_squared_error']
[81.900110581341906, 81.900110581341906]
loss:  81.9001105813
mse:  81.9001105813
```

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/1.%20MLP/1-Basics-of-MLP/1-Basics-of-MLP.ipynb)에서 열람하실 수 있습니다!
