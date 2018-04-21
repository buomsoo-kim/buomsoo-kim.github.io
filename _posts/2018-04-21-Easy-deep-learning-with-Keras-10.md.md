---
layout: post
title: 케라스 기초 - 다층 퍼셉트론 2 (Classification with MLP)
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 다층 퍼셉트론 1 (Regression with MLP)

Objective: 케라스로 다층 퍼셉트론 모델을 만들고, 이를 분류(classification) 문제에 적용해 본다

## 다층 퍼셉트론이란?

가장 기본적인 형태의 인공신경망(Artificial Neural Networks) 구조이며, 하나의 입력층(input layer), 하나 이상의 은닉층(hidden layer), 그리고 하나의 출력층(output layer)로 구성된다.

<p align = "center">
<img src ="/data/images/2018-04-21/1.jpeg" width = "800px"/>
</p>

[지난 포스트](https://buomsoo-kim.github.io/keras/2018/04/21/Easy-deep-learning-with-Keras-1.md/)에서 MLP를 회귀 과업에 적용하는 방법에 대해 익혔다. 이번 포스팅에서는 분류 과업을 위해 MLP를 활용해 보자!

## 다층 퍼셉트론의 분류 과업 적용

- 분류 과업(classification task)은 머신러닝에서 예측하고자 하는 변수(y)가 카테고리 속성을 가질 때(categorical)를 일컫는다.
- 이미지 분류(image classification), 이탈/잔존(churn/retention) 예측 등

<p align = "center">
<img src ="https://image.slidesharecdn.com/mllightningtalk-160204050203/95/machine-learning-in-5-minutes-classification-7-638.jpg?cb=1454562230" width = "700px"/>
</p>

- 손실 함수(loss function)를 위해서는 cross-entropy (혹은 softmax) loss가 흔히 사용되며 평가 지표(evaluation metric)로는 정확도(accuracy)가 가장 널리 사용된다.

<p align = "center">
<img src ="http://cs231n.github.io/assets/svmvssoftmax.png" width = "500px"/> <br> </p>


### Breast cancer 데이터 셋 가져오기

- 총 569개의 데이터 인스턴스(양성 357개, 악성 212개)를 포함
- 30개의 피쳐(feature)를 통해 각 데이터 인스턴스가 양성(benign)인지 악성(malign)인지를 "분류(classify)"한다.
- documentation: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()

X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 7) # 학습 데이터(0.7)와 검증 데이터(0.3)로  전체 데이터 셋을 나눈다

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

```python
(398, 30), (171, 30), (398,), (171,)
```

### 모델 만들기

- 회귀 과업 때와 동일하다. Sequential Model API를 통해 레이어가 하나하나 순서대로 추가되는 모델을 만들 수 있다
- ```Sequential()```로 모델을 생성하며, 생성한 직후에는 아무런 레이어가 추가되지 않은 '빈 모델'이다. (```add()```함수로 레이어를 추가해야 함)
- documentation: https://keras.io/getting-started/sequential-model-guide/

```python
from keras.models import Sequential
model = Sequential()      # 현재 이 모델은 레이어가 하나도 추가되어 있지 않음
```

### 레이어 추가하기

- 생성된 모델에 레이어를 하나하나 추가한다.
- ```add()``` 함수를 활용하여 레고 블럭을 쌓듯이 하나하나 추가해 나간다.
- 회귀 과업 때와는 다르게 분류 문제에서는 마지막 레이어에 sigmoid가 추가되는 것에 유의한다(결과값을 [0, 1] 확률로 변환하기 위함)

```python
from keras.layers import Activation, Dense
model.add(Dense(10, input_shape = (30,)))    # 입력층 => input_shape이 명시되어야 함
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # 은닉층 1
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # 은닉층 2
model.add(Activation('sigmoid'))
model.add(Dense(1))                          # 출력층 => output dimension == 1 (regression problem)
model.add(Activation('sigmoid'))             
```

혹은 레이어 추가를 아래와 같이 더 간단하게 실행할 수 있다(결과는 위와 같음).

```python
model.add(Dense(10, input_shape = (30,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))
```

### 모델 컴파일

- 케라스 모델은 학습 이전에 컴파일되어야 하며, 이 과정에서 손실 함수(loss function)와 최적화 방법(optimizer)가 구체화외더야 한다.
- documentation (optmizers): https://keras.io/optimizers/
- documentation (losses): https://keras.io/losses/

```python
from keras import optimizers
sgd = optimizers.SGD(lr = 0.01)    # stochastic gradient descent optimizer
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['acc'])   
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
dense_13 (Dense)             (None, 10)                310       
_________________________________________________________________
activation_7 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_14 (Dense)             (None, 10)                110       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_15 (Dense)             (None, 10)                110       
_________________________________________________________________
activation_9 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_16 (Dense)             (None, 1)                 11        
_________________________________________________________________
activation_10 (Activation)   (None, 1)                 0         
=================================================================
Total params: 541
Trainable params: 541
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
   - 결과는 리스트([손실, 정확도])로 반환한다.
   - 분류 문제에서는 정확도(accuracy)로 평가하는 것에 유의한다.

```python
results = model.evaluate(X_test, y_test)

print(model.metrics_names)     # 모델의 평가 지표 이름
print(results)                 # 모델 평가 지표의 결과값

print('loss: ', results[0])
print('accuracy: ', results[1])
```

```python
['loss', 'acc']
[0.63870607063784235, 0.67836257240228481]
loss:  0.638706070638
accuracy:  0.678362572402
```

분류 문제의 경우 검증 데이터 인스턴스 중에 몇 개를 맞추었는가(정확도)로 모델을 평가하다 보니, 회귀 문제에 비해 훨씬 직관적인 평가 방법이라고 할 수 있다. 위의 모델의 경우 클래스가 2개(0/1)인 이진 분류 문제(binary classification problem)인데 검증 정확도가 67.8%이므로 찍는 경우(예상 정확도 0.5)에 비해서는 높은 정확도를 보이지만 그리 높은 정확도는 아니라고 할 수 있다.

이제 다음 세션에서 어떻게 모델의 학습 과정과 성능을 개선시킬 수 있는지에 대해서 알아본다.

# 전체 코드

본 실습의 전체 코드는 [여기](Easy-deep-learning-with-Keras/1. MLP/1-Basics-of-MLP/1-Basics-of-MLP.ipynb)에서 열람하실 수 있습니다!
