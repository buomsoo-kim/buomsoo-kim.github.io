---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (8) - CNN 구조 이해하기 2
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 합성곱 신경망 2 - CNN 구조 이해하기 두번째

Objective: 이미지 및 영상 분석, 자연어처리 등에 폭넓게 쓰이는 합성곱 신경망의 구조에 대해 알아본다.


[지난 포스팅](https://buomsoo-kim.github.io/keras/2018/04/26/Easy-deep-learning-with-Keras-7.md/)에서 CNN 구조를 이해하기에 앞서, 컴퓨터가 이미지 데이터를 어떻게 받아들이고 이미지가 텐서로 어떻게 표현되는지에 대해서 알아보았다.

이번 포스팅에서는 현대 CNN을 이루는 핵심적인 구조인 합성곱 레이어(convolution layer)와 풀링 레이어(pooling layer)의 연산 과정에 대해서 알아보자.

언급했듯이, CNN은 아래와 같이 크게 합성곱 레이어(CONV), 풀링 레이어(POOL), 그리고 완전 연결된 레이어(FC)로 이루어져 있다.

<p align = "center"><br>
<img src ="http://cs231n.github.io/assets/cnn/convnet.jpeg" width = "600px"/>
</p>

이름에서 유추할 수 있듯이 합성곱 레이어에서 convolution 연산이 이루어지고, 풀링 레이어에서 pooling 연산이 이루어진다. 그리고 패딩(padding)과 필터(filter), 혹은 커널(kernel)이라는 개념이 중요하게 등장한다. 자세한 설명은 [여기](http://cs231n.github.io/convolutional-networks/)를 참고하자.

케라스에서는 convolution, pooling 연산을 함수로 제공해 따로 구현할 필요가 없으며, 패딩과 필터도 인자값을 설정함으로써 쉽게 조절할 수 있다. 그래도 기본적인 연산 과정을 이해해 두는것이 좋으니 하나하나 알아보자.

## 패딩(Padding)

앞서 언급하였듯이, convolution과 pooling 연산은 파라미터의 수를 줄여나가는 과정이다. 하지만 이러한 과정에서 지나치게 데이터가 축소되어 정보가 소실되는 것을 방지하기 위해 데이터에 0으로 이루어진 패딩을 주는 경우가 있다.

케라스에서 padding을 설정하는 방법은 아래와 같이 두 가지가 있다.

- 합성곱 혹은 풀링 연산을 수행하는 레이어에 파라미터로 설정. 이 경우 아래와 같은 두 가지 옵션 중 하나를 선택할 수 있다.

1. ```valid```: 패딩을 하지 않음(사이즈가 맞지 않을 경우 가장 우측의 열 혹은 가장 아래의 행을 드랍한다).

```python
# when padding = 'valid'
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
print(model.output_shape)
```

패딩을 주지 않으니(```valid```) 아래와 같이 (10 X 10)에서 (8 X 8)로 차원이 축소된 것을 볼 수 있다.

```python
(None, 8, 8, 10)
```

2. ```same```: 필터의 사이즈가 ```k```이면 사방으로 ```k/2``` 만큼의 패딩을 준다.

```python
# when padding = 'same'
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
print(model.output_shape)
```

패딩을 설정을 ```same```으로 주니 아래와 같이 차원이 (10 X 10)으로 유지되는 것을 확인할 수 있다.

```python
(None, 10, 10, 10)
```

- 패딩 레이어(```ZeroPadding1D, ZeroPadding2D, ZeroPadding3D``` 등)를 따로 생성해 모델에 포함시키는 방법. 이 방법은 모델을 생성할 때 Sequential API가 아닌 Functional API를 활용할 때 유용하다(Functional API는 아직 다루지 않았는데, 궁금하신 분은 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/5.%20ETC/0-Creating-models/0-2-functional-API.ipynb)를 참고하면 된다)

```python

# user-customized padding# user-c
input_layer = Input(shape = (10, 10, 3))
padding_layer = ZeroPadding2D(padding = (1,1))(input_layer)

model = Model(inputs = input_layer, outputs = padding_layer)
print(model.output_shape)
```

(10 X 10) 크기의 데이터에 (1, 1)의 패딩을 해주었더니 사방으로 1만큼 확장되어 아래와 같이 (12 X 12)의 output shape이 출력되는 것을 볼 수 있다.

```python
(None, 12, 12, 3)
```

## 필터/커널(Filter/kernel)

합성곱 레이어를 보면 ```padding```외에도 크게 세 개의 중요한 파라미터가 등장한다. 바로 ```filters```, ```kernel_size```, ```strides```이다. 이 세 개의 파라미터가 합성곱 레이어에의 출력 모양을 결정한다고 할 수 있다.

<p align = "center"><br>
<img src ="data/images/2018-05-02/1.JPG" width = "500px"/>
</p>

- ```filters```: 몇 개의 다른 종류의 필터를 활용할 것인지를 나타냄. 출력 모양의 **깊이(depth)** 를 결정한다.
- ```kernel_size```: 연산을 수행할 때 윈도우의 크기를 의미한다.
- ```strides```: 연산을 수행할 때 윈도우가 가로 그리고 세로로 움직이면서 내적 연산을 수행하는데, 한 번에 얼마나 움직일지를 의미한다.


```python
# when filter size = 10
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
print(model.output_shape)
```

필터 수가 10이므로 아래와 같이 출력의 depth도 10이다.

```python
(None, 10, 10, 10)
```

```python
# when filter size = 10
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 20, kernel_size = (3,3), strides = (1,1), padding = 'same'))
print(model.output_shape)
```

필터 수를 늘리면 아래와 같이 출력의 depth도 늘어난다.

```python
(None, 10, 10, 20)
```

## 풀링(pooling)

일반적으로 윈도우 내에서 출력의 최대값을 추출하는 맥스 풀링(max pooling)이 활용되나, 평균값을 뽑는 애버리지 풀링(average pooling)이 활용되기도 한다.

```pool_size``` 는 합성곱 레이어에서 ```kernel_size```와 같이 윈도우의 크기를 의미하며, 패딩은 합성곱 레이어와 똑같이 적용되며(```valid```혹은 ```same```), ```strides```가 미리 설정되지 않을 경우 ```pool_size```와 동일하게 설정된다.

```python
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
print(model.output_shape)

# when 'strides' parameter is not defined, strides are equal to 'pool_size'
model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))
print(model.output_shape)
```

(10 X 10) 크기의 출력에 (2 X 2) 크기의 풀링의 (2, 2)만큼의 stride로 적용하니 출력 모양이 (5 X 5)로 결정된다.

```python
(None, 10, 10, 10)
(None, 5, 5, 10)
```

stride를 명시적으로 설정해줄 경우(1,1)

```python
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (1,1), padding = 'valid'))
print(model.output_shape)
```

```python
(None, 9, 9, 10)
```

애버리지 풀링도 출력 모양은 똑같다(값은 다르다는 점에 유의한다).

```python
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(AveragePooling2D(pool_size = (2,2), padding = 'valid'))
print(model.output_shape)
```

```python
(None, 5, 5, 10)
```

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/2.%20CNN/1-Basic-CNN/0-understanding-cnn-architecture.ipynb)에서 열람하실 수 있습니다!
