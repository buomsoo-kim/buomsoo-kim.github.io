---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (7) - CNN 구조 이해하기
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 합성곱 신경망 1 - CNN 구조 이해하기

Objective: 이미지 및 영상 분석, 자연어처리 등에 폭넓게 쓰이는 합성곱 신경망의 구조에 대해 알아본다.


[지난 포스팅](https://buomsoo-kim.github.io/keras/2018/04/26/Easy-deep-learning-with-Keras-6.md/)을 마지막으로, 케라스에서 MLP를 효율적으로 학습시키기 위한 방법에 대해서 알아보았다. 그 방법은 아래와 같다.

- 가중치 초기화(Weight Initialization)
- 활성함수(Activation Function)
- 최적화(Optimization)
- 배치 정규화(Batch Normalization)
- 드랍아웃(Dropout)
- 앙상블(Model Ensemble)
- 학습 데이터 추가(More training samples)

이 방법은 MLP뿐 아니라 다른 신경망 구조에도 비슷하게 적용될 수 있으며, 현업에 적용할 만한 인공신경망을 만드는데 꼭 고려해야 하는 요소이므로 잘 알아두어야 한다.

이번 포스팅부터는 이미지를 분석하는데 특화된 인공신경망 구조로 알려져 있지만, 사실은 이미지뿐 아니라 영상, 텍스트, 음성 등 다양한 데이터를 분석하는데 활용되는 합성곱 신경망(CNN; Convolutional Neural Networks)에 대해서 알아보자.

## 이미지 데이터에 대한 이해

MLP에서 MNIST 데이터를 분석하면서 간단히 알아보았지만, 이미지 데이터는 픽셀(pixel)을 단위로 이루어져 있으며 기본적으로 3차원이다. MNIST의 경우 2차원(28 X 28)이었는데 왜 3차원이냐고 묻는 사람이 있을 수도 있겠지만, MNIST는 흑백(greyscale)이기때문에 2차원처럼 보이는 것이지, 실제로는 3차원(28 X 28 X 1)이라고 할 수 있다. 그리고 일반적인 이미지는 RGB 세 개의 채널 강도를 가지고 있으며 a X b X c의 3차원 구조를 가지고 있다.

<p align = "center"><br>
<img src ="/data/images/2018-04-28/1.png" width = "600px"/>
</p>

a X b X c의 이미지 구조에서 일반적으로 a는 너비(width)를, b는 높이(height)를, c는 컬러 채널의 수(일반적으로 RGB 채널의 경우 c=3)을 결정한다. 그러므로 위와 같은 이미지 데이터는 4 X 4 X 3 의 차원 구조를 가지고 있다고 할 수 있다.

케라스의 ```processing.image``` 모듈은 이미지를 불러오고 처리할 수 있는 유용한 함수들을 제공한다. 이제 실제 아래와 같은 강아지 이미지를 불러와 이미지 데이터에 대한 이해를 도와보자.

<p align = "center"><br>
<img src ="/data/images/2018-04-28/dog.jpg" width = "400px"/>
</p>

image 모듈에서 제공하는 ```keras.processing.image.load_img``` 함수를 통해 외부 이미지를 Python 환경으로 불러올 수 있다. 인자로 불러올 이미지의 크기(```target_size```)를 픽셀 단위로 설정할 수 있다.

```python
img = image.load_img('dog.jpg', target_size = (100, 100))
img
```

<p align = "center"><br>
<img src ="/data/images/2018-04-28/dog.jpg" width = "100px"/>
</p>

불러온 이미지에 ```img_to_array``` 함수를 씌우면 이미지를 NumPy 배열로 변환해 준다. 이미지의 모양을 출력해보면 아래와 같다. 이미지의 크기를 (100 X 100)으로 설정하였고, 컬러 이미지이므로 컬러 채널의 크기가 3이 되어 (100 X 100 X 3)의 3차원 구조를 갖는 배열이 생성되었다.

```python
img = image.img_to_array(img)
print(img.shape)
```

```python
(100, 100, 3)
```

## 이미지 텐서

이미지 데이터는 3차원인데, 실제 케라스나 텐서플로에서 이미지 데이터를 다루기 위한 텐서(tensor)의 모양을 출력해보면 아래와 같이 4차원 구조를 가지고 있다. 이는 초심자들에게 굉장히 많은 혼란을 가져오곤 하는데, 이는 대용량 이미지를 학습할 때 일반적인 경사하강법(Gradient Descent)에 비해 빠르게 수렴하는 Mini-batch Stochastic Gradient Descent (Mini-batch SGD)를 흔히 사용하기 때문이다.

<p align = "center"><br>
<img src ="/data/images/2018-04-28/2.PNG" width = "500px"/>
Keras Conv2D 레이어의 입력과 출력
</p>

자세히 보면 4차원 텐서의 shape이 ```(samples, rows, cols, channels)``` (텐서플로 기준)과 같이 이루어져 있는 것을 볼 수 있다. 여기에서 ```rows```와 ```cols```는 위에서 본 이미지 데이터의 너비와 높이에 해당하고, ```channels```는 RGB 채널과 같다. 즉, 달라지는 것은 맨 처음에 붙는 ```samples```인데, 이는 Mini-batch SGD를 위해 전체 학습 데이터에서 랜덤샘플링을 통해 미니 배치를 만들때 그 미니 배치의 크기를 의미한다.

결론적으로, 한 번 경사하강법으로 학습을 할 때 NumPy 배열이 ```(samples, rows, cols, channels)``` 형태로 입력된다고 보면 된다. 이미지 데이터와 이미지 텐서의 차이에 대해서 혼란이 없길 바란다. 이미지 데이터는 3차원이다!


이제 다음 포스팅에서는 CNN을 이루는 핵심적인 요소인 패딩(padding)과 필터(filter), 풀링(pooling) 연산 등에 대해서 알아보자.

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/2.%20CNN/1-Basic-CNN/0-understanding-cnn-architecture.ipynb)에서 열람하실 수 있습니다!
