---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (17) - 순환형 신경망(RNN) 기초
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 순환형 신경망 3 - RNN 구조 기초

Objective: RNN의 기본적인 구조를 이해하고 케라스로 이를 구현해본다

[지난 포스팅](https://buomsoo-kim.github.io/keras/2019/06/25/Easy-deep-learning-with-Keras-16.md/)기본적인 RNN 셀(SimpleRNN)에 이어 LSTM에 대해 알아보았다. 이번 포스팅에서는 또 다른 형태의 RNN인 GRU에 대해 알아보자.

### GRU

GRU는 Gated Recurrent Unit의 약자로, LSTM과 유사하게 forget gate가 있지만 output gate가 없다. 그러므로 기본적인 RNN 셀과 같이 hidden state가 없고, LSTM보다 적은 수의 파라미터로 학습이 가능하다.


<p align = "center"><br>
<img src ="https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Gated_Recurrent_Unit%2C_base_type.svg/220px-Gated_Recurrent_Unit%2C_base_type.svg.png" width = "600px"/>
</p>

### 케라스에서 GRU 구현하기

```python
# 셀 사이즈 50의 lstm 셀 생성
gru = GRU(50)(Input(shape = (10, 30)))
print(gru.shape)
```

```python
(?, 50)
```

- ```return_sequences``` 파라미터가 False로 설정된 경우

```python
gru = GRU(50, return_sequences = False, return_state = True)(Input(shape = (10, 30)))
print(gru[0].shape)         # shape of output
print(gru[1].shape)         # shape of hidden state
```

```python
(?, 50)
(?, 50)
```

- ```return_sequences``` 파라미터가 True로 설정하는 경우.

```python
gru = GRU(50, return_sequences = True, return_state = True)(Input(shape = (10, 30)))
print(gru[0].shape)         # shape of output
print(gru[1].shape)         # shape of hidden state
```

```python
(?, ?, 50)
(?, 50)
```

아래와 같이 output과 hidden state 를 서로 다른 변수로 받을 수도 있다.

```python
output, hidden_state = GRU(50, return_sequences = True, return_state = True)(Input(shape = (10, 30)))
print(output.shape)
print(hidden_state.shape)
```

```python
(?, ?, 50)
(?, 50)
```

이번 포스팅에서는 기본적인 RNN 셀과는 조금 다른 구조를 갖는 GRU셀에 대해서 알아보았다. 다음 포스팅에서는 실제로 순차형 데이터의 학습을 위해 RNN 을 활용하는 방법에 대해서 알아보자.

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/1-Basic-RNN/0-understanding-rnn-structure.ipynb)에서 열람하실 수 있습니다!
