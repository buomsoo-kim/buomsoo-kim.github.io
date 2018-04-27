---
layout: post
title: 케라스와 함께하는 쉬운 딥러닝 (6) - 뉴럴 네트워크의 학습 과정 개선하기
category: Keras
tags: [Python, Keras, Deep Learning, 케라스]
---

# 다층 퍼셉트론 6 (Improving techniques for training neural networks 3)

Objective: 인공신경망 모델을 효율적으로 학습시키기 위한 개선 방법들에 대해 학습한다.



[지지난 포스팅](https://buomsoo-kim.github.io/keras/2018/04/22/Easy-deep-learning-with-Keras-4.md/)과 [지난 포스팅](https://buomsoo-kim.github.io/keras/2018/04/22/Easy-deep-learning-with-Keras-4.md/)에서 뉴럴 네트워크의 학습 과정을 개선하기 위한 아래의 여섯 가지 방법에 대해서 알아보았다.

- 가중치 초기화(Weight Initialization)
- 활성함수(Activation Function)
- 최적화(Optimization)
- 배치 정규화(Batch Normalization)
- 드랍아웃(Dropout)
- 앙상블(Model Ensemble)

여기에 덧붙여, 지난번 포스팅에서는 학습 시간을 단축시키기 위하여 1/3의 학습 데이터만 가지고 학습을 시켰는데, 이번 포스팅에서는 전체 학습데이터를 가지고 신경망 모델을 학습 후 검증해 보자.

### MNIST 데이터 셋 불러오기

```python
# 케라스에 내장된 mnist 데이터 셋을 함수로 불러와 바로 활용 가능하다
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

```python
plt.imshow(X_train[0])    # show first number in the dataset
plt.show()
print('Label: ', y_train[0])
```
<p align = "center"><br>
<img src ="/data/images/2018-04-22/2.png" width = "400px"/>
</p>

```python
plt.imshow(X_test[0])    # show first number in the dataset
plt.show()
print('Label: ', y_test[0])
```

<p align = "center"><br>
<img src ="/data/images/2018-04-22/3.png" width = "400px"/>
</p>

### 데이터 셋 전처리

앞서 언급했다시피, MNIST 데이터는 흑백 이미지 형태로, 2차원 행렬(28 X 28)과 같은 형태라고 할 수 있다.

<p align = "center"><br>
<img src ="https://www.tensorflow.org/versions/r1.1/images/MNIST-Matrix.png" width = "600px"/>
</p>

하지만 이와 같은 이미지 형태는 우리가 지금 활용하고자 하는 다층 퍼셉트론 모델에는 적합하지 않다. 다층 퍼셉트론은 죽 늘어놓은 1차원 벡터와 같은 형태의 데이터만 받아들일 수 있기 때문이다. 그러므로 우리는 28 X 28의 행렬 형태의 데이터를 재배열(reshape)해 784 차원의 벡터로 바꾼다.

```python
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# reshaping X data: (n, 28, 28) => (n, 784)
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))


# 타겟 변수를 one-hot encoding 한다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

아래에 보다시피 이번에는 6만개의 학습 데이터를 가지고 모델을 학습시키고, 1만개의 데이터를 가지고 학습 결과를 검증해 본다.

```python
(60000, 784), (10000, 784), (60000,), (10000,)
```

### 모델 생성 및 학습

우리가 처음 생성해 보았던 [Vanilla MLP](https://buomsoo-kim.github.io/keras/2018/04/22/Easy-deep-learning-with-Keras-3.md/)와 어떻게 다른지 한번 비교해 보자.

```python
def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_shape = (784, ), kernel_initializer='he_normal'))   # 가중치 초기화 방식 변경
    model.add(BatchNormalization())     # 배치 정규화 레이어 추가
    model.add(Activation('relu'))       # 활성함수로 Relu 사용
    model.add(Dropout(0.2))             # 드랍아웃 레이어 추가
    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr = 0.001)    # Adam optimizer 사용
    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model
```

앙상블 할 때 모델의 개수도 5개로 늘려본다.

```python
# create 5 models to ensemble
model1 = KerasClassifier(build_fn = mlp_model, epochs = 100)
model2 = KerasClassifier(build_fn = mlp_model, epochs = 100)
model3 = KerasClassifier(build_fn = mlp_model, epochs = 100)
model4 = KerasClassifier(build_fn = mlp_model, epochs = 100)
model5 = KerasClassifier(build_fn = mlp_model, epochs = 100)

:
ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3), ('model4', model4), ('model5', model5)], voting = 'soft')
ensemble_clf.fit(X_train, y_train)
```

### 모델 평가

기본 모델에 배치정규화만 적용했음에도 불구하고 91.54%의 높은 정확도를 보인다. 앞으로 네트워크를 만들 때 배치 정규화는 웬만하면 적용해 주는 것이 좋다([논문](https://arxiv.org/abs/1502.03167)에 따르면 뒤에서 나올 regularization의 효과도 있어 배치 정규화를 적용하면 드랍아웃을 적용하지 않아도 된다고 한다).

```python
y_pred = ensemble_clf.predict(X_test)
print('Acc: ', accuracy_score(y_pred, y_test))
```

```python
Acc:  0.9801
```

최종 결과로 98%가 넘는 높은 정확도를 기록하였다. 모델 구조(복잡도나 크기)는 바꾸지 않았음에도 불구하고 아무런 개선 사항도 적용하지 않은 [Vanilla MLP](https://buomsoo-kim.github.io/keras/2018/04/22/Easy-deep-learning-with-Keras-3.md/)가 20% 정도의 정확도를 기록했던 것과 비교하면 괄목할 만한 성과이다.

일반적으로 뉴럴 네트워크를 학습시킬 때에는 여기서 나온 <strong>7가지 학습 개선 방법(가중치 초기화, 활성함수, 최적화, 배치 정규화, 드랍아웃, 앙상블, 학습 데이터 추가)</strong>을 모두 총동원하여 정확도(혹은 정밀도, 재현율, ROC 등)를 조금이라도 높이기 위해서 노력한다. 즉, 최소한 현업에 적용할 만한 뉴럴 네트워크를 만들 때 여기서 나온 방법은 모두 활용을 해볼만한 가치가 있다는 얘기다. 물론 어떤 상황에 어떤 방식이 유효할지는 그때그때 다르기 때문에 최적화된 결과를 위해서는 삽질을 통해 실험을 많이 해봐야 한다... (*데이터 사이언스의 8할은 노가다이다*)

이제 다음 포스팅에서는 이미지 데이터를 학습하는 데 최적화된 모델로 알려져 있는 합성곱 신경망(CNN; Convolutional Neural Networks)에 대해서 알아 보자!

# 전체 코드

본 실습의 전체 코드는 [여기](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/1.%20MLP/3-Advanced-MLP-2/2-Advanced-MLP-2.ipynb)에서 열람하실 수 있습니다!
