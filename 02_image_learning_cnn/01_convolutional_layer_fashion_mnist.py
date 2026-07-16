"""Fashion MNIST CNN 모델: Convolutional Layer로 이미지 특징 추출하기.

이 파일에서 공부하는 내용
- Dense 기본 모델과 CNN 모델의 차이
- Conv2D가 이미지에서 특징을 뽑는 방식
- kernel/filter가 하는 일
- MaxPooling2D로 중요한 특징만 남기는 이유
- CNN 결과를 Flatten해서 Dense 층으로 분류하는 흐름

핵심 아이디어
Dense 기본 모델은 이미지를 1차원으로 펼쳐서 보기 때문에
픽셀의 위치 관계를 충분히 살리기 어렵습니다.

CNN은 이미지를 2차원 형태로 유지한 채,
작은 필터를 움직이면서 선, 모서리, 질감 같은 특징을 뽑습니다.
그래서 이미지 분류 문제에 더 잘 어울립니다.
"""

import tensorflow as tf
import numpy as np


# 1. Fashion MNIST 데이터셋 불러오기
# Fashion MNIST는 10종류의 패션 이미지 분류 연습용 데이터셋입니다.
# 이미지는 28x28 크기의 흑백 이미지입니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# 2. 정답 번호를 이름으로 보기 위한 리스트
# 모델 학습에는 직접 사용하지 않지만,
# 나중에 예측 결과를 사람이 읽기 좋게 바꿀 때 사용할 수 있습니다.
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# 3. CNN 입력 모양으로 바꾸기
# Dense 기본 모델에서는 x_train 모양이 (60000, 28, 28)이었습니다.
# 하지만 Conv2D는 보통 4차원 입력을 받습니다.
#
# Conv2D 입력 모양:
# (이미지 개수, 세로, 가로, 채널)
#
# Fashion MNIST는 흑백 이미지라 채널이 1개입니다.
# 컬러 이미지는 보통 RGB라 채널이 3개입니다.
#
# -1은 이미지 개수를 자동으로 맞추라는 뜻입니다.
# 그래서 (60000, 28, 28)을 (60000, 28, 28, 1)로 바꿉니다.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# 4. 픽셀값 정규화하기
# 픽셀값은 0~255 범위입니다.
# 255로 나누면 모든 값이 0~1 사이가 되어 학습이 더 안정적입니다.
x_train = x_train / 255.0
x_test = x_test / 255.0


# 5. CNN 모델 만들기
# CNN 모델은 보통 다음 흐름을 가집니다.
#
# Conv2D      : 이미지에서 특징을 뽑음
# MaxPooling2D: 중요한 특징만 남기고 크기를 줄임
# Flatten     : Dense 층에 넣기 위해 1차원으로 펼침
# Dense       : 뽑힌 특징을 보고 최종 분류함
model = tf.keras.Sequential([
    # 첫 번째 특징 추출 단계입니다.
    #
    # Conv2D(32, (3, 3))의 뜻:
    # - 32: 필터를 32개 사용합니다. 즉 서로 다른 특징 32종류를 뽑아봅니다.
    # - (3, 3): 필터 크기입니다. 3x3 작은 창으로 이미지를 훑습니다.
    # - padding="same": 출력 이미지 크기가 입력과 비슷하게 유지되도록 가장자리를 채웁니다.
    # - activation="relu": 뽑은 특징 중 중요한 양수 신호를 살립니다.
    # - input_shape=(28, 28, 1): 입력 이미지 크기는 28x28 흑백 이미지라는 뜻입니다.
    #
    # kernel/filter란?
    # 이미지 위를 지나가면서 특정 패턴을 찾는 작은 숫자판입니다.
    # 예를 들어 선, 모서리, 곡선, 질감 같은 특징을 감지할 수 있습니다.
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        input_shape=(28, 28, 1),
    ),

    # MaxPooling2D는 특징맵의 크기를 줄입니다.
    # pool_size=(2, 2)는 2x2 영역마다 가장 큰 값만 남긴다는 뜻입니다.
    #
    # 왜 줄이는가?
    # - 계산량이 줄어듭니다.
    # - 사소한 위치 변화에 덜 예민해집니다.
    # - 중요한 특징만 남기는 효과가 있습니다.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 두 번째 특징 추출 단계입니다.
    # 앞 층에서는 단순한 선이나 모서리 같은 특징을 보고,
    # 뒤쪽 층으로 갈수록 더 복잡한 패턴을 조합해서 볼 수 있습니다.
    #
    # 필터를 64개로 늘린 이유:
    # 앞 단계보다 더 다양한 특징 조합을 뽑아보기 위해서입니다.
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Conv2D와 Pooling을 거친 결과는 아직 2차원 형태의 특징맵입니다.
    # Dense 층에 넣으려면 1차원 배열로 펼쳐야 해서 Flatten을 사용합니다.
    tf.keras.layers.Flatten(),

    # Dense 층은 CNN이 뽑은 특징을 바탕으로 최종 판단을 합니다.
    tf.keras.layers.Dense(128, activation="relu"),

    # Fashion MNIST는 정답 종류가 10개라 마지막 노드는 10개입니다.
    # softmax는 10개 클래스 각각의 확률을 출력합니다.
    tf.keras.layers.Dense(10, activation="softmax"),
])


# 6. 모델 구조 확인하기
# summary를 보면 Conv2D와 Pooling을 지나면서 이미지 크기가 어떻게 변하는지 볼 수 있습니다.
# CNN을 공부할 때 이 출력은 꽤 중요합니다.
model.summary()


# 7. 모델 학습 설정하기
# optimizer="adam": 가중치를 업데이트하는 방법입니다.
# loss="sparse_categorical_crossentropy": 정답이 0~9 정수 라벨일 때 쓰는 다중 분류 loss입니다.
# metrics=["accuracy"]: 정확도를 함께 확인합니다.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# 8. 모델 학습하기
# baseline 파일은 validation_split으로 검증 데이터를 나눴고,
# 여기서는 이미 분리되어 있는 x_test, y_test를 validation_data로 사용합니다.
#
# validation_data=(x_test, y_test):
# 학습 중간중간 테스트 데이터 기준 성능도 같이 확인합니다.
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
)


# 핵심 정리
# - Conv2D는 이미지에서 특징을 뽑는 층입니다.
# - kernel/filter는 이미지를 훑으며 특정 패턴을 찾는 작은 창입니다.
# - MaxPooling2D는 중요한 특징만 남기고 크기를 줄입니다.
# - Conv2D + Pooling을 여러 번 쌓으면 더 추상적인 특징을 배울 수 있습니다.
# - CNN은 픽셀 위치 관계를 살릴 수 있어서 이미지 분류에 Dense 기본 모델보다 잘 어울립니다.
# - Flatten은 CNN이 뽑은 특징맵을 Dense 분류기에 넣기 위해 필요합니다.
