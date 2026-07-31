"""TensorBoard와 EarlyStopping 사용해보기.

이 파일은 Fashion MNIST 이미지 분류 모델을 학습하면서
TensorBoard와 EarlyStopping을 어떻게 사용하는지 연습하는 예제입니다.

공부 목표
- TensorBoard로 학습 과정을 로그로 저장하기
- 터미널에서 TensorBoard를 실행해 학습 그래프 확인하기
- EarlyStopping으로 학습을 자동으로 멈추게 하기
- 모델 구조가 달라지면 결과도 달라질 수 있다는 점 확인하기

중요한 흐름
1. Fashion MNIST 데이터를 불러옵니다.
2. Conv2D에 넣을 수 있게 이미지 모양을 바꿉니다.
3. 첫 번째 CNN 모델을 학습합니다.
4. TensorBoard 로그를 저장합니다.
5. EarlyStopping으로 불필요한 학습을 막습니다.
6. 두 번째 CNN 모델을 만들어 첫 번째 모델과 비교합니다.
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


# 1. Fashion MNIST 데이터 불러오기
# Fashion MNIST는 옷, 신발, 가방 같은 패션 이미지 10종류를 분류하는 연습용 데이터셋입니다.
# TensorFlow 안에 기본으로 들어 있어서 바로 불러올 수 있습니다.
#
# x_train: 학습용 이미지
# y_train: 학습용 정답
# x_test: 테스트용 이미지
# y_test: 테스트용 정답
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# 2. 픽셀값 정규화
# 이미지 픽셀값은 원래 0~255 사이 숫자입니다.
# - 0은 검은색
# - 255는 흰색
#
# 딥러닝 모델은 0~255처럼 큰 숫자보다 0~1 사이 숫자를 더 안정적으로 학습합니다.
# 그래서 255로 나누어 모든 픽셀값을 0~1 사이로 바꿉니다.
x_train = x_train / 255.0
x_test = x_test / 255.0


# 3. 이미지 모양 바꾸기
# reshape 해주는 이유는 Conv2D 레이어를 사용하기 위해서입니다.
#
# Fashion MNIST를 처음 불러오면 이미지 모양은 이렇게 생겼습니다.
# x_train.shape -> (60000, 28, 28)
#
# 하지만 Conv2D는 보통 4차원 데이터를 입력으로 받습니다.
# (이미지 개수, 세로, 가로, 채널)
#
# Fashion MNIST는 흑백 이미지라 채널이 1개입니다.
# 그래서 아래처럼 바꿉니다.
# (60000, 28, 28) -> (60000, 28, 28, 1)
#
# -1은 이미지 개수를 자동으로 맞추라는 뜻입니다.
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))


# 4. 첫 번째 모델 만들기
# 모델이 달라짐에 따라 결과가 달라지는지 확인하려면
# 모델을 하나 학습하고, 다른 모델을 또 학습해서 결과를 비교하면 됩니다.

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),


    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])


# 5. 첫 번째 모델 학습 설정
# sparse_categorical_crossentropy:
# - 정답이 0~9 같은 정수 라벨일 때 사용하는 다중 분류 loss입니다.
#
# optimizer="adam":
# - 모델의 가중치를 업데이트하는 방법입니다.
# - 입문 예제에서 자주 쓰는 안정적인 optimizer입니다.
#
# metrics=["accuracy"]:
# - 학습 중 정확도를 같이 확인합니다.
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)


# 6. TensorBoard와 EarlyStopping 준비
# TensorBoard:
# - 학습 과정의 loss, accuracy를 로그 파일로 저장합니다.
# - 나중에 브라우저에서 그래프로 볼 수 있습니다.
#
# time.time():
# - 현재 시간을 숫자로 가져옵니다.
# - 로그 폴더 이름에 시간을 붙이면 실행할 때마다 새 폴더가 생겨서 이전 기록과 섞이지 않습니다.
#
# os.path.join():
# - 폴더명과 파일명을 안전하게 이어붙입니다.
# - Windows, Mac, Linux마다 경로 구분자가 다를 수 있어서 사용합니다.
log_dir = os.path.join("logs", "first_model_" + str(int(time.time())))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# EarlyStopping:
# - 학습을 오래 한다고 항상 좋아지는 것은 아닙니다.
# - 검증 데이터 성능이 더 이상 좋아지지 않으면 자동으로 멈추게 할 수 있습니다.
#
# monitor="val_loss":
# - validation loss를 지켜봅니다.
#
# patience=3:
# - val_loss가 3번 연속 좋아지지 않으면 멈춥니다.
#
# 지금은 epochs=3이라 EarlyStopping 효과가 크게 보이진 않습니다.
# EarlyStopping을 제대로 보고 싶으면 epochs를 20, 30처럼 더 크게 두면 됩니다.
earlystopping = EarlyStopping(monitor="val_loss", patience=3)


# 7. 첫 번째 모델 학습
# validation_data=(x_test, y_test):
# - 학습 중간마다 테스트 데이터로 검증 성능을 확인합니다.
#
# callbacks=[tensorboard, earlystopping]:
# - 학습 중 TensorBoard 로그를 저장하고,
# - EarlyStopping 조건도 같이 확인합니다.
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=3,
    callbacks=[tensorboard, earlystopping],
)


# 8. 두 번째 모델 만들기
# 두 번째 모델은 첫 번째 모델보다 Conv2D 층이 하나 더 많습니다.
#
# 모델을 더 깊게 만들면 더 복잡한 특징을 배울 수 있습니다.
# 하지만 무조건 성능이 좋아지는 것은 아닙니다.
# 그래서 TensorBoard로 두 모델의 학습 결과를 비교해보는 것이 좋습니다.
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # 첫 번째 모델에는 없던 세 번째 Conv2D 층입니다.
    # 더 많은 특징을 뽑아보려는 목적입니다.
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])


# 9. 두 번째 모델 학습 설정
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)


# 10. 두 번째 모델용 TensorBoard 로그 만들기
# first_model과 second_model의 로그 폴더 이름을 다르게 해야
# TensorBoard에서 두 실험을 따로 비교할 수 있습니다.
log_dir = os.path.join("logs", "second_model_" + str(int(time.time())))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# 11. 두 번째 모델용 EarlyStopping
# mode="min":
# - val_loss는 낮을수록 좋기 때문에 min을 사용합니다.
#
# restore_best_weights=True:
# - 학습이 멈춘 뒤 마지막 상태가 아니라 가장 성능이 좋았던 상태로 되돌립니다.
# - EarlyStopping을 쓸 때 자주 같이 사용하는 옵션입니다.
earlystopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    mode="min",
    restore_best_weights=True,
)


# 12. 두 번째 모델 학습
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=3,
    callbacks=[tensorboard, earlystopping],
)


# 13. TensorBoard 실행 방법
# PowerShell에서 아래 명령어를 실행하면 브라우저에서 그래프를 볼 수 있습니다.
# tensorboard --logdir logs



# 핵심 정리
# - TensorBoard는 학습 로그를 저장하고 그래프로 확인하는 도구입니다.
# - EarlyStopping은 검증 성능이 좋아지지 않으면 자동으로 학습을 멈추는 도구입니다.
# - patience는 몇 번까지 기다릴지 정하는 값입니다.
# - restore_best_weights=True는 가장 좋았던 모델 상태로 되돌리는 옵션입니다.
# - 모델 구조를 바꾸며 여러 번 실험할 때 TensorBoard를 쓰면 비교하기 쉽습니다.
