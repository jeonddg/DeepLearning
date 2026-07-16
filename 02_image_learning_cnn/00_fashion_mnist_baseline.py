"""Fashion MNIST 기본 모델: 이미지를 숫자로 바꿔 분류하기.

이 파일에서 공부하는 내용
- 이미지는 컴퓨터 안에서 숫자 배열로 저장된다는 점
- Fashion MNIST 데이터셋을 불러오는 방법
- 픽셀값을 0~1 사이로 정규화하는 이유
- Flatten으로 2차원 이미지를 1차원으로 펼치는 이유
- Dense 층만 사용한 기본 이미지 분류 모델 만들기

중요한 관점
컴퓨터는 이미지를 '옷', '신발', '가방'처럼 바로 이해하지 못합니다.
이미지는 픽셀이라는 작은 칸들의 숫자 모음입니다.
딥러닝 모델은 이 숫자 패턴을 보고 어떤 종류의 이미지인지 배웁니다.
"""

import tensorflow as tf
import matplotlib.pyplot as plt


# 1. Fashion MNIST 데이터셋 불러오기
# Fashion MNIST는 옷, 신발, 가방 같은 10가지 패션 이미지를 모아둔 연습용 데이터셋입니다.
# TensorFlow 안에 기본으로 들어 있어서 따로 파일을 다운로드하지 않아도 쉽게 사용할 수 있습니다.
#
# x_train: 학습용 이미지 데이터
# y_train: 학습용 정답 라벨
# x_test: 테스트용 이미지 데이터
# y_test: 테스트용 정답 라벨
#
# train 데이터는 모델이 공부할 때 사용하고,
# test 데이터는 모델이 공부 후 얼마나 잘 맞히는지 확인할 때 사용합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# 2. 데이터 모양 확인하기
# x_train.shape가 (60000, 28, 28)이라는 뜻:
# - 이미지가 60000장 있습니다.
# - 각 이미지는 세로 28칸, 가로 28칸입니다.
# - 흑백 이미지라 색상 채널 정보는 따로 없습니다.
#
# y_train.shape가 (60000,)이라는 뜻:
# - 이미지 60000장에 대한 정답 번호가 60000개 있습니다.
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)


# 3. 숫자 라벨을 사람이 읽을 수 있는 이름으로 바꾸기
# y_train에는 정답이 0~9 숫자로 들어 있습니다.
# 예를 들어 0은 T-shirt/top, 9는 Ankle boot를 의미합니다.
#
# 모델은 숫자 라벨로 학습하지만,
# 사람이 결과를 볼 때는 이름이 있어야 이해하기 쉬워서 class_names를 만듭니다.
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


# 4. 이미지 직접 확인해보기
# 이미지는 숫자 배열이지만, matplotlib으로 그림처럼 볼 수 있습니다.
#
# 지금은 학습할 때마다 창이 뜨는 것을 막기 위해 주석 처리해두었습니다.
# 이미지가 어떻게 생겼는지 보고 싶으면 아래 주석을 풀면 됩니다.
"""
plt.imshow(x_train[1], cmap="gray")
plt.title(class_names[y_train[1]])
plt.axis("off")
plt.show()
"""


# 5. 픽셀값 정규화하기
# 흑백 이미지는 각 픽셀이 0~255 사이 숫자로 들어 있습니다.
# - 0은 검은색
# - 255는 흰색
# - 중간값은 회색입니다.
#
# 딥러닝 모델은 너무 큰 숫자보다 0~1 사이의 작은 숫자를 더 안정적으로 학습합니다.
# 그래서 255로 나누어 모든 픽셀값을 0~1 사이로 바꿉니다.
x_train = x_train / 255.0
x_test = x_test / 255.0


# 6. 기본 이미지 분류 모델 만들기
# 이 모델은 CNN을 쓰지 않고 Dense 층만 사용하는 기본 모델입니다.
# 그래서 이미지의 위치 정보나 주변 픽셀 관계를 깊게 보지는 못합니다.
# 대신 이미지 분류의 기본 흐름을 이해하기 좋습니다.
model = tf.keras.Sequential([
    # Flatten은 28x28 이미지를 길이 784짜리 1차원 배열로 펼칩니다.
    # Dense 층은 1차원 입력을 받기 때문에 Flatten이 필요합니다.
    #
    # 예:
    # 28 x 28 이미지 -> 784개 숫자
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Dense는 모든 입력 노드가 모든 출력 노드와 연결되는 층입니다.
    # 256은 노드 개수입니다.
    # relu는 음수는 0으로 만들고, 양수는 그대로 통과시키는 활성화 함수입니다.
    # relu를 쓰면 모델이 단순한 직선이 아니라 더 복잡한 패턴을 배울 수 있습니다.
    tf.keras.layers.Dense(256, activation="relu"),

    # 두 번째 Dense 층입니다.
    # 층을 하나 더 쌓으면 모델이 더 복잡한 특징 조합을 배울 수 있습니다.
    tf.keras.layers.Dense(128, activation="relu"),

    # 마지막 층은 정답 종류가 10개라서 노드도 10개입니다.
    # softmax는 10개 출력값을 확률처럼 바꿔줍니다.
    # 예: [0.01, 0.03, ..., 0.80]처럼 전체 합이 1에 가까운 값이 됩니다.
    tf.keras.layers.Dense(10, activation="softmax"),
])


# 7. 모델 구조 확인하기
# summary()를 보면 각 층의 출력 모양과 학습할 파라미터 수를 확인할 수 있습니다.
# 모델이 내가 생각한 구조로 만들어졌는지 확인할 때 자주 사용합니다.
model.summary()


# 8. 모델 학습 설정하기
# compile은 모델을 어떤 방식으로 학습할지 정하는 단계입니다.
#
# optimizer="adam"
# - 모델의 가중치를 어떻게 업데이트할지 정합니다.
# - Adam은 입문 단계에서 많이 쓰는 안정적인 optimizer입니다.
#
# loss="sparse_categorical_crossentropy"
# - 정답 라벨이 0~9 같은 정수 하나로 들어 있을 때 사용합니다.
# - 만약 정답이 [0, 0, 1, 0, ...] 같은 원-핫 인코딩 형태라면 categorical_crossentropy를 씁니다.
#
# metrics=["accuracy"]
# - 학습 중 정확도를 같이 보여줍니다.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# 9. 모델 학습하기
# epochs=10은 전체 학습 데이터를 10번 반복해서 본다는 뜻입니다.
# batch_size=32는 데이터를 32개씩 묶어서 학습한다는 뜻입니다.
# validation_split=0.2는 학습 데이터 중 20%를 검증용으로 잠깐 떼어낸다는 뜻입니다.
#
# 검증 데이터는 모델이 학습 중간에 실력이 좋아지고 있는지 확인하는 용도입니다.
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
)


# 핵심 정리
# - 이미지는 픽셀 숫자 배열입니다.
# - Fashion MNIST 이미지는 28x28 크기의 흑백 이미지입니다.
# - 정규화는 0~255 픽셀값을 0~1로 줄여 학습을 안정적으로 만듭니다.
# - Flatten은 2차원 이미지를 Dense 층에 넣기 위해 1차원으로 펼칩니다.
# - Dense 기본 모델도 이미지 분류는 가능하지만, 이미지의 공간적인 특징을 잘 살리지는 못합니다.
