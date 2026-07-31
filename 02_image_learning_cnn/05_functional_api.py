"""Sequential 모델과 Functional API 비교하기.

이번 파일에서 새로 배우는 핵심은 Functional API입니다.

이전에 사용한 Sequential 방식은 레이어가 위에서 아래로 한 줄로 이어지는 구조에 좋습니다.
하지만 모델이 중간에 갈라지거나, 여러 입력/출력을 다루거나, 복잡한 연결을 만들 때는
Functional API가 훨씬 자연스럽습니다.

이 파일에서는 같은 Fashion MNIST 데이터를 사용해서
1. Sequential 모델을 먼저 만들고
2. Functional API로 조금 더 자유로운 구조의 모델을 만든 뒤
3. 두 모델 구조를 그림으로 저장해봅니다.
"""

import tensorflow as tf
from tensorflow.keras.utils import plot_model


# 1. 데이터 준비
# Fashion MNIST는 이전 예제에서도 사용한 28x28 흑백 의류 이미지 데이터입니다.
# 여기서는 Functional API 구조를 공부하는 것이 목적이라 데이터 설명은 짧게만 정리합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 이미지 픽셀값은 0~255 범위입니다.
# 0~1 사이로 줄이면 모델이 더 안정적으로 학습합니다.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Conv2D나 Input(shape=(28, 28, 1))처럼 이미지 채널까지 받는 레이어를 쓰려면
# 데이터 모양을 (개수, 세로, 가로, 채널) 형태로 맞춰야 합니다.
# Fashion MNIST는 흑백 이미지라 채널 수가 1입니다.
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))


# 2. 비교용 Sequential 모델
# Sequential은 레이어가 순서대로만 쌓이는 가장 단순한 모델 작성 방식입니다.
# 입력 -> Flatten -> Dense -> Dense 처럼 한 방향으로만 흐를 때 쓰기 좋습니다.
sequential_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ],
    name="fashion_mnist_sequential_model",
)

sequential_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)


def save_model_image(model, file_name):
    """모델 구조를 이미지 파일로 저장합니다.

    plot_model은 모델의 레이어 연결 구조를 그림으로 보여주는 도구입니다.
    특히 Functional API처럼 연결이 갈라지고 다시 합쳐지는 모델을 볼 때 유용합니다.

    주의:
        plot_model을 사용하려면 Python 패키지 pydot뿐만 아니라
        Graphviz 프로그램의 dot 명령어도 컴퓨터에 설치되어 있어야 합니다.

    그래서 여기서는 try/except로 감싸둡니다.
    Graphviz 설정이 안 되어 있어도 학습 자체는 계속 진행되게 하기 위해서입니다.
    """
    try:
        plot_model(
            model,
            to_file=file_name,
            show_shapes=True,
            show_layer_names=True,
        )
        print(f"[모델 그림 저장 완료] {file_name}")
    except Exception as error:
        print(f"[모델 그림 저장 건너뜀] {file_name}")
        print("plot_model을 쓰려면 pydot과 Graphviz가 필요합니다.")
        print(f"현재 오류: {error}")


# Sequential 모델 구조를 그림으로 저장합니다.
# 이 그림은 Functional API 모델 그림과 비교하기 위한 기준 역할입니다.
save_model_image(sequential_model, "model.png")


# =========================
# 3. Functional API 모델 만들기
# =========================

# Functional API는 먼저 입력층을 따로 만듭니다.
#
# Sequential:
#   모델 안에 첫 레이어부터 순서대로 넣음
#
# Functional API:
#   "이 입력이 어느 레이어를 지나서 어느 출력으로 가는지"를 직접 연결함
#
# 즉, Functional API에서는 레이어를 함수처럼 사용합니다.
# 아래 코드의 의미는 "inputs가 Flatten 레이어를 통과해서 x가 된다"입니다.
inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="input_image")

# 28x28x1 이미지를 Dense 레이어에 넣기 위해 1차원으로 펼칩니다.
# 예: 28 * 28 * 1 = 784개의 숫자로 변환
x = tf.keras.layers.Flatten(name="flatten_image")(inputs)

# 첫 번째 Dense 레이어입니다.
# 여기서 나온 dense1은 뒤에서 두 갈래 branch에 같이 사용됩니다.
# 이처럼 중간 결과를 변수에 담고 재사용할 수 있는 것이 Functional API의 장점입니다.
dense1 = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)

# branch 1
# 첫 번째 가지는 dense1 결과를 그대로 사용합니다.
branch1 = dense1

# branch 2
# 두 번째 가지는 dense1 결과의 모양을 잠깐 바꿔봅니다.
#
# dense1의 출력 개수는 128개입니다.
# 16 * 8 = 128이므로 Reshape((16, 8))이 가능합니다.
#
# 여기서는 성능 향상 목적이라기보다,
# Functional API에서 "한 흐름을 다른 형태로 바꾼 뒤 다시 이어붙일 수 있다"는 것을 보기 위한 예제입니다.
branch2 = tf.keras.layers.Reshape((16, 8), name="reshape_16x8")(dense1)
branch2 = tf.keras.layers.Flatten(name="flatten_reshape")(branch2)

# 두 갈래로 나뉜 값을 다시 합칩니다.
#
# Concatenate는 여러 텐서를 옆으로 이어붙이는 레이어입니다.
# branch1도 128개, branch2도 다시 펼치면 128개이므로
# 합치면 총 256개의 특징이 됩니다.
concat = tf.keras.layers.Concatenate(name="concat_branches")([branch1, branch2])

# 합쳐진 특징을 Dense 레이어에 넣어 최종 판단에 사용할 정보를 다시 정리합니다.
x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(concat)

# 최종 출력층입니다.
# Fashion MNIST는 정답 종류가 10개라 Dense(10)을 사용합니다.
# softmax는 10개 클래스 중 각각일 확률처럼 해석할 수 있게 해줍니다.
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_class")(x)

# Functional API에서는 마지막에 Model로 입력과 출력을 묶어줍니다.
# "inputs에서 시작해서 outputs까지 이어진 계산 그래프가 하나의 모델이다"라는 뜻입니다.
functional_model = tf.keras.models.Model(
    inputs=inputs,
    outputs=outputs,
    name="fashion_mnist_functional_model",
)

# summary는 모델의 레이어 이름, 출력 모양, 파라미터 수를 표로 보여줍니다.
# Functional API에서는 연결 구조가 복잡해질 수 있으니 summary로 중간 모양을 자주 확인하면 좋습니다.
functional_model.summary()

# Functional API 모델은 연결이 갈라졌다가 합쳐지기 때문에 그림으로 보면 이해가 쉽습니다.
save_model_image(functional_model, "model_functional.png")

# 모델 학습 설정입니다.
# sparse_categorical_crossentropy는 정답이 0~9 같은 정수 라벨일 때 사용합니다.
functional_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 실제 학습입니다.
# validation_data를 넣으면 학습 데이터뿐 아니라 테스트 데이터 성능도 같이 확인할 수 있습니다.
functional_model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=3,
)
