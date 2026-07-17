"""Fashion MNIST 모델 저장과 불러오기.

이 파일에서 공부하는 내용
- 학습한 모델을 파일로 저장하는 이유
- model.save()로 '모델 전체'를 저장하는 방법
- load_model()로 저장된 모델을 다시 불러오는 방법
- ModelCheckpoint로 학습 중 가장 좋은 가중치만 저장하는 방법
- load_weights()로 가중치만 불러오는 방법

왜 모델을 저장할까?
모델 학습은 시간이 오래 걸릴 수 있습니다.
학습이 끝난 모델을 저장해두면, 다음에 다시 학습하지 않고 바로 불러와서 사용할 수 있습니다.

저장 방식은 크게 두 가지입니다.

1. 모델 전체 저장
  - 모델 구조
  - 학습된 가중치
  - compile 설정(loss, optimizer, metrics 등)
  을 함께 저장합니다.

2. 가중치만 저장
  - 모델 구조는 저장하지 않습니다.
  - 학습된 숫자 값, 즉 weights만 저장합니다.
  - 나중에 같은 구조의 모델을 다시 만든 뒤 load_weights()로 불러와야 합니다.
"""

import os

import tensorflow as tf


# 1. 저장할 폴더 준비
#
# os.path.dirname(__file__)
# - 현재 파이썬 파일이 들어 있는 폴더 경로를 가져옵니다.
# - 터미널을 어디에서 실행해도 02_image_learning_cnn 폴더 안에 저장되게 하기 위해 사용합니다.
#
# os.path.join(...)
# - 폴더명과 파일명을 운영체제에 맞게 안전하게 이어붙입니다.
#
# os.makedirs(..., exist_ok=True)
# - 폴더가 없으면 새로 만듭니다.
# - 이미 폴더가 있어도 에러를 내지 않습니다.
script_dir = os.path.dirname(__file__)
model_dir = os.path.join(script_dir, "model")
checkpoint_dir = os.path.join(script_dir, "checkpoints")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


# 2. Fashion MNIST 데이터 불러오기
# Fashion MNIST는 28x28 흑백 패션 이미지 데이터셋입니다.
# 여기서는 모델 저장 방법을 연습하는 것이 목적이라 익숙한 데이터셋을 사용합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# 3. 이미지 전처리
# 원래 이미지 모양은 (이미지 개수, 28, 28)입니다.
# Conv2D는 (이미지 개수, 세로, 가로, 채널) 형태를 기대합니다.
# Fashion MNIST는 흑백 이미지라 채널이 1개입니다.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 픽셀값은 0~255 범위입니다.
# 255로 나누어 0~1 사이 값으로 바꾸면 학습이 더 안정적입니다.
x_train = x_train / 255.0
x_test = x_test / 255.0


# 4. 모델을 만드는 함수
# 같은 모델 구조를 여러 번 사용해야 하므로 함수로 따로 빼두었습니다.
#
# 왜 함수로 만들까?
# - 아래에서 첫 번째 모델과 두 번째 모델이 같은 구조여야 합니다.
# - 코드를 복사해서 두 번 쓰면 실수하기 쉽습니다.
# - 함수로 만들면 같은 구조의 새 모델을 쉽게 만들 수 있습니다.
def build_model():
    model = tf.keras.Sequential([
        # Conv2D는 이미지에서 선, 모서리, 질감 같은 특징을 뽑는 층입니다.
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 두 번째 Conv2D 층입니다.
        # 앞 층보다 더 복잡한 특징 조합을 배울 수 있습니다.
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Conv2D 결과는 2차원 특징맵이므로 Dense 층에 넣기 위해 1차원으로 펼칩니다.
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),

        # Fashion MNIST는 정답 종류가 10개라 마지막 노드도 10개입니다.
        # softmax는 10개 클래스 각각의 확률을 출력합니다.
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    # sparse_categorical_crossentropy는 정답이 0~9 같은 정수 라벨일 때 사용합니다.
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# 5. 첫 번째 모델 학습하기
model = build_model()
model.summary()

model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=3,
)


# 6. 모델 전체 저장하기
# .keras 형식은 Keras에서 권장하는 모델 저장 형식입니다.
# 모델 구조, 가중치, compile 정보가 함께 저장됩니다.
full_model_path = os.path.join(model_dir, "fashion_mnist.keras")
model.save(full_model_path)
print(f"\n모델 전체 저장 완료: {full_model_path}")


# 7. 저장한 모델 전체 불러오기
# load_model()을 사용하면 저장했던 모델을 그대로 복원할 수 있습니다.
# 모델 구조와 가중치가 함께 들어 있으므로 build_model()을 다시 호출하지 않아도 됩니다.
loaded_model = tf.keras.models.load_model(full_model_path)

print("\n[불러온 모델 구조]")
loaded_model.summary()

print("\n[불러온 모델 평가]")
loaded_model.evaluate(x_test, y_test)


# 8. 가중치만 저장하는 방법: ModelCheckpoint
# 이번에는 모델 전체가 아니라 weights만 저장해봅니다.
#
# ModelCheckpoint는 학습 중간중간 모델을 저장해주는 callback입니다.
# callback은 fit()이 진행되는 동안 특정 시점마다 자동으로 실행되는 도구입니다.
#
# save_weights_only=True
# - 모델 전체가 아니라 가중치만 저장합니다.
#
# monitor="val_loss"
# - 검증 데이터의 loss를 기준으로 가장 좋은 모델을 고릅니다.
#
# mode="min"
# - val_loss는 작을수록 좋으므로 min을 사용합니다.
#
# save_best_only=True
# - 매 epoch마다 저장하지 않고, val_loss가 더 좋아졌을 때만 저장합니다.
checkpoint_path = os.path.join(checkpoint_dir, "fashion_mnist_weights.h5")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq="epoch",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)


# 9. 체크포인트를 사용해서 다시 학습하기
# 체크포인트 저장 연습을 위해 같은 구조의 새 모델을 하나 더 만듭니다.
checkpoint_model = build_model()

checkpoint_model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=3,
    callbacks=[checkpoint],
)


# 10. 가중치만 불러오기
# 가중치만 저장한 경우에는 먼저 같은 구조의 모델을 만들어야 합니다.
# 그 다음 load_weights()로 저장된 가중치를 넣습니다.
restored_model = build_model()
restored_model.load_weights(checkpoint_path)

print("\n[체크포인트 가중치를 불러온 모델 평가]")
restored_model.evaluate(x_test, y_test)


# 핵심 정리
# - model.save()는 모델 전체를 저장합니다.
# - load_model()은 저장된 모델 전체를 다시 불러옵니다.
# - ModelCheckpoint는 학습 중 좋은 모델 또는 가중치를 자동 저장합니다.
# - save_weights_only=True이면 가중치만 저장합니다.
# - 가중치만 불러오려면 반드시 같은 구조의 모델을 먼저 만들어야 합니다.
# - 저장된 모델 파일, 체크포인트 파일은 보통 GitHub에 올리지 않습니다.

