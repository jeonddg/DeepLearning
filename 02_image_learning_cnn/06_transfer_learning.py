"""전이학습으로 고양이/강아지 분류하기.

오늘 공부할 내용은 전이학습입니다. -> 정확도 97.6 달성

전이학습이란?
이미 다른 사람이 큰 데이터로 학습해둔 모델을 가져와서,
내 데이터에 맞게 마지막 부분만 새로 붙이거나 일부만 추가 학습시키는 방법입니다.

이번 파일의 흐름
1. Kaggle 고양이/강아지 데이터를 TensorFlow Dataset으로 준비한다.
2. 이미 학습된 InceptionV3 모델 구조를 가져온다.
3. 다운받은 inception_v3.h5 weight 파일을 적용한다.
4. InceptionV3의 앞부분은 이미지 특징 추출기로 사용한다.
5. 뒤에 내가 필요한 고양이/강아지 분류층을 붙인다.
6. 새로 만든 모델을 학습시킨다.
"""

import os
import shutil

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3


# =========================
# 1. 데이터 준비
# =========================

# 데이터는 Kaggle Dogs vs. Cats Redux 데이터셋을 사용합니다.
# https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
#
# 이미 여러 번 했던 부분이라 간단히 정리합니다.
# 원본 train 폴더에는 cat.0.jpg, dog.0.jpg처럼 고양이/강아지 이미지가 한 폴더에 섞여 있습니다.
# image_dataset_from_directory를 쓰려면 dataset/cat, dataset/dog처럼 폴더를 나눠야 합니다.
DATA_ROOT = "data/dogs-vs-cats-redux-kernels-edition"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
DATASET_DIR = os.path.join(DATA_ROOT, "dataset")
CAT_DIR = os.path.join(DATASET_DIR, "cat")
DOG_DIR = os.path.join(DATASET_DIR, "dog")
WEIGHT_PATH = "inception_v3.h5"

os.makedirs(CAT_DIR, exist_ok=True)
os.makedirs(DOG_DIR, exist_ok=True)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"원본 이미지 폴더를 찾을 수 없습니다: {TRAIN_DIR}")

for file_name in os.listdir(TRAIN_DIR):
    source_path = os.path.join(TRAIN_DIR, file_name)

    if "cat" in file_name:
        target_path = os.path.join(CAT_DIR, file_name)
    elif "dog" in file_name:
        target_path = os.path.join(DOG_DIR, file_name)
    else:
        continue

    # 이미 복사된 파일은 다시 복사하지 않습니다.
    # 매번 25,000장을 다시 복사하면 실행 시간이 오래 걸리기 때문입니다.
    if not os.path.exists(target_path):
        shutil.copyfile(source_path, target_path)

print("cat 이미지 개수:", len(os.listdir(CAT_DIR)))
print("dog 이미지 개수:", len(os.listdir(DOG_DIR)))


# =========================
# 2. Dataset 만들기
# =========================

# image_dataset_from_directory는 폴더명을 보고 라벨을 자동으로 만듭니다.
# dataset/cat 폴더의 이미지는 cat, dataset/dog 폴더의 이미지는 dog으로 인식합니다.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    image_size=(150, 150),
    batch_size=64,
    subset="training",
    validation_split=0.2,
    seed=1234,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    image_size=(150, 150),
    batch_size=64,
    subset="validation",
    validation_split=0.2,
    seed=1234,
)

print(train_ds)


# =========================
# 3. InceptionV3용 전처리
# =========================

# 전처리는 모델마다 맞춰줘야 합니다.
# 이전에 MobileNetV2를 쓸 때는 mobilenet_v2.preprocess_input을 썼지만,
# 지금은 InceptionV3를 쓰므로 inception_v3.preprocess_input을 사용합니다.
#
# 이유:
# 사전학습 모델은 학습 당시 사용한 입력값 형태가 있습니다.
# 그 방식과 비슷하게 이미지를 넣어줘야 가져온 weight를 제대로 활용할 수 있습니다.
def preprocess(image, label):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label


train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)


# =========================
# 4. InceptionV3 모델 가져오기
# =========================

# 1. inception_v3.h5 파일 다운받기
# 강의에서는 아래 weight 파일을 직접 다운로드해서 사용합니다.
# https://github.com/kohpangwei/influence-release/raw/refs/heads/master/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
#
# 이 파일은 용량이 큰 weight 파일이라 GitHub에 안 올렸습니다.
# .gitignore의 *.h5 규칙 때문에 자동으로 제외됩니다.
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError(
        f"weight 파일을 찾을 수 없습니다: {WEIGHT_PATH}\n"
        "inception_v3.h5 파일을 02_image_learning_cnn 폴더 안에 넣고 다시 실행해주세요."
    )

# InceptionV3는 구글이 만든 이미지 분류 모델입니다.
# 이미 ImageNet 같은 큰 이미지 데이터로 학습된 모델이라 이미지 특징을 잘 뽑아냅니다.
#
# input_shape=(150, 150, 3)
#   우리 데이터셋 이미지를 150x150 RGB 이미지로 읽고 있기 때문에 이렇게 맞춥니다.
#   InceptionV3의 기본 입력 크기는 보통 299x299이지만, include_top=False일 때는 다른 크기도 사용할 수 있습니다.
#
# include_top=False
#   InceptionV3의 원래 마지막 분류층을 빼고 가져옵니다.
#   원래 모델은 1000개 클래스를 분류하지만, 우리는 cat/dog 2개만 분류하면 되기 때문입니다.
#
# weights=None
#   여기서는 직접 받은 inception_v3.h5 파일을 load_weights로 넣을 것이기 때문에 None으로 둡니다.
#   자동 다운로드를 쓰고 싶다면 weights="imagenet"으로 바꿀 수도 있습니다.
inception_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None,
)

# 다운받은 weight 파일을 InceptionV3 구조에 적용합니다.
# import InceptionV3는 모델 구조를 가져오는 것이고,
# load_weights는 그 구조 안에 학습된 숫자값(weight)을 채워 넣는 단계입니다.
inception_model.load_weights(WEIGHT_PATH)

# 모델 구조를 확인합니다.
# conv, pooling, concatenate 같은 레이어가 아주 많이 쌓여 있는 것을 볼 수 있습니다.
inception_model.summary()


# =========================
# 5. 사전학습 모델 고정하기
# =========================

# 전이학습의 기본 아이디어는 이미 학습된 모델의 앞부분을 특징 추출기로 쓰는 것입니다.
# 그래서 처음에는 InceptionV3의 weight를 학습으로 바꾸지 않게 고정합니다.
#
# trainable=False
#   이 레이어들은 학습 중 업데이트하지 않겠다는 뜻입니다.
#   즉, 이미 배운 이미지 분석 능력은 그대로 두고 뒤에 붙인 분류층만 학습합니다.
inception_model.trainable = False

# 파인튜닝을 하고 싶을 때만 True로 바꿔서 사용합니다.
# 파인튜닝은 사전학습 모델의 일부 레이어도 내 데이터에 맞게 조금 더 학습시키는 방법입니다.
#
# 처음부터 True로 두면 데이터가 적을 때 학습이 흔들릴 수 있으므로,
# 보통은 먼저 False로 학습해보고 나중에 일부만 풀어줍니다.
USE_FINE_TUNING = True

if USE_FINE_TUNING:
    unfreeze_layers = False

    for layer in inception_model.layers:
        if layer.name == "mixed6":
            unfreeze_layers = True

        # mixed6 이후 레이어만 학습 가능하게 만듭니다.
        # 앞쪽 레이어는 선, 색, 모서리 같은 일반적인 특징을 잡고,
        # 뒤쪽 레이어는 더 구체적인 이미지 특징을 잡는 경우가 많아서 뒤쪽만 살짝 학습시키는 방식입니다.
        if unfreeze_layers:
            layer.trainable = True


# =========================
# 6. InceptionV3 중간 레이어 가져오기
# =========================

# InceptionV3를 처음부터 끝까지 다 쓸 필요는 없습니다.
# 원하는 중간 레이어까지만 잘라서 특징 추출기로 사용할 수도 있습니다.
#
# get_layer("mixed7")
#   InceptionV3 안에서 이름이 mixed7인 레이어를 가져옵니다.
#   mixed7까지의 출력 결과를 가져와서 그 뒤에 내가 만든 Flatten, Dense 층을 붙일 수 있습니다.
#
# 주의:
# final_layer 자체는 레이어 객체입니다.
# 그래서 final_layer.shape는 사용할 수 없고,
# 실제 출력 텐서 모양은 final_layer.output 또는 final_layer.output_shape로 확인합니다.
final_layer = inception_model.get_layer("mixed7")

print(final_layer)
print(final_layer.output) # 출력: Tensor("mixed7/concat:0", shape=(None, 7, 7, 768), dtype=float32)
print(final_layer.output_shape) # 출력: (None, 7, 7, 768)


# =========================
# 7. 내 분류층 붙이기
# =========================

# 여기부터가 내가 직접 붙이는 새 모델의 뒷부분입니다.
# InceptionV3가 이미지 특징을 뽑아주면, 우리는 그 특징을 보고 cat/dog을 판단하는 층만 새로 만듭니다.
#
# Functional API 방식으로 연결하는 이유:
# Sequential처럼 처음부터 끝까지 순서대로 쌓는 것이 아니라,
# inception_model의 중간 출력(final_layer.output)에서 시작해서 새 레이어를 붙이기 때문입니다.
# 이런 경우에는 Functional API가 더 자연스럽습니다.
x = tf.keras.layers.Flatten()(final_layer.output)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# 최종 모델을 만듭니다.
# 입력은 InceptionV3의 입력 이미지를 그대로 사용하고,
# 출력은 내가 새로 붙인 cat/dog 분류층의 출력으로 사용합니다.
model = tf.keras.models.Model(inputs=inception_model.input, outputs=outputs)


# =========================
# 8. 모델 학습
# =========================

# 이진 분류이므로 마지막 출력층은 Dense(1, sigmoid), loss는 binary_crossentropy를 사용합니다.
# optimizer의 learning_rate를 작게 둔 이유는 사전학습 모델을 기반으로 하기 때문에
# 너무 큰 폭으로 weight를 바꾸면 기존에 배운 좋은 특징이 망가질 수 있기 때문입니다.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
)
