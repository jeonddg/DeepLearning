"""개/고양이 이미지 분류 프로젝트.

이 파일에서 공부하는 내용
- Kaggle에서 받은 원본 이미지 데이터를 확인한다.
- train 폴더 안에 섞여 있는 cat/dog 이미지를 클래스별 폴더로 나눈다.
- image_dataset_from_directory()로 폴더 구조를 TensorFlow Dataset으로 바꾼다.
- 이미지 픽셀값을 0~1 사이로 정규화한다.
- CNN 모델을 만들어 개/고양이 이미지를 분류한다.

중요한 흐름
Keras의 image_dataset_from_directory()는 폴더 이름을 보고 자동으로 라벨을 만듭니다.
그래서 아래처럼 폴더를 정리해야 합니다.

    dataset/
    ├─ cat/
    │  ├─ cat.0.jpg
    │  └─ ...
    └─ dog/
        ├─ dog.0.jpg
        └─ ...

cat 폴더에 들어간 이미지는 cat 라벨,
dog 폴더에 들어간 이미지는 dog 라벨로 자동 처리됩니다.
"""

import os
import shutil

import tensorflow as tf


# 1. 데이터 경로 정리
# Kaggle에서 받은 원본 데이터는 아래 폴더에 있다고 가정합니다.
#
# train_dir:
# - cat.0.jpg, dog.0.jpg처럼 정답 이름이 파일명에 들어 있는 학습 이미지 폴더입니다.
#
# test_dir:
# - 1.jpg, 2.jpg처럼 정답이 없는 테스트 이미지 폴더입니다.
# - Kaggle 제출용 데이터라 지금 학습에는 사용하지 않습니다.
#
# dataset_dir:
# - 우리가 직접 만들 폴더입니다.
# - image_dataset_from_directory()가 읽기 좋게 cat/dog 폴더로 나눕니다.
base_dir = "data/dogs-vs-cats-redux-kernels-edition"
# join()은 경로를 합쳐주는 함수입니다.
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
dataset_dir = os.path.join(base_dir, "dataset")
cat_dir = os.path.join(dataset_dir, "cat")
dog_dir = os.path.join(dataset_dir, "dog")


# 2. 원본 이미지 개수 확인
# os.listdir(폴더경로)는 해당 폴더 안에 있는 파일/폴더 이름 목록을 가져옵니다.
# len()으로 감싸면 몇 개가 있는지 확인할 수 있습니다.
print("train 이미지 개수:", len(os.listdir(train_dir)))
print("test 이미지 개수:", len(os.listdir(test_dir)))


# 3. cat/dog 폴더 만들기
# image_dataset_from_directory()를 쓰려면 클래스별 폴더가 필요합니다.
# exist_ok=True는 이미 폴더가 있어도 에러를 내지 말라는 뜻입니다.
# data/dataset/cat 폴더 만들기
# data/dataset/dog 폴더 만들기
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)


# 4. train 폴더의 이미지를 cat/dog 폴더로 나누어 복사하기
# 원본 train 폴더에는 cat 이미지와 dog 이미지가 섞여 있습니다.
# 파일 이름에 cat이 들어 있으면 cat 폴더로,
# dog가 들어 있으면 dog 폴더로 복사합니다.
#
# shutil.copyfile(src, dst)
# - src 위치의 파일을 dst 위치로 복사합니다.
#
# os.path.exists(dst)를 확인하는 이유
# - 이미 복사된 파일은 다시 복사하지 않기 위해서입니다.
# - 이전 실행이 중간에 멈췄어도, 다시 실행하면 부족한 파일만 이어서 복사됩니다.
for file_name in os.listdir(train_dir):
    source_path = os.path.join(train_dir, file_name)

    if "cat" in file_name:
        target_path = os.path.join(cat_dir, file_name)
    elif "dog" in file_name:
        target_path = os.path.join(dog_dir, file_name)
    else:
        # 혹시 cat/dog가 아닌 파일이 들어 있으면 건너뜁니다.
        continue

    if not os.path.exists(target_path):
        shutil.copyfile(source_path, target_path)


# 5. 클래스별 이미지 개수 확인
# 정상이라면 cat 12500개, dog 12500개가 나와야 합니다.
print("cat 이미지 개수:", len(os.listdir(cat_dir)))
print("dog 이미지 개수:", len(os.listdir(dog_dir)))


# 6. 폴더를 TensorFlow Dataset으로 바꾸기
# image_dataset_from_directory()는 폴더 구조를 보고 자동으로 이미지와 라벨을 만들어줍니다.
#
# dataset_dir 아래에 cat, dog 폴더가 있으므로
# TensorFlow가 자동으로 2개의 클래스를 인식합니다.
#
# image_size=(64, 64)
# - 모든 이미지를 64x64 크기로 맞춥니다.
# - 실제 사진 크기는 제각각이라 모델에 넣기 전에 같은 크기로 맞춰야 합니다.
#
# batch_size=32
# - 이미지를 32장씩 묶어서 모델에 넣습니다.
#
# validation_split=0.2
# - 전체 데이터 중 20%를 검증 데이터로 사용합니다.
#
# subset="training" / subset="validation"
# - 같은 폴더에서 학습용 데이터와 검증용 데이터를 나누어 가져옵니다.
#
# seed=1234
# - training과 validation이 같은 기준으로 나뉘게 하기 위한 고정값입니다.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=(64, 64),
    batch_size=32,
    subset="training",
    validation_split=0.2,
    seed=1234,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=(64, 64),
    batch_size=32,
    subset="validation",
    validation_split=0.2,
    seed=1234,
)


# 7. Dataset 정보 확인
# len(train_ds)는 이미지 개수가 아니라 batch 개수입니다.
# 예를 들어 batch_size가 32라면, 이미지 32장이 1묶음입니다.
print("train batch 개수:", len(train_ds))
print("validation batch 개수:", len(val_ds))
print("class 이름:", train_ds.class_names)
print(train_ds)
print(val_ds)


# 8. 이미지가 Dataset 안에서 어떤 모양인지 확인하기
# train_ds.take(1)은 batch 하나만 가져옵니다.
# images.shape는 (32, 64, 64, 3) 형태입니다.
# - 32: batch 안 이미지 개수
# - 64: 이미지 세로
# - 64: 이미지 가로
# - 3: RGB 컬러 채널
"""
import matplotlib.pyplot as plt

for images, labels in train_ds.take(1):
    print(images.shape) # 출력은 (32, 64, 64, 3) 형태입니다. batch 안 이미지 개수만큼 이미지가 있습니다.
    print(labels.shape) # 출력은 (32,) 형태입니다. batch 안 이미지 개수만큼 라벨이 있습니다.

    # astype("uint8")는 데이터 타입을 정수형으로 바꿉니다.
    # numpy()는 Tensor를 numpy 배열로 바꿉니다. matplotlib는 numpy 배열을 이미지로 보여줄 수 있습니다.
    plt.imshow(images[0].numpy().astype("uint8")) 

    plt.show()
"""


# 9. 이미지 정규화하기
# image_dataset_from_directory()로 가져온 이미지는 픽셀값이 0~255 범위입니다.
# 모델이 더 안정적으로 학습하도록 255로 나누어 0~1 사이 값으로 바꿉니다.
#
# map()은 Dataset 안의 각 batch에 같은 함수를 적용할 때 사용합니다.
# normalize(images, labels)는 이미지는 정규화하고, 라벨은 그대로 돌려줍니다.
# cast()는 데이터 타입을 바꾸는 함수입니다. images는 tf.uint8 타입이라 나눗셈이 안 되므로 tf.float32로 바꿉니다.
def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)


# 10. CNN 모델 만들기
# 이 모델은 이미지에서 특징을 뽑는 Conv2D 층과,
# 중요한 특징만 남기는 MaxPooling2D 층을 여러 번 사용합니다.
#
# Dropout은 과적합을 줄이기 위해 일부 노드를 랜덤하게 쉬게 하는 층입니다.
# 데이터가 많지 않거나 모델이 너무 잘 외우는 것 같을 때 도움이 됩니다.
model = tf.keras.Sequential([
    # 첫 번째 특징 추출 단계입니다.
    # 입력 이미지는 64x64 크기의 RGB 이미지라 input_shape=(64, 64, 3)입니다.
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        input_shape=(64, 64, 3),
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 두 번째 특징 추출 단계입니다.
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),

    # 세 번째 특징 추출 단계입니다.
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Conv2D 결과는 아직 2차원 특징맵이므로 Dense 층에 넣기 위해 1차원으로 펼칩니다.
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    # 개/고양이는 정답이 2종류입니다.
    # 출력 노드를 1개로 두고 sigmoid를 쓰면 0~1 사이 확률 하나를 출력합니다.
    # 보통 0.5 이상이면 한 클래스, 0.5 미만이면 다른 클래스로 해석합니다.
    tf.keras.layers.Dense(1, activation="sigmoid"),
])


# 11. 모델 구조 확인하기
model.summary()


# 12. 모델 학습 설정하기
# binary_crossentropy는 정답이 2종류인 이진 분류 문제에서 자주 쓰는 loss입니다.
# accuracy는 전체 중 몇 개를 맞혔는지 보여주는 지표입니다.
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# 13. 모델 학습하기
# train_ds로 학습하고 val_ds로 검증합니다.
# validation accuracy가 train accuracy보다 많이 낮으면 과적합을 의심할 수 있습니다.
model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
)

model.save("model/cat_dog_model.keras")

# 핵심 정리
# - Kaggle 원본 train 폴더에는 cat/dog 이미지가 섞여 있습니다.
# - image_dataset_from_directory()를 쓰려면 클래스별 폴더 구조가 필요합니다.
# - 그래서 dataset/cat, dataset/dog 폴더로 이미지를 나누어 복사했습니다.
# - 이미지 크기는 64x64로 맞추고, 픽셀값은 0~1로 정규화했습니다.
# - Conv2D는 이미지 특징을 뽑고, MaxPooling2D는 중요한 특징만 남기며 크기를 줄입니다.
# - Dropout은 과적합을 줄이기 위해 사용했습니다.

# 정확도는 현재 아무리 높아도 85 이상으로 올리기 쉽지가 않다.
# 정확도를 더 높이려면 데이터의 퀄리티를 높여야 되는데 
# 데이터의 양을 늘리거나, 데이터의 질을 늘리거나.
