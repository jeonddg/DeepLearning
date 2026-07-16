"""대학원 합격 예측 v2: 기존 코드에 전처리를 추가해서 성능 개선하기.

이 파일은 baseline 코드보다 조금 더 성능이 잘 나오도록 고친 버전입니다.
하지만 일부러 어렵게 함수로 많이 나누거나, Path 같은 아직 안 배운 문법은 사용하지 않았습니다.

baseline에서 개선한 점
1. GRE, GPA 숫자 범위를 비슷하게 맞춥니다. 이것을 정규화라고 합니다.
2. rank를 숫자 하나로 넣지 않고 원-핫 인코딩으로 바꿉니다.
3. 학습 데이터와 테스트 데이터를 나눠서 진짜 성능을 확인합니다.
4. 데이터가 적기 때문에 모델 크기를 조금 줄입니다.
5. EarlyStopping으로 성능이 더 좋아지지 않으면 학습을 일찍 멈춥니다.
"""

import numpy as np
import pandas as pd
import tensorflow as tf


# 실행할 때마다 결과가 너무 많이 달라지지 않도록 랜덤 기준을 고정합니다.
# 딥러닝은 내부적으로 랜덤하게 시작하는 값이 있어서 실행할 때마다 결과가 조금씩 달라질 수 있습니다.
tf.random.set_seed(42)
np.random.seed(42)


# 1. 데이터 불러오기
# gpascore.csv는 이 파이썬 파일과 같은 폴더에서 실행한다고 가정합니다.
# 그래서 기존 코드처럼 파일 이름만 적었습니다.
data = pd.read_csv("gpascore.csv")

print("\n[데이터 앞부분 확인]")
print(data.head())


# 2. 결측치 처리
# 결측치는 비어 있는 값입니다.
# 모델은 비어 있는 값을 그대로 학습할 수 없기 때문에 처리해야 합니다.
print("\n[결측치 확인]")
print(data.isnull().sum())

# 이 데이터는 gre에 결측치가 1개 있습니다.
# 데이터가 약 400개 정도라서, 입문 단계에서는 비어 있는 행을 제거하는 방식이 가장 단순합니다.
data = data.dropna()


# 3. 데이터 섞기
# 데이터가 특정 순서로 정렬되어 있을 수도 있으므로 먼저 섞어줍니다.
# frac=1은 전체 데이터를 100% 다시 섞는다는 뜻입니다.
# random_state=42는 매번 같은 방식으로 섞이게 만드는 값입니다.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# 4. 입력 데이터와 정답 데이터 분리
# y는 모델이 맞혀야 하는 정답입니다.
# 여기서는 admit, 즉 합격 여부입니다.
y = data["admit"].to_numpy(dtype=np.float32)


# 5. GRE, GPA 정규화
# baseline에서는 gre, gpa, rank를 그대로 넣었습니다.
# 그런데 gre는 400~800, gpa는 2~4 정도라 숫자 범위 차이가 큽니다.
# 숫자 범위가 너무 다르면 모델이 gre를 훨씬 더 큰 값으로 받아들여 학습이 불안정해질 수 있습니다.
#
# 정규화(표준화) 공식:
#     정규화된 값 = (원래 값 - 평균) / 표준편차
#
# 이렇게 하면 gre와 gpa가 비슷한 크기의 숫자로 바뀝니다.
numeric_columns = ["gre", "gpa"]

mean = data[numeric_columns].mean()
std = data[numeric_columns].std()

x_numeric = (data[numeric_columns] - mean) / std


# 6. rank 원-핫 인코딩
# rank는 1, 2, 3, 4로 되어 있습니다.
# 숫자가 낮을수록 좋은 학교 등급이라는 의미는 있지만,
# rank=4가 rank=2보다 정확히 2배 나쁘다는 뜻은 아닙니다.
#
# 그런데 숫자 그대로 넣으면 모델은 rank를 일반 숫자처럼 받아들일 수 있습니다.
# 그래서 원-핫 인코딩을 사용합니다.
#
# 예시:
# rank가 1이면 [1, 0, 0, 0]
# rank가 2이면 [0, 1, 0, 0]
# rank가 3이면 [0, 0, 1, 0]
# rank가 4이면 [0, 0, 0, 1]
# .get_dummies()는 pandas에서 원-핫 인코딩을 쉽게 해주는 함수입니다.
# rank 컬럼을 int로 바꾼 뒤, prefix="rank"를 주면 rank_1, rank_2, rank_3, rank_4라는 이름으로 컬럼이 만들어집니다.
x_rank = pd.get_dummies(data["rank"].astype(int), prefix="rank")


# 7. 최종 입력 데이터 만들기
# 정규화된 gre/gpa와 원-핫 인코딩된 rank를 옆으로 붙입니다.
# 모델은 이 데이터를 보고 합격 여부를 예측합니다.
# axis=1은 옆으로 붙이라는 뜻입니다.
x = pd.concat([x_numeric, x_rank], axis=1).to_numpy(dtype=np.float32)

print("\n[전처리 후 입력 데이터 확인]")
print("입력 컬럼:", list(pd.concat([x_numeric, x_rank], axis=1).columns))
print("x 모양:", x.shape)
print("앞 5개 입력:\n", x[:5])


# 8. 학습 데이터와 테스트 데이터 나누기
# baseline에서는 전체 데이터로 학습만 했습니다.
# 그러면 모델이 진짜 잘하는지, 그냥 학습 데이터를 외운 건지 알기 어렵습니다.
#
# 여기서는 80%는 학습용, 20%는 테스트용으로 나눕니다.
# 테스트 데이터는 모델이 학습 중 보지 않은 데이터라서 더 현실적인 성능 확인에 사용합니다.
train_size = int(len(x) * 0.8)

x_train = x[:train_size]
y_train = y[:train_size]

x_test = x[train_size:]
y_test = y[train_size:]

print("\n[데이터 분리]")
print("학습 데이터:", x_train.shape, y_train.shape)
print("테스트 데이터:", x_test.shape, y_test.shape)


# 9. 모델 만들기
# baseline에서는 Dense(64), Dense(128)을 사용했습니다.
# 데이터가 약 400개라 모델이 너무 크면 학습 데이터에만 과하게 맞을 수 있습니다.
#
# 그래서 v2에서는 Dense(16), Dense(8)로 줄였습니다.
# 모델을 작게 만든다고 항상 좋은 것은 아니지만,
# 데이터가 적을 때는 너무 큰 모델보다 작은 모델이 안정적인 경우가 많습니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])


# 10. 모델 학습 설정
# binary_crossentropy:
# admit이 0 또는 1인 이진 분류 문제라서 사용합니다.
#
# accuracy:
# 전체 중 몇 개를 맞혔는지 확인하는 지표입니다.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# 11. EarlyStopping 설정
# epochs를 크게 줘도, 성능이 더 좋아지지 않으면 자동으로 멈추게 합니다.
#
# monitor="val_loss": 검증 데이터의 loss를 지켜봅니다.
# patience=20: 20번 동안 좋아지지 않으면 멈춥니다.
# restore_best_weights=True: 가장 좋았던 순간의 모델 상태로 되돌립니다.
# callbacks는 모델 학습 중간에 특정 이벤트가 발생했을 때 호출되는 함수입니다.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)


# 12. 모델 학습
# validation_split=0.2는 학습 데이터 중 20%를 검증용으로 잠깐 떼어낸다는 뜻입니다.
# 이 검증 데이터는 EarlyStopping이 학습을 멈출지 판단할 때 사용합니다.
#
# verbose=0은 학습 로그를 숨기는 옵션입니다.
# 출력이 너무 많아지는 것을 막고, 마지막 결과만 보기 위해 사용했습니다.
history = model.fit(
    x_train,
    y_train,
    epochs=300,
    batch_size=16, # batch_size = 한 번에 몇 개씩 묶어서 학습할지 정하는 값
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0,
)


# 13. 성능 확인
# 학습 데이터 성능과 테스트 데이터 성능을 둘 다 봅니다.
#
# 학습 데이터 accuracy가 높고 테스트 데이터 accuracy가 낮으면
# 모델이 학습 데이터를 외웠을 가능성이 있습니다.
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n[학습 결과]")
print(f"실제로 학습한 epoch 수: {len(history.history['loss'])}")
print(f"학습 데이터 loss: {train_loss:.4f}")
print(f"학습 데이터 accuracy: {train_accuracy:.4f}")
print(f"테스트 데이터 loss: {test_loss:.4f}")
print(f"테스트 데이터 accuracy: {test_accuracy:.4f}")


# 14. 새로운 지원자 예측하기
# 주의할 점:
# 학습할 때 gre/gpa를 정규화하고 rank를 원-핫 인코딩했으므로,
# 새 데이터를 예측할 때도 똑같이 전처리해야 합니다.
new_people = pd.DataFrame(
    [
        [750, 3.70, 3],
        [400, 2.20, 1],
    ],
    columns=["gre", "gpa", "rank"],
)

# 새 지원자의 gre/gpa도 위에서 구한 평균과 표준편차로 정규화합니다.
new_numeric = (new_people[numeric_columns] - mean) / std

# 새 지원자의 rank도 원-핫 인코딩합니다.
new_rank = pd.get_dummies(new_people["rank"].astype(int), prefix="rank")

# 학습 데이터에는 rank_1~rank_4가 있었는데,
# 새 지원자 데이터에는 일부 rank만 있을 수 있습니다.
# 그래서 reindex로 컬럼을 학습 때와 똑같이 맞춥니다.
new_rank = new_rank.reindex(columns=x_rank.columns, fill_value=0)

new_x = pd.concat([new_numeric, new_rank], axis=1).to_numpy(dtype=np.float32)

predictions = model.predict(new_x, verbose=0)

print("\n[새 지원자 예측]")
for person, prediction in zip(new_people.to_numpy(), predictions):
    probability = prediction[0]
    result = "합격 가능성이 더 높음" if probability >= 0.5 else "불합격 가능성이 더 높음"
    print(
        f"GRE {person[0]:.0f}, GPA {person[1]:.2f}, rank {person[2]:.0f} "
        f"-> 합격 확률 {probability:.2%} ({result})"
    )


# 핵심 정리
# - 정규화는 입력값들의 숫자 범위를 비슷하게 맞춰 학습을 안정적으로 만듭니다.
# - 원-핫 인코딩은 rank 같은 등급 데이터를 단순 숫자가 아닌 카테고리처럼 표현합니다.
# - train/test 분리는 모델이 처음 보는 데이터도 잘 맞히는지 확인하기 위해 필요합니다.
# - EarlyStopping은 의미 없이 오래 학습하거나 과적합되는 것을 줄이는 데 도움이 됩니다.
# - 그래도 이 데이터는 작고 입력 정보가 적어서 성능에는 한계가 있습니다.
