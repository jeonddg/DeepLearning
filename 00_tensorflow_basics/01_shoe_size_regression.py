"""TensorFlow 빠른 기초 2: 키로 신발 사이즈 예측하기.

목표
- 사람들의 키와 신발 사이즈 데이터를 준비한다.
- 신발 사이즈 = 키 * weight + bias 라는 아주 단순한 모델을 만든다.
- TensorFlow가 weight와 bias를 조금씩 고쳐가며 정답에 가까워지는 과정을 본다.

이 예제는 엄밀한 의미의 복잡한 딥러닝 모델은 아닙니다.
하지만 딥러닝 학습의 핵심 흐름인
"예측 -> 오차 계산 -> 기울기 계산 -> 변수 업데이트"를 가장 작게 보여줍니다.
"""

import tensorflow as tf


def predict_shoe_size(heights, weight, bias):
    """현재 weight와 bias를 사용해서 신발 사이즈를 예측합니다.

    수식:
        예측 신발 사이즈 = 키 * weight + bias

    여기서 weight와 bias는 사람이 직접 정하는 값이 아니라,
    학습을 통해 TensorFlow가 점점 더 좋은 값으로 바꿔갑니다.
    """
    return heights * weight + bias


def mean_squared_error(predictions, answers):
    """예측값과 실제값의 차이를 하나의 숫자로 바꿉니다.

    예측값 - 실제값:
        얼마나 틀렸는지 확인합니다.

    square:
        음수 오차와 양수 오차가 서로 상쇄되지 않게 제곱합니다.

    reduce_mean:
        여러 사람의 오차를 평균내서 loss 하나로 만듭니다.
    """
    return tf.reduce_mean(tf.square(predictions - answers))


# 1. 학습 데이터 준비
# TensorFlow는 보통 숫자 데이터를 Tensor 형태로 다룹니다.
# dtype=tf.float32를 쓰는 이유는 딥러닝 계산에서 실수를 많이 쓰기 때문입니다.
heights = tf.constant([150, 160, 170, 180, 190], dtype=tf.float32)
shoe_sizes = tf.constant([250, 260, 270, 280, 290], dtype=tf.float32)


# 2. 학습할 변수 만들기
# 우리가 찾고 싶은 식은 "신발 사이즈 = 키 * weight + bias" 입니다.
#
# weight:
#   키가 1cm 커질 때 신발 사이즈가 얼마나 커지는지에 해당합니다.
#
# bias:
#   전체 예측값을 위아래로 움직이는 기본값입니다.
#
# tf.Variable을 쓰는 이유:
#   학습 과정에서 값이 계속 바뀌어야 하기 때문입니다.
weight = tf.Variable(0.1)
bias = tf.Variable(0.0)


# 3. optimizer 준비
# optimizer는 loss를 줄이는 방향으로 weight와 bias를 업데이트합니다.
# Adam은 자주 쓰이는 optimizer이며, 초보 실습에서도 안정적으로 잘 동작합니다.
#
# learning_rate:
#   한 번 업데이트할 때 얼마나 크게 움직일지 정합니다.
#   너무 크면 학습이 튈 수 있고, 너무 작으면 학습이 느립니다.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# 4. 학습 반복
# 한 번에 완벽한 weight와 bias를 찾기는 어렵습니다.
# 그래서 여러 번 반복하면서 조금씩 좋은 값으로 바꿉니다.
epochs = 3000

for epoch in range(epochs):
    # GradientTape는 loss가 weight와 bias에 따라 어떻게 변하는지 기록합니다.
    # 이 기록을 이용해서 기울기, 즉 gradient를 계산할 수 있습니다.
    with tf.GradientTape() as tape:
        predictions = predict_shoe_size(heights, weight, bias)
        loss = mean_squared_error(predictions, shoe_sizes)

    # gradient는 "loss를 줄이려면 각 변수를 어느 방향으로 바꿔야 하는지"입니다.
    gradients = tape.gradient(loss, [weight, bias])

    # optimizer가 gradient를 보고 weight와 bias를 실제로 업데이트합니다.
    optimizer.apply_gradients(zip(gradients, [weight, bias]))

    # 매번 출력하면 너무 길어지기 때문에 300번마다 학습 상태를 확인합니다.
    if epoch % 300 == 0:
        print(
            f"{epoch:4d}번째 학습 | "
            f"loss: {loss.numpy():9.4f} | "
            f"weight: {weight.numpy():7.4f} | "
            f"bias: {bias.numpy():7.4f}"
        )


# 5. 학습 결과 확인
print("\n[학습 완료]")
print(f"최종 weight: {weight.numpy():.4f}")
print(f"최종 bias: {bias.numpy():.4f}")


# 6. 새로운 키로 예측하기
# 학습에 사용하지 않은 키를 넣어서 신발 사이즈를 예측해봅니다.
new_height = tf.constant(175, dtype=tf.float32)
predicted_size = predict_shoe_size(new_height, weight, bias)

print("\n[새로운 예측]")
print(f"키 {new_height.numpy():.0f}cm인 사람의 예상 신발 사이즈: {predicted_size.numpy():.2f}")


# 7. 학습에 사용한 5개 데이터도 예측해보기
# 위에서는 175cm 한 명만 예측했습니다.
# 이번에는 학습 데이터였던 150, 160, 170, 180, 190cm 각각에 대해
# 모델이 예측한 신발 사이즈를 실제값과 비교해서 보여줍니다.
trained_predictions = predict_shoe_size(heights, weight, bias)

print("\n[학습 데이터 5개 예측]")
print("키(cm) | 실제 신발 | 예측 신발")
print("-" * 32)

for height, real_size, predicted_size in zip(
    heights.numpy(),
    shoe_sizes.numpy(),
    trained_predictions.numpy(),
):
    print(f"{height:6.0f} | {real_size:9.0f} | {predicted_size:9.2f}")


# 핵심 정리
# - 데이터: 정답을 알려주는 예시입니다.
# - 모델: 입력을 받아 예측값을 만드는 식입니다.
# - loss: 예측이 얼마나 틀렸는지 나타내는 숫자입니다.
# - gradient: loss를 줄이기 위해 변수를 어느 방향으로 바꿀지 알려줍니다.
# - optimizer: gradient를 사용해서 변수를 실제로 업데이트합니다.

