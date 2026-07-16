"""TensorFlow 기초 정리

이 파일은 TensorFlow에서 가장 자주 만나는 기본 개념을
작은 예제로 확인할 수 있게 정리한 실습 노트입니다.
"""

import tensorflow as tf


def show(title, value):
    """출력 결과를 보기 쉽게 구분해서 보여줍니다."""
    print(f"\n[{title}]")
    print(value)


# 1. Tensor 만들기
# Tensor는 숫자 데이터를 담는 다차원 배열입니다.
# 0차원: 스칼라, 1차원: 벡터, 2차원: 행렬, 3차원 이상: 고차원 텐서
scalar = tf.constant(7)
vector = tf.constant([4, 5], dtype=tf.float32)
matrix = tf.constant(
    [
        [1, 2],
        [3, 4],
    ],
    dtype=tf.float32,
)

show("스칼라", scalar)
show("벡터", vector)
show("행렬", matrix)


# 2. Tensor 정보 확인하기
# shape: 텐서의 모양
# dtype: 텐서 안에 들어있는 값의 자료형
show("벡터 shape", vector.shape)
show("행렬 shape", matrix.shape)
show("행렬 dtype", matrix.dtype)


# 3. 기본 연산
another_vector = tf.constant([3, 4], dtype=tf.float32)

show("덧셈", tf.add(vector, another_vector))
show("뺄셈", tf.subtract(vector, another_vector))
show("곱셈", tf.multiply(vector, another_vector))
show("나눗셈", tf.divide(vector, another_vector))


# 4. 행렬 곱
# tf.matmul은 일반 곱셈이 아니라 행렬 곱셈입니다.
show("행렬 곱", tf.matmul(matrix, matrix))


# 5. 0으로 채워진 Tensor 만들기
# [2, 2, 3]은 2 x 2 x 3 모양의 텐서를 뜻합니다.
zeros = tf.zeros([2, 2, 3])

show("0으로 채운 텐서", zeros)
show("0으로 채운 텐서 shape", zeros.shape)


# 6. 자료형 바꾸기
# tf.cast를 사용하면 dtype을 바꿀 수 있습니다.
float64_vector = tf.cast(vector, tf.float64)

show("float64로 변환한 벡터", float64_vector)
show("변환 후 dtype", float64_vector.dtype)


# 7. Variable 사용하기
# constant는 한 번 만들면 값을 바꾸기 어렵고,
# Variable은 학습 과정에서 계속 업데이트되는 값을 저장할 때 사용합니다.
weights = tf.Variable([1.0, 2.0, 3.0])

show("처음 weights", weights)
show("numpy 배열로 변환", weights.numpy())

weights.assign([4.0, 5.0, 6.0])
show("assign으로 값 변경 후 weights", weights)


# 정리
# - tf.constant: 바뀌지 않는 Tensor
# - tf.Variable: 바뀔 수 있는 Tensor, 모델의 가중치에 자주 사용
# - shape: Tensor의 모양
# - dtype: Tensor 값의 자료형
# - tf.cast: Tensor 자료형 변환
# - tf.matmul: 행렬 곱셈
