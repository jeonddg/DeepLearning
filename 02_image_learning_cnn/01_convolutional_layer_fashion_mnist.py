# convolutional layer를 사용한 CNN 모델 -> feature extraction(특징 추출) 가능

# 중요한 걸 뽑아서 압축 후 새롭게 만들고 계속 압축하면서 새로운 특징을 뽑아내는 구조 -> kernel(필터)로 이미지의 특징을 뽑아내는 구조
# kernel(필터)로 이미지의 특징을 뽑아내는 구조 -> convolutional layer
# convolutional layer를 여러 개 쌓으면 이미지의 특징을 점점 더 추상화해서 뽑아낼 수 있음
# kernel을 어떤식으로 디자인 하냐에 따라서 제각각의 특징을 뽑아낼 수 있음 -> kernel을 여러 개 만들어서 다양한 특징을 뽑아낼 수 있음
# sharpen kernel, blur kernel, edge detection kernel, gaussian kernel, median kernel 등등 여러개의 커널이 존재한다.
# pooling layer는 convolutional layer에서 뽑아낸 특징을 압축하는 역할을 함 -> 특징을 압축해서 중요한 특징만 남기고 나머지는 버리는 역할 ex) max pooling, average pooling, global average pooling 등등 여러개의 pooling layer가 존재한다.

# 특징추출(convolutional layer) + 특징을 모아줌(pooling layer) -> 이미지가 어디에 있든 잘 인식할 수 있음 -> translation invariance

import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

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
    "Ankle boot"
]

# 정규화
x_train = x_train / 255.0 # 0~255까지의 숫자를 0~1까지의 숫자로 바꿔줌
x_test = x_test / 255.0 # 0~255까지의 숫자를 0~1까지의 숫자로 바꿔줌  

# 전처리
# CNN의 Conv2D는 보통 입력을 이렇게 받음. -> (이미지 개수, 세로, 가로, 채널)
x_train = x_train.reshape(-1, 28, 28, 1) # 마지막 1은 흑백 이미지 채널 1개
x_test = x_test.reshape(-1, 28, 28, 1) 

model = tf.keras.Sequential([
    # Conv2D는 4차원 입력을 받음. -> (이미지 개수, 세로, 가로, 채널)
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)), # convolutional layer 32개 필터, 3x3 커널, 활성화 함수 relu, padding='same'로 이미지 크기 유지, input_shape=(28, 28, 1)로 입력 이미지 크기 지정
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # 28x28 이미지를 1차원으로 펼치기
    tf.keras.layers.Dense(256, activation='relu'), # 은닉층 256개 노드, 활성화 함수 relu
    tf.keras.layers.Dense(128, activation='relu'), # 은닉층 128개 노드, 활성화 함수 relu
    tf.keras.layers.Dense(10, activation='softmax') # 출력층 10개 노드, 활성화 함수 softmax
])

model.summary() # 모델 구조 확인

model.compile(optimizer='adam', # 최적화 알고리즘 adam
              loss='sparse_categorical_crossentropy', # 손실 함수 sparse_categorical_crossentropy
              metrics=['accuracy']) # 평가 지표 accuracy  

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)