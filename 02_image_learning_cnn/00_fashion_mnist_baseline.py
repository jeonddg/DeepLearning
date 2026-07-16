# 뉴럴네트워크에 집어넣을 수 있는 건 무조건 숫자임.. 그럼 사진은 어케 넣냐?
# 이미지는 몇백 몇천 만개의 픽셀로 이루어진 -> 픽셀 마다 숫자가 저장되어있음 숫자는 색상 정보이다. 
# 픽셀에 저장된 숫자 정보를 하나의 노드 하나의 인풋으로 만들면 된다.
# 흑백이 더 쉬움 우선 흑백으로 해보자. 흑백은 0~255까지의 숫자로 이루어져있음. 0은 검정색, 255는 흰색, 중간값은 회색.

# 쇼핑몰 이미지 데이터셋 가져오기
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() # 흑백 이미지 데이터셋 가져오기

print(x_train.shape) # (60000, 28, 28) 6만개의 이미지 데이터셋
print(y_train.shape) # (60000,) 6만개의 라벨 데이터셋

# 라벨 데이터셋은 0~9까지의 숫자로 이루어져있음. 0~9까지의 숫자는 각각의 옷 종류를 의미함.
# y_train[0] # 9 -> 앵클부츠
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

"""
# 이미지를 파이썬으로 띄워보는법

plt.imshow(x_train[1], cmap='gray') # 흑백이미지라 cmap='gray'를 해줘야함
plt.title(class_names[y_train[1]]) # 라벨에 맞는 옷 이름 띄우기
plt.axis('off') # 축 없애기
plt.show() # 이미지 띄우기
"""

# 정규화
x_train = x_train / 255.0 # 0~255까지의 숫자를 0~1까지의 숫자로 바꿔줌
x_test = x_test / 255.0 # 0~255까지의 숫자를 0~1까지의 숫자로 바꿔줌

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)), # 28x28 이미지를 1차원으로 펼치기
    tf.keras.layers.Dense(256, activation='relu'), # 은닉층 256개 노드, 활성화 함수 relu
    tf.keras.layers.Dense(128, activation='relu'), # 은닉층 128개 노드, 활성화 함수 relu
    tf.keras.layers.Dense(10, activation='softmax') # 출력층 10개 노드, 활성화 함수 softmax
])

model.summary() # 모델 구조 확인

# sparse_categorical_crossentropy는 라벨이 원-핫 인코딩이 아닌 정수형일 때 사용
# categorical_crossentropy는 라벨이 원-핫 인코딩일 때 사용
model.compile(optimizer='adam', # 최적화 알고리즘 adam
            loss='sparse_categorical_crossentropy', # 손실 함수 sparse_categorical_crossentropy 
            metrics=['accuracy']) # 평가 지표 accuracy

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

