# convolutional layer를 사용한 CNN 모델 -> feature extraction(특징 추출) 가능

# 중요한 걸 뽑아서 압축 후 새롭게 만들고 계속 압축하면서 새로운 특징을 뽑아내는 구조 -> kernel(필터)로 이미지의 특징을 뽑아내는 구조
# kernel(필터)로 이미지의 특징을 뽑아내는 구조 -> convolutional layer
# convolutional layer를 여러 개 쌓으면 이미지의 특징을 점점 더 추상화해서 뽑아낼 수 있음
# kernel을 어떤식으로 디자인 하냐에 따라서 제각각의 특징을 뽑아낼 수 있음 -> kernel을 여러 개 만들어서 다양한 특징을 뽑아낼 수 있음
# sharpen kernel, blur kernel, edge detection kernel, gaussian kernel, median kernel 등등 여러개의 커널이 존재한다.
# pooling layer는 convolutional layer에서 뽑아낸 특징을 압축하는 역할을 함 -> 특징을 압축해서 중요한 특징만 남기고 나머지는 버리는 역할 ex) max pooling, average pooling, global average pooling 등등 여러개의 pooling layer가 존재한다.

# 특징추출(convolutional layer) + 특징을 모아줌(pooling layer) -> 이미지가 어디에 있든 잘 인식할 수 있음 -> translation invariance

