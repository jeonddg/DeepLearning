# Deep Learning Study

딥러닝을 공부하면서 정리한 개념, 실습 코드, 미니 프로젝트를 모아두는 저장소이다.

이 저장소는 완성된 대형 프로젝트를 올리는 공간이라기보다, TensorFlow와 딥러닝 개념을 하나씩 배우면서 실습한 내용을 기록하는 학습용 레포지토리이다. 큰 규모의 딥러닝 프로젝트나 별도로 관리할 만한 프로젝트는 다른 저장소에서 따로 관리할 예정이다.

## 목적

- TensorFlow 기본 문법을 익힌다.
- 딥러닝 모델이 학습되는 전체 흐름을 이해한다.
- 이미지 분류, CNN, RNN 같은 핵심 개념을 작은 예제로 직접 실습한다.
- 공부하면서 헷갈렸던 부분을 주석과 Markdown 문서로 정리한다.
- 미니 프로젝트를 통해 데이터 준비, 전처리, 모델 구성, 학습, 평가 과정을 연습한다.

## 폴더 구조

```text
DL/
├─ 00_tensorflow_basics/
│  ├─ 00_tensorflow_basics.py
│  └─ 01_shoe_size_regression.py
│
├─ 01_graduate_admission_prediction/
│  ├─ data/
│  ├─ 00_admission_prediction_baseline.py
│  └─ 01_admission_prediction_v2.py
│
├─ 02_image_learning_cnn/
│  ├─ 00_fashion_mnist_baseline.py
│  ├─ 01_convolutional_layer_fashion_mnist.py
│  ├─ 02_cat_dog_classification.py
│  ├─ 03_save_model.py
│  ├─ 04_earlystopping.py
│  ├─ 05_functional_api.py
│  └─ 06_transfer_learning.py
│
├─ 03_sequence_learning_and_rnn/
│  ├─ 01_순서가 중요한 데이터는 어떻게 학습할까 (RNN 개념).md
│  └─ 02_LSTM_GRU_model.md
│
├─ notes/
├─ .gitignore
└─ README.md
```

## 학습 내용

## 00. TensorFlow Basics

TensorFlow의 기본 개념과 텐서, 변수, 간단한 학습 과정을 정리한다.

- TensorFlow 기본 문법
- Tensor와 Variable
- loss 계산
- GradientTape
- optimizer를 이용한 변수 업데이트
- 키를 이용한 신발 사이즈 예측 예제

## 01. Graduate Admission Prediction

대학원 합격 여부를 예측하는 미니 프로젝트이다.

작은 tabular 데이터를 사용해서 딥러닝 모델을 만들고, 성능이 잘 나오지 않는 이유도 함께 정리한다.

- CSV 데이터 불러오기
- 입력값과 정답 분리
- 모델 구성
- 학습과 예측
- 입력값 스케일 차이 문제
- 정규화와 전처리의 필요성
- 작은 데이터셋에서 딥러닝 모델이 흔들릴 수 있는 이유

## 02. Image Learning and CNN

이미지 데이터를 다루는 방법과 CNN 관련 실습을 정리한다.

- Fashion MNIST 이미지 분류
- Flatten 기반 모델과 CNN 모델 비교
- Convolutional Layer
- MaxPooling
- 고양이/강아지 이미지 분류
- Dataset 만들기
- 모델 저장과 불러오기
- TensorBoard
- EarlyStopping
- Functional API
- Transfer Learning

## 03. Sequence Learning and RNN

순서가 중요한 데이터를 학습하는 방법을 정리한다.

- Sequence 데이터 개념
- RNN이 필요한 이유
- Hidden State
- Sequence to Vector
- Vector to Sequence
- Sequence to Sequence
- Simple RNN의 한계
- LSTM과 GRU 개념

## 사용 환경

주로 아래 환경에서 실습한다.

```text
Python
TensorFlow
pandas
numpy
matplotlib
```

가상환경 폴더와 데이터셋, 모델 파일, 로그 파일은 GitHub에 올리지 않는다.

예시:

```text
.venv/
*.h5
*.keras
logs/
02_image_learning_cnn/data/
```

## 실행 방법

가상환경을 활성화한 뒤 원하는 파일을 실행한다.

```powershell
.\.venv\Scripts\Activate.ps1
python 00_tensorflow_basics\01_shoe_size_regression.py
```

폴더 안으로 이동해서 실행할 수도 있다.

```powershell
cd 02_image_learning_cnn
python 00_fashion_mnist_baseline.py
```

## 메모

이 저장소의 코드는 학습 목적으로 작성되었다. 그래서 실무용으로 최대한 짧고 완성도 있게 만든 코드보다는, 처음 보는 사람도 흐름을 이해할 수 있도록 주석과 설명을 많이 남기는 방향으로 정리한다.

성능을 무조건 높이는 것보다 다음을 우선한다.

- 코드가 왜 필요한지 이해하기
- 모델이 어떤 순서로 학습되는지 이해하기
- 에러가 났을 때 원인을 추적해보기
- 같은 코드를 조금씩 바꿔보며 결과 비교하기

