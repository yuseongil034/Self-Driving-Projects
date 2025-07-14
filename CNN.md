# CNN (Convolutional Neural Network) 완벽 가이드

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Latest-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green.svg" alt="Level">
</div>

## 📖 목차

1. [CNN이란?](#1-cnn이란)
2. [CNN의 기본 구성요소](#2-cnn의-기본-구성요소)
3. [각 구성요소의 작동 원리](#3-각-구성요소의-작동-원리)
4. [주요 용어 정리](#4-주요-용어-정리)
5. [CNN의 학습 과정](#5-cnn의-학습-과정)
6. [코드 예제](#6-코드-예제)
7. [실전 팁](#7-실전-팁)
8. [실전 응용](#8-실전-응용)
9. [참고자료](#9-참고자료)

---

## 1. CNN이란?

**CNN (Convolutional Neural Network)**은 이미지와 같은 격자 구조의 데이터를 처리하는 데 특화된 딥러닝 모델입니다.

### 🔍 주요 특징

- **공간적 구조 보존**: 이미지의 2D 구조를 유지하면서 학습
- **파라미터 공유**: 같은 필터를 이미지 전체에 적용하여 효율성 증대
- **평행 이동 불변성**: 객체의 위치가 바뀌어도 같은 특징을 감지
- **계층적 특징 추출**: 저수준 특징부터 고수준 특징까지 점진적 학습

### 🆚 일반 신경망과의 차이점

| 특성        | 일반 신경망 | CNN             |
|-------------|-------------|-----------------|
| 입력 형태   | 1D 벡터     | 2D/3D 텐서      |
| 연결 방식   | 완전 연결   | 지역적 연결     |
| 파라미터 수 | 많음        | 적음 (공유)     |
| 공간 정보   | 손실        | 보존            |

---

## 2. CNN의 기본 구성요소

CNN은 다음과 같은 핵심 구성요소들로 이루어져 있습니다:

입력 이미지 → [합성곱층] → [활성화 함수] → [풀링층] → ... → [완전연결층] → 출력

### 🔧 주요 구성요소

1. **합성곱층 (Convolutional Layer)**: 특징 추출
2. **활성화 함수 (Activation Function)**: 비선형성 추가
3. **풀링층 (Pooling Layer)**: 차원 축소 및 불변성 제공
4. **완전연결층 (Fully Connected Layer)**: 최종 분류

---

## 3. 각 구성요소의 작동 원리

### 3.1 합성곱층 (Convolutional Layer)

합성곱층은 입력 이미지에 **필터(커널)**를 적용하여 특징맵을 생성합니다.

#### 📐 합성곱 연산 수식

\[
(f * g)(x, y) = \sum\sum f(i, j) \cdot g(x-i, y-j)
\]

#### 🔢 출력 크기 계산

\[
\text{출력 크기} = \frac{\text{입력 크기} - \text{필터 크기} + 2 \times \text{패딩}}{\text{스트라이드}} + 1
\]

### 3.2 활성화 함수 (Activation Function)

CNN에서 주로 사용되는 활성화 함수들:

- **ReLU**: \( f(x) = \max(0, x) \)
- **Leaky ReLU**: \( f(x) = \max(0.01x, x) \)

### 3.3 풀링층 (Pooling Layer)

특징맵의 크기를 줄이고 중요한 정보만 추출합니다.

- **최대 풀링 (Max Pooling)**: \( f(x) = \max(x_i) \) (pooling window 내)
- **평균 풀링 (Average Pooling)**: \( f(x) = \frac{1}{n} \sum x_i \) (pooling window 내)

### 3.4 완전연결층 (Fully Connected Layer)

\[
y = W \cdot x + b
\]

---

## 4. 주요 용어 정리

### 📚 핵심 용어

| 용어                | 설명                       | 예시                  |
|---------------------|----------------------------|-----------------------|
| **Filter (필터)**   | 특징을 추출하는 작은 행렬  | 3×3, 5×5 크기         |
| **Kernel (커널)**   | 필터와 동일한 의미         | 동일                  |
| **Stride (스트라이드)** | 필터가 이동하는 간격      | 1, 2, 3...            |
| **Padding (패딩)**  | 입력 주변에 추가하는 값    | 0으로 채우기          |
| **Feature Map**     | 합성곱 연산의 출력         | 특징이 추출된 맵      |
| **Receptive Field** | 한 뉴런이 보는 입력 영역   | 필터 크기에 따라 결정 |
| **Depth (깊이)**    | 특징맵의 채널 수           | RGB = 3, 그레이스케일 = 1 |

### 🎯 패딩 타입

- **Valid Padding**: 패딩 없음
- **Same Padding**: 출력 크기 = 입력 크기
- **Full Padding**: 필터 크기 - 1만큼 패딩

---

## 5. CNN의 학습 과정

### 5.1 순전파 (Forward Propagation)

1. 입력 이미지 → 합성곱층 → 특징맵 생성
2. 특징맵 → 활성화 함수 → 비선형 변환
3. 활성화된 특징맵 → 풀링층 → 다운샘플링
4. 반복 후 → 완전연결층 → 최종 예측

### 5.2 역전파 (Backpropagation)

1. 손실 함수 계산: \( L = \text{CrossEntropy}(y_{\text{pred}}, y_{\text{true}}) \)
2. 그래디언트 계산: \( \frac{\partial L}{\partial W}, \frac{\partial L}{\partial b} \)
3. 가중치 업데이트: \( W = W - \alpha \cdot \frac{\partial L}{\partial W} \)

### 5.3 최적화 알고리즘

- **SGD**: \( W = W - \alpha \cdot \nabla W \)
- **Adam**: 모멘텀 + 적응적 학습률
- **RMSprop**: 그래디언트 크기 정규화

---


## 7. 실전 팁

### 7.1 과적합 방지 (Overfitting Prevention)

#### 🛡️ 주요 기법들

1. **Dropout**
   - PyTorch: `self.dropout = nn.Dropout(0.5)`
   - TensorFlow: `layers.Dropout(0.5)`
2. **Batch Normalization**
   - PyTorch: `self.bn1 = nn.BatchNorm2d(32)`
   - TensorFlow: `layers.BatchNormalization()`
3. **Data Augmentation**
   - PyTorch:
     ```
     transforms.Compose([
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.RandomCrop(32, padding=4)
     ])
     ```
   - TensorFlow:
     ```
     tf.keras.preprocessing.image.ImageDataGenerator(
         rotation_range=20,
         width_shift_range=0.2,
         height_shift_range=0.2,
         horizontal_flip=True
     )
     ```

### 7.2 하이퍼파라미터 튜닝

| 파라미터         | 권장값         | 설명                        |
|------------------|---------------|-----------------------------|
| **Learning Rate**| 0.001 ~ 0.01  | 너무 크면 발산, 너무 작으면 학습 느림 |
| **Batch Size**   | 32 ~ 128      | GPU 메모리와 학습 안정성 고려 |
| **Filter Size**  | 3×3, 5×5      | 작은 필터 여러 개가 효과적   |
| **Dropout Rate** | 0.2 ~ 0.5     | 과적합 정도에 따라 조정      |

### 7.3 학습 최적화

1. **학습률 스케줄링**
   - PyTorch: `scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)`
   - TensorFlow:
     ```
     callbacks = [
         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
     ]
     ```
2. **Early Stopping**
   - TensorFlow:
     ```
     early_stopping = tf.keras.callbacks.EarlyStopping(
         monitor='val_loss', 
         patience=10, 
         restore_best_weights=True
     )
     ```
3. **모델 체크포인트**
   - PyTorch: `torch.save(model.state_dict(), 'best_model.pth')`
   - TensorFlow:
     ```
     checkpoint = tf.keras.callbacks.ModelCheckpoint(
         'best_model.h5', 
         save_best_only=True
     )
     ```

---

## 8. 실전 응용

### 8.1 이미지 분류 (Image Classification)

class CatDogClassifier(nn.Module):
def init(self):
super(CatDogClassifier, self).init()
self.features = nn.Sequential(
nn.Conv2d(3, 64, 3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
nn.Conv2d(64, 128, 3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
nn.Conv2d(128, 256, 3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
)
self.classifier = nn.Sequential(
nn.Linear(256 * 28 * 28, 512),
nn.ReLU(),
nn.Dropout(0.5),
nn.Linear(512, 2)
)



### 8.2 객체 탐지 (Object Detection)

주요 알고리즘:
- **YOLO (You Only Look Once)**
- **R-CNN 계열**
- **SSD (Single Shot Detector)**

### 8.3 이미지 분할 (Image Segmentation)

주요 알고리즘:
- **U-Net**
- **FCN (Fully Convolutional Network)**
- **DeepLab**

### 8.4 전이 학습 (Transfer Learning)

import torchvision.models as models

resnet = models.resnet50(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

for param in resnet.parameters():
param.requires_grad = False
for param in resnet.fc.parameters():
param.requires_grad = True


---

## 📝 마무리

CNN은 컴퓨터 비전 분야의 핵심 기술로, 이미지 처리에서 혁신적인 성과를 보여주고 있습니다. 이 가이드를 통해 CNN의 기본 원리부터 실전 응용까지 체계적으로 학습하실 수 있습니다.
