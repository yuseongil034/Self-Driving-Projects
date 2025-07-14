# 🧠 CNN (Convolutional Neural Network) 완벽 가이드

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Latest-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green.svg" alt="Level">
</div>

📝 **업데이트**: 2024년 최신 버전으로 지속적으로 업데이트 중입니다.  
🌟 도움이 되셨다면 Star를 눌러주세요!

---

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
- **파라미터 공유**: 같은 필터를 이미지 전체에 적용하여 효율성 증가  
- **평행 이동 불변성**: 객체 위치 변화에도 특징 감지 가능  
- **계층적 특징 추출**: 저수준 → 고수준 특징을 단계적으로 학습  

### 🆚 일반 신경망과 차이점

| 특성         | 일반 신경망         | CNN                          |
|--------------|---------------------|-------------------------------|
| 입력 형태     | 1D 벡터             | 2D/3D 텐서                    |
| 연결 방식     | 완전 연결            | 지역적 연결                   |
| 파라미터 수   | 많음                | 적음 (공유)                   |
| 공간 정보     | 손실                | 보존                          |

---

## 2. CNN의 기본 구성요소

입력 이미지 → **합성곱층** → **활성화 함수** → **풀링층** → ... → **완전연결층** → 출력

| 구성 요소         | 설명                        |
|------------------|-----------------------------|
| 합성곱층         | 특징 추출                   |
| 활성화 함수       | 비선형성 추가               |
| 풀링층           | 차원 축소, 위치 불변성      |
| 완전연결층       | 최종 출력 계산               |

---

## 3. 각 구성요소의 작동 원리

### 3.1 합성곱층

- 필터(커널)를 이미지에 슬라이딩하며 특징맵 생성  
- 수식:  
  ```math
  (f * g)(x, y) = \sum_i \sum_j f(i,j) \cdot g(x - i, y - j)
  ```
- 출력 크기 계산:  
  ```
  출력 = (입력 - 필터 + 2×패딩) / 스트라이드 + 1
  ```

### 3.2 활성화 함수

- ReLU: `f(x) = max(0, x)`  
- Leaky ReLU: `f(x) = max(0.01x, x)`

### 3.3 풀링층

- **Max Pooling**: 윈도우 내 최대값 추출  
- **Average Pooling**: 평균값 추출

### 3.4 완전연결층

- 수식: `y = W·x + b`

---

## 4. 주요 용어 정리

| 용어         | 설명                             |
|--------------|----------------------------------|
| Filter        | 특징을 추출하는 작은 행렬 (3×3 등) |
| Kernel        | 필터와 동일 의미                  |
| Stride        | 필터 이동 간격                    |
| Padding       | 입력 주변에 0 추가                 |
| Feature Map   | 합성곱 연산 결과                  |
| Receptive Field | 뉴런이 참조하는 입력 범위         |
| Depth         | 채널 수 (예: RGB = 3)            |

**패딩 종류**

- Valid: 패딩 없음
- Same: 출력 크기 = 입력 크기
- Full: 필터 크기 - 1만큼 패딩

---

## 5. CNN의 학습 과정

### 5.1 순전파

1. 입력 → 합성곱 → 활성화 → 풀링 반복
2. 완전연결층 → 예측값 도출

### 5.2 역전파

```math
1. L = CrossEntropy(y_pred, y_true)  
2. \frac{∂L}{∂W}, \frac{∂L}{∂b} 계산  
3. W ← W - α·∇W
```

### 5.3 최적화 알고리즘

- **SGD**
- **Adam**
- **RMSprop**

---

## 6. 코드 예제

### 6.1 PyTorch (MNIST)

<details>
<summary>👨‍💻 코드 보기</summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

</details>

---

### 6.2 TensorFlow (CIFAR-10)

<details>
<summary>👨‍💻 코드 보기</summary>

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

</details>

---

## 7. 실전 팁

### 7.1 과적합 방지

- Dropout
- Batch Normalization
- Data Augmentation

### 7.2 하이퍼파라미터

| 파라미터       | 권장값            | 설명                     |
|----------------|------------------|--------------------------|
| Learning Rate  | 0.001 ~ 0.01     | 작으면 느림, 크면 발산 위험 |
| Batch Size     | 32 ~ 128         | GPU 메모리 고려           |
| Filter Size    | 3×3, 5×5         | 작은 필터 반복 추천       |
| Dropout Rate   | 0.2 ~ 0.5        | 과적합 방지              |

### 7.3 학습 최적화

- ReduceLROnPlateau
- EarlyStopping
- ModelCheckpoint

---

## 8. 실전 응용

- 이미지 분류 (Image Classification)  
- 객체 탐지 (YOLO, SSD, Faster R-CNN)  
- 이미지 분할 (U-Net, FCN)  
- 전이 학습 (Transfer Learning with ResNet 등)

---

## 9. 참고자료

### 📚 추천 도서

- "Deep Learning" - Ian Goodfellow  
- "Hands-On Machine Learning" - Aurélien Géron

### 🌐 온라인 강의

- Stanford CS231n  
- Fast.ai Practical Deep Learning

### 🔗 유용한 링크

- [Papers with Code](https://paperswithcode.com)  
- [Distill.pub](https://distill.pub)  
- [Towards Data Science](https://towardsdatascience.com)
