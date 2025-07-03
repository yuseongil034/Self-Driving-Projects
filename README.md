# 📘 AI 학습 정리

## 1. About GitHub, Markdown, Colab
- [GitHub 사용법](github-guide.md)
- [Markdown 문법](markdown-guide.md)  
- [Colab 기초](colab-guide.md)

## GitHub 사용법

### ✅ GitHub 계정 만드는 순서 (2025년 기준)

1. **웹 브라우저 열기**
   크롬(Chrome), 엣지(Edge), 사파리(Safari) 중 편한 걸 사용하세요.

2. **GitHub 웹사이트 접속**
   주소창에 아래 주소를 입력하고 Enter 누르세요: https://github.com

3. **회원가입 시작하기**
   화면 오른쪽 위 또는 중간에 있는 Sign up 버튼 클릭

4. **이메일 주소 입력**
   평소 자주 사용하는 이메일을 입력

5. **비밀번호 만들기**
   영어 대문자, 소문자, 숫자, 특수문자를 섞어 안전하게!
   예시: Git1234!hub

6. **사용자 이름(Username) 정하기**
   나만의 고유한 이름을 지어요 (다른 사람이 쓰고 있으면 불가)
   - 예시: jetsunmom, sungsookjang66 등
   - 영어, 숫자, 하이픈(-) 가능 (띄어쓰기 ❌)

### ✅ Repository 만들기 순서

1. **GitHub에 로그인 후 New Repository 클릭**
2. ![new](https://github.com/user-attachments/assets/3481a680-f677-403b-be8c-1fe59d5aa7cb)

3. **Repository 이름 입력**
4. **Public/Private 선택**
5. **README.md 파일 생성 체크**
6. **Create repository 버튼 클릭**
   
![create_repository](https://github.com/user-attachments/assets/8c2eb16b-8dfc-465a-88cd-d35770d12df0)

## Markdown 문법

아래에 **설명과 마크다운 문법을 함께** 적어드릴게요.
복사해서 GitHub `README.md` 등에 붙여넣으면 바로 보실 수 있습니다.

````
# 마크다운(Markdown) 사용법

## 제목 만들기  
글의 제목이나 소제목을 만들 때 사용합니다.  
`#`을 많이 쓸수록 제목이 작아져요.

```markdown
# 큰 제목
## 중간 제목
### 작은 제목
````

---

## 굵은 글씨 / 기울임 / 취소선

강조하고 싶은 글자를 꾸미는 방법입니다.

```markdown
**굵은 글씨**   → 글자를 진하게  
*기울임*       → 글자가 기울어짐  
~~취소선~~     → 글자에 줄이 그어짐
```

---

## 목록 만들기

순서 없이 여러 항목을 나열할 때 사용합니다.

```markdown
- 첫 번째 항목
- 두 번째 항목
  - 안쪽 항목 (들여쓰기)
```

---

## 체크박스 목록

할 일 목록을 만들고 완료 여부를 표시할 수 있습니다.

```markdown
- [ ] 해야 할 일 1   → 아직 안 한 일
- [x] 끝낸 일        → 체크 표시된 일
```

---

## 링크 만들기

다른 사이트나 페이지로 이동하는 링크를 걸 수 있습니다.

```markdown
[보여질 글자](https://example.com)
```

---

## 이미지 넣기

이미지 주소(URL)를 사용해서 화면에 그림을 보여줄 수 있습니다.

```markdown
![그림 설명](https://이미지주소.com/image.png)
```

---

## 코드 표시하기

### 한 줄 코드

글 중간에 짧은 코드를 보여줄 때 사용합니다.

```markdown
`print("Hello")`
```

### 여러 줄 코드

여러 줄짜리 코드를 넣을 때 사용합니다.

````markdown
```python
print("Hello")
print("World")
````

````

※ 위 예시는 실제 쓸 때 백틱(```) 3개 써야 하고, 언어는 생략 가능

---

## 수평선 (줄 긋기)  
글 사이에 선을 그어서 구분할 수 있습니다.

```markdown
---
````

---

# 마크다운 문법 요약표

| 구분       | 문법 예시                       | 설명                        |
| -------- | --------------------------- | ------------------------- |
| 제목       | `# 제목1`<br>`## 제목2`         | `#` 개수로 제목 크기 조절 (최대 6단계) |
| 굵은 글씨    | `**굵게**`                    | 글자를 진하게                   |
| 기울임 글씨   | `*기울임*`                     | 글자를 기울임                   |
| 취소선      | `~~취소선~~`                   | 글자에 가로줄                   |
| 순서 없는 목록 | `- 항목`<br>`* 항목`            | 점(•) 목록 만들기               |
| 체크박스 목록  | `- [ ] 할 일`<br>`- [x] 끝낸 일` | 할 일 체크리스트                 |
| 링크       | `[텍스트](https://주소)`         | 클릭 가능한 링크 만들기             |
| 이미지      | `![설명](https://이미지주소)`      | 이미지를 삽입                   |
| 한 줄 코드   | `` `코드` ``                  | 글 중간에 코드처럼 보이게 표시         |
| 여러 줄 코드  | <pre>`<br>코드<br>`</pre>     | 여러 줄짜리 코드 블록 만들기          |
| 수평선      | `---`                       | 가로 줄로 구분선 넣기              |


## Colab 기초  

---

## 1. Colab이란?

- **무료**로 파이썬 코드를 작성하고 실행할 수 있는 웹 기반 도구
- **구글 계정**만 있으면 누구나 사용 가능
- **코드**와 **설명(마크다운)**을 섞어서 사용 가능

---

## 2. Colab 시작 방법

1. **구글에 "Colab" 검색** 또는 [https://colab.research.google.com/](https://colab.research.google.com/) 접속
2. **새 노트북 만들기** 클릭
3. **코드 셀** 또는 **텍스트 셀** 추가해서 사용

---

## 3. Colab의 셀(Cell) 종류

| 셀 종류   | 설명                        | 사용 방법 예시                    |
| --------- | --------------------------- | ---------------------------------- |
| 코드 셀   | 파이썬 코드 입력 및 실행    | `print("안녕하세요!")`             |
| 텍스트 셀 | 마크다운으로 설명 작성 가능 | **굵게**, *기울임* 등 마크다운 사용 |

---

## 4. 마크다운 문법 요약 (Colab에서 사용)

| 기능           | 마크다운 문법 예시                | 결과 예시                  |
| -------------- | -------------------------------- | -------------------------- |
| 제목           | `# 제목1``## 제목2`          | # 제목1## 제목2        |
| 굵게           | `**굵게**` 또는 `__굵게__`       | **굵게**                   |
| 기울임         | `*기울임*` 또는 `_기울임_`       | *기울임*                   |
| 순서 없는 목록 | `- 사과``- 바나나`           | - 사과- 바나나         |
| 순서 있는 목록 | `1. 첫 번째``2. 두 번째`     | 1. 첫 번째2. 두 번째   |
| 링크           | `[네이버](https://naver.com)`    | [네이버](https://naver.com) |
| 이미지         | `` |  |
| 구분선         | `---`                            | ---                        |
| 인용문         | `> 인용문 예시입니다.`           | > 인용문 예시입니다.       |

---

## 5. Colab에서 마크다운 사용 방법

1. **셀 추가** 버튼 클릭
2. **텍스트 셀** 선택
3. 위 표에 있는 마크다운 문법으로 글 작성
4. **실행(Shift+Enter)** 하면 예쁘게 보임

---

## 6. 파이썬 코드 실행 방법

1. **코드 셀** 선택
2. 파이썬 코드 입력 (예: `print("Hello Colab!")`)
3. **실행(Shift+Enter)** 하면 결과가 아래에 나옴




## 2. About Python3
- [Python basic](docs/python3.md)
- https://www.w3schools.com/python/default.asp
 기본 문법

- [for문 예제](0625_for&while.ipynb)
- [함수 정의 (define)](0625_python_define.ipynb)
- [for & while 심화](0625_python_for&while.ipynb)



 자료구조

- [리스트 기초](0626_python_list.ipynb)
- [딕셔너리 기초](0626_python_dictionary.ipynb)
- [딕셔너리 심화](0627_python_dictionary.ipynb)
- [튜플 자료형](0627_python_tuple.ipynb)



 라이브러리 학습 (NumPy, Matplotlib)

- [NumPy 기초](0703_python_Numpy.ipynb)
- [NumPy + 자율주행 융합](0703_python_numpy융합.ipynb)
- [Matplotlib 기초](0703_python_Matplotlib.ipynb)
- [Matplotlib 마크다운 정리](0703_python_Matplotlib_md.ipynb)
  

## 3.  data structure / data sciencs

- [데이터 구조 개요](data_structures.md)
- [Pandas](pandas.md)
- [Numpy](numpy.md)
- [Matplotlib](Matplotlib.md)

## 4. Machine Learning

- [Machine Learning Basic](ml_basic.md)
- [모델 훈련 및 평가](ml_test.md)

## 5. OpenCV

- [OpenCV Basic](OpenCV_basic.md)
- [이미지 처리](image_test.md)

  
## 6. CNN(Convolution Neural Network
- [CNN_Basic](CNN_basic.md)
- [CNN_자율주행 관련 코드](cnn_test.md)

## 7. Ultralytics
- [Ultralytics_Basic](Ultralytics_basic.md)
- [YOLOv8](YOLOv8_test.md)
- [YOLOv12](YOLOv12_test.md)
  
## 8. TensorRT vs PyTorch 
- [PyTorch_Basic](PyTorch_basic.md)
- [TensorRT](TensorRT_test.md)
- [YOLOv12](YOLOv12_test.md)

## 9. TAO Toolkit on RunPod
- [TAO_사용법](TAO_install.md)
- [TAO_Toolkit](TAO_Toolkit.md)

## 10. 칼만필터, CARLA, 경로 알고리즘
- [kalman](kalman.md)
- [CARLA_simulator](CARLA.md)

## 11. ADAS & (ADAS TensorRT vs PyTorch)
- [adas_basic](adas_basic.md)
- [TensorRT vs PyTorch 비교](vs.md)
