# 자율주행 초보자를 위한 센서와 객체 검출 완벽 가이드 🚗

## 📌 들어가며

자율주행 자동차가 "보고, 판단하고, 움직이는" 과정을 이해하는 첫걸음! 이 가이드는 센서의 원리부터 시뮬레이션까지 전체 개발 과정을 쉽게 설명합니다.

---

## 🎯 1. 라이다(LiDAR)와 레이더(Radar)의 이해

### 1.1 두 센서의 기본 원리

#### 🔦 라이다 (LiDAR: Light Detection and Ranging)
**"레이저로 3D 지도를 그리는 센서"**

```
작동 원리:
1. 레이저 빔 발사 (초당 수십만 번)
2. 물체에 반사되어 돌아옴
3. 시간 측정 → 거리 계산
4. 360도 회전하며 주변 스캔
```

**일상생활 비유:**
- 어두운 방에서 손전등을 빠르게 돌리며 주변을 파악하는 것과 유사
- 단, 라이다는 초당 수십만 번 빛을 쏘고 정확한 거리까지 측정!

#### 📡 레이더 (Radar: Radio Detection and Ranging)
**"전파로 속도와 거리를 측정하는 센서"**

```
작동 원리:
1. 전파(라디오파) 발사
2. 물체에 반사되어 돌아옴
3. 도플러 효과로 속도 측정
4. 시간차로 거리 계산
```

**일상생활 비유:**
- 박쥐가 초음파로 주변을 인식하는 것과 유사
- 구급차 사이렌 소리가 다가올 때와 멀어질 때 달라지는 원리(도플러 효과) 활용

### 1.2 수집 데이터의 차이

#### 라이다가 생성하는 데이터: Point Cloud (점군)
```python
# 라이다 데이터 예시 (x, y, z, intensity)
point_cloud = [
    [10.5, 2.3, 0.8, 120],  # x=10.5m, y=2.3m, z=0.8m, 반사강도=120
    [10.6, 2.3, 0.8, 115],
    [10.7, 2.4, 0.8, 118],
    # ... 수십만 개의 점들
]
```

**시각화하면:**
```
     ·····  ← 차량
   ·········
  ···········  ← 보행자
 ·············
```

#### 레이더가 생성하는 데이터
```python
# 레이더 데이터 예시 (거리, 속도, 각도, 신호강도)
radar_data = [
    {"range": 50.0, "velocity": -15.0, "angle": 30, "rcs": 10.5},
    {"range": 25.0, "velocity": 0.0, "angle": -45, "rcs": 5.2},
    # ... 수십~수백 개의 탐지점
]
```

### 1.3 센서 특성 비교

| 특성 | 라이다 | 레이더 |
|------|--------|--------|
| **측정 원리** | 레이저 (빛) | 전파 |
| **거리 정확도** | ±2cm | ±1m |
| **최대 거리** | ~200m | ~250m |
| **해상도** | 매우 높음 (수십만 점) | 낮음 (수십~수백 점) |
| **속도 측정** | 불가능 | 가능 (도플러) |
| **날씨 영향** | 큼 (비/안개/눈) | 적음 |
| **가격** | 비쌈 ($5,000~) | 저렴 ($100~) |
| **크기** | 크고 무거움 | 작고 가벼움 |

### 1.4 날씨별 성능 변화

```
맑은 날 (100% 성능 기준):
라이다: ████████████ 100%
레이더: ████████████ 100%

안개:
라이다: ████░░░░░░░░ 30%
레이더: ██████████░░ 85%

폭우:
라이다: ██████░░░░░░ 50%
레이더: █████████░░░ 75%

눈보라:
라이다: ████░░░░░░░░ 30%
레이더: ████████░░░░ 70%
```

### 1.5 자율주행에서의 활용 포지션

#### 🎯 라이다의 역할
- **정밀 3D 매핑**: 주변 환경의 정확한 형상 파악
- **차선 인식**: 도로 표면의 미세한 변화 감지
- **정적 장애물 탐지**: 주차된 차, 가로수, 건물 등

#### 🎯 레이더의 역할
- **이동 물체 추적**: 다른 차량의 속도/방향 파악
- **긴급 제동 시스템**: 전방 충돌 위험 감지
- **악천후 보조**: 라이다가 약할 때 백업

#### 🤝 센서 융합의 중요성
```
최적의 자율주행 = 라이다 + 레이더 + 카메라
                    ↓        ↓         ↓
                 정밀도    속도    색상/표지판
```

---

## 🔍 2. 객체 검출(Object Detection)의 기본 원리

### 2.1 객체 검출이란?

**"센서 데이터에서 '무엇이 어디에 있는지' 찾아내는 기술"**

```
입력: 센서 데이터 (점군, 이미지 등)
  ↓
처리: AI 알고리즘
  ↓
출력: 물체 종류 + 위치 (Bounding Box)
```

### 2.2 3D Point Cloud 기반 검출 원리

#### Step 1: 전처리 (Preprocessing)
```python
# 원시 점군 데이터
raw_points = load_point_cloud()  # 수십만 개의 점

# 1. 범위 제한 (관심 영역만)
roi_points = filter_by_range(raw_points, x=(-50, 50), y=(-50, 50))

# 2. 지면 제거
ground_removed = remove_ground_plane(roi_points)

# 3. 복셀화 (Voxelization) - 3D 격자로 나누기
voxels = voxelize(ground_removed, voxel_size=0.1)  # 10cm 크기 복셀
```

**복셀화 시각화:**
```
원본 점군:           복셀화 후:
·····················  ■■■■■■■■
·····················  ■■■■■■■■
·····················  ■■■■■■■■
(연속적인 점들)      (격자 단위로 정리)
```

#### Step 2: 특징 추출 (Feature Extraction)
```python
# 각 복셀의 특징 계산
features = []
for voxel in voxels:
    feature = {
        'num_points': len(voxel.points),
        'center': voxel.center,
        'density': voxel.density,
        'height': voxel.max_z - voxel.min_z
    }
    features.append(feature)
```

#### Step 3: 객체 검출 네트워크
```
복셀 특징 → 3D CNN → 후보 영역 → NMS → 최종 박스
           ↓         ↓          ↓
      특징 학습   중복 제거   신뢰도 필터링
```

### 2.3 이미지 기반(CNN) vs Point Cloud 기반 비교

| 구분 | 이미지 기반 (2D) | Point Cloud 기반 (3D) |
|------|------------------|---------------------|
| **입력 데이터** | RGB 이미지 | 3D 점군 |
| **차원** | 2D (가로×세로) | 3D (X×Y×Z) |
| **정보량** | 색상, 텍스처 | 정확한 거리, 크기 |
| **처리 속도** | 빠름 | 상대적으로 느림 |
| **거리 정확도** | 추정만 가능 | 정확한 측정 |
| **알고리즘** | YOLO, R-CNN | PointPillars, VoxelNet |

### 2.4 Bounding Box 생성 과정

#### 📦 Bounding Box란?
"검출된 물체를 감싸는 3D 상자"

```
구성 요소:
- 중심점 (x, y, z)
- 크기 (길이, 너비, 높이)
- 회전각 (yaw)
- 클래스 (차량, 보행자 등)
- 신뢰도 (0~1)
```

#### 생성 과정
```python
# 1. 초기 예측 (네트워크 출력)
raw_predictions = model(point_cloud)
# 출력: 수천 개의 후보 박스

# 2. 신뢰도 필터링
confident_boxes = [box for box in raw_predictions if box.score > 0.5]

# 3. NMS (Non-Maximum Suppression) - 중복 제거
final_boxes = []
for class_name in ['car', 'pedestrian', 'cyclist']:
    class_boxes = filter_by_class(confident_boxes, class_name)
    nms_boxes = apply_nms(class_boxes, iou_threshold=0.5)
    final_boxes.extend(nms_boxes)

# 4. 후처리
for box in final_boxes:
    # 크기 조정 (비현실적인 크기 제거)
    if not is_valid_size(box, class_name):
        continue
    
    # 지면 정렬
    box.z = align_to_ground(box)
    
    # 최종 결과에 추가
    detections.append(box)
```

### 2.5 예측-후처리 과정 시각화

```
1. 원시 예측 (중복 많음)
   ┌──┐┌──┐
   │차││차│  ← 같은 차를 여러 번 검출
   └──┘└──┘

2. NMS 적용 후
   ┌────┐
   │ 차 │   ← 하나로 통합
   └────┘

3. 최종 결과
   ┌────┐
   │차량│ 신뢰도: 0.92
   └────┘ 클래스: Car
          거리: 15.3m
```

---

## 🛠️ 3. 오프라인 개발 워크플로우

### 3.1 오프라인 vs 온라인 처리

#### 🔄 온라인 (실시간)
```
센서 → 처리 → 판단 → 제어
 ↑                      ↓
 └──────── 실시간 ──────┘
        (밀리초 단위)
```

#### 📁 오프라인 (사후 분석)
```
1. 데이터 수집 (주행 중 녹화)
2. 사무실에서 분석/학습
3. 개선된 모델 개발
4. 시뮬레이션 검증
5. 실차 적용
```

### 3.2 전체 개발 프로세스

```mermaid
데이터 수집 → 라벨링 → 학습 → 검증 → 시뮬레이션 → 배포
    ↓           ↓        ↓       ↓         ↓          ↓
 ROSbag     LabelCloud  PyTorch  mAP    CARLA    실차 테스트
```

### 3.3 단계별 상세 설명

#### 📹 Step 1: 데이터 수집
```bash
# ROS를 이용한 센서 데이터 녹화
rosbag record -a -o driving_data.bag

# 녹화되는 데이터:
# - /velodyne_points (라이다)
# - /radar/tracks (레이더)
# - /camera/image_raw (카메라)
# - /gps/fix (GPS)
# - /imu/data (IMU)
```

**수집 시 체크리스트:**
- ✅ 다양한 날씨 조건
- ✅ 다양한 시간대 (낮/밤)
- ✅ 다양한 도로 환경
- ✅ 다양한 교통 상황

#### 🏷️ Step 2: 라벨링
```python
# 라벨링 도구 예시
# 1. 3D 점군 라벨링
labelCloud --input point_cloud.pcd

# 2. 라벨 형식 (KITTI format)
# Class x y z l w h rotation
Car 15.3 2.1 0.8 4.5 1.8 1.5 1.57
Pedestrian 8.2 -1.5 0.9 0.6 0.6 1.7 0.0
```

**라벨링 팁:**
- 일관된 기준 유지
- 가려진 물체도 표시
- 불확실한 경우 팀 논의

#### 🧠 Step 3: 모델 학습
```python
# OpenPCDet을 이용한 학습
import OpenPCDet

# 설정
config = {
    'model': 'PointPillars',
    'dataset': 'custom_dataset',
    'batch_size': 4,
    'epochs': 80,
    'learning_rate': 0.001
}

# 학습 실행
model = OpenPCDet.build_model(config)
model.train()

# 학습 모니터링
# - Loss 감소 확인
# - Validation 성능 체크
# - Overfitting 방지
```

#### ✅ Step 4: 검증
```python
# 평가 지표 계산
results = model.evaluate(test_dataset)

print(f"mAP: {results['mAP']:.3f}")
print(f"Car AP: {results['Car']:.3f}")
print(f"Pedestrian AP: {results['Pedestrian']:.3f}")

# 시각화
visualize_predictions(
    point_cloud=test_data,
    predictions=model.predict(test_data),
    ground_truth=test_labels
)
```

**주요 평가 지표:**
- **mAP** (mean Average Precision): 전체 정확도
- **AP** (Average Precision): 클래스별 정확도
- **IoU** (Intersection over Union): 박스 겹침 정도

#### 🎮 Step 5: 시뮬레이션 적용
```python
# CARLA 시뮬레이터에서 테스트
import carla

# 모델 로드
detection_model = load_trained_model('model.pth')

# 시뮬레이션 실행
while True:
    # 센서 데이터 수집
    lidar_data = lidar_sensor.get_data()
    
    # 객체 검출
    detections = detection_model(lidar_data)
    
    # 결과 시각화
    render_bounding_boxes(detections)
    
    # 성능 측정
    fps = measure_fps()
    latency = measure_latency()
```

### 3.4 대표 도구들

#### 🔧 필수 도구 모음
| 도구 | 용도 | 특징 |
|------|------|------|
| **ROSbag** | 데이터 녹화/재생 | 모든 센서 동기화 녹화 |
| **LabelCloud** | 3D 라벨링 | 직관적인 UI |
| **OpenPCDet** | 3D 검출 학습 | 다양한 모델 지원 |
| **CARLA** | 시뮬레이션 | 현실적인 환경 |
| **Autoware** | 통합 플랫폼 | 전체 스택 제공 |

---

## 🎮 4. CARLA 시뮬레이터 데이터 활용

### 4.1 CARLA란?

**"현실적인 자율주행 시뮬레이터"**

```
특징:
- 오픈소스 (무료!)
- 물리 엔진 탑재
- 다양한 센서 시뮬레이션
- 날씨/시간 조절 가능
- Python API 제공
```

### 4.2 CARLA의 자율주행 학습 활용

#### 🎯 주요 용도
1. **안전한 실험 환경**: 실제 도로 위험 없이 테스트
2. **무한 데이터 생성**: 원하는 시나리오 반복 생성
3. **극한 상황 테스트**: 사고 상황, 악천후 등
4. **알고리즘 검증**: 개발한 모델 성능 확인

### 4.3 센서 설정

#### 📸 기본 센서 구성
```python
import carla

# CARLA 서버 연결
client = carla.Client('localhost', 2000)
world = client.get_world()

# 차량 생성
vehicle_bp = world.get_blueprint_library().filter('model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 센서 붙이기
sensor_transforms = {
    'lidar': carla.Transform(carla.Location(x=0, z=2.5)),
    'camera': carla.Transform(carla.Location(x=1.5, z=1.5)),
    'radar': carla.Transform(carla.Location(x=2.0, z=1.0))
}
```

#### 🔧 라이다 설정
```python
# 라이다 생성
lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', '64')  # 64선 라이다
lidar_bp.set_attribute('points_per_second', '1000000')  # 초당 100만 포인트
lidar_bp.set_attribute('rotation_frequency', '10')  # 10Hz
lidar_bp.set_attribute('range', '100')  # 100m 범위

lidar = world.spawn_actor(lidar_bp, sensor_transforms['lidar'], 
                         attach_to=vehicle)
```

#### 📷 카메라 설정
```python
# RGB 카메라
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1920')
camera_bp.set_attribute('image_size_y', '1080')
camera_bp.set_attribute('fov', '90')  # 시야각 90도

camera = world.spawn_actor(camera_bp, sensor_transforms['camera'], 
                          attach_to=vehicle)
```

### 4.4 데이터 수집 방식

#### 📹 녹화 모드
```python
# 전체 시뮬레이션 녹화
client.start_recorder("my_recording.log", True)

# 자동 주행 시작
vehicle.set_autopilot(True)

# 10분간 녹화
time.sleep(600)

# 녹화 중지
client.stop_recorder()

# 재생
client.replay_file("my_recording.log", start_time=0, duration=0, 
                   camera_follow_id=vehicle.id)
```

#### 💾 센서 데이터 저장
```python
import numpy as np
import open3d as o3d

# 라이다 데이터 저장 콜백
def save_lidar_data(data):
    # numpy 배열로 변환
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    
    # PCD 파일로 저장
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(f"lidar_{data.frame}.pcd", pcd)
    
    # NPY 파일로도 저장 (빠른 로딩용)
    np.save(f"lidar_{data.frame}.npy", points)

# 콜백 등록
lidar.listen(save_lidar_data)
```

### 4.5 학습용 데이터셋 구축 예시

#### 📁 프로젝트 구조
```
carla_dataset/
├── sequences/
│   ├── 0000/
│   │   ├── lidar/
│   │   │   ├── 000000.pcd
│   │   │   ├── 000001.pcd
│   │   │   └── ...
│   │   ├── camera/
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   ├── labels/
│   │   │   ├── 000000.txt
│   │   │   └── ...
│   │   └── calib.txt
│   └── ...
└── ImageSets/
    ├── train.txt
    └── val.txt
```

#### 🏷️ 자동 라벨링
```python
# CARLA는 Ground Truth 제공!
def generate_labels(world, vehicle, lidar_location):
    labels = []
    
    # 모든 차량 검색
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == vehicle.id:
            continue
            
        # 상대 위치 계산
        rel_pos = actor.get_location() - lidar_location
        
        # 바운딩 박스 정보
        bbox = actor.bounding_box
        
        label = {
            'type': 'Car',
            'location': [rel_pos.x, rel_pos.y, rel_pos.z],
            'dimensions': [bbox.extent.x*2, bbox.extent.y*2, bbox.extent.z*2],
            'rotation': actor.get_transform().rotation.yaw
        }
        labels.append(label)
    
    return labels
```

#### 🔄 완전한 데이터 수집 파이프라인
```python
def collect_training_data(num_frames=10000):
    frame_count = 0
    
    while frame_count < num_frames:
        # 랜덤 날씨 설정
        weather = random.choice([
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.HardRainNoon
        ])
        world.set_weather(weather)
        
        # 랜덤 교통 상황 생성
        spawn_vehicles(num_vehicles=random.randint(20, 50))
        spawn_pedestrians(num_pedestrians=random.randint(10, 30))
        
        # 데이터 수집
        world.tick()  # 시뮬레이션 한 스텝
        
        # 센서 데이터 저장
        save_all_sensors(frame_count)
        
        # 라벨 생성 및 저장
        labels = generate_labels(world, vehicle, lidar.get_location())
        save_labels(labels, frame_count)
        
        frame_count += 1
        
        # 진행 상황 출력
        if frame_count % 100 == 0:
            print(f"Collected {frame_count}/{num_frames} frames")
```

### 4.6 실습 예제: 간단한 객체 검출

```python
# 전체 실습 코드
import carla
import numpy as np
import cv2

class SimpleDetector:
    def __init__(self):
        # CARLA 연결
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.setup_vehicle_and_sensors()
        
    def setup_vehicle_and_sensors(self):
        # 차량 생성
        bp = self.world.get_blueprint_library().filter('model3')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        
        # 라이다 부착
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_transform = carla.Transform(carla.Location(z=2.5))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, 
                                           attach_to=self.vehicle)
        
    def detect_objects(self, point_cloud):
        # 간단한 클러스터링 기반 검출
        clusters = self.cluster_points(point_cloud)
        
        bboxes = []
        for cluster in clusters:
            if len(cluster) > 100:  # 최소 100개 점
                bbox = self.fit_bounding_box(cluster)
                bboxes.append(bbox)
                
        return bboxes
    
    def run(self):
        # 자동 운전 시작
        self.vehicle.set_autopilot(True)
        
        # 라이다 콜백 설정
        self.lidar.listen(lambda data: self.process_lidar(data))
        
        # 실행
        try:
            while True:
                self.world.tick()
        except KeyboardInterrupt:
            self.cleanup()
            
    def cleanup(self):
        self.vehicle.destroy()
        self.lidar.destroy()

# 실행
if __name__ == "__main__":
    detector = SimpleDetector()
    detector.run()
```

---

## 🎓 핵심 정리 및 다음 단계

### ✅ 오늘 배운 내용 정리

1. **센서의 이해**
   - 라이다: 정밀한 3D 지도, 날씨에 약함
   - 레이더: 속도 측정 가능, 날씨에 강함

2. **객체 검출 프로세스**
   - Point Cloud → 전처리 → 특징 추출 → 검출 → 후처리

3. **개발 워크플로우**
   - 데이터 수집 → 라벨링 → 학습 → 검증 → 시뮬레이션

4. **CARLA 활용**
   - 안전한 가상 환경에서 무한 데이터 생성
   - Ground Truth 자동 제공

### 🚀 다음 학습 추천

1. **초급**: CARLA 튜토리얼 따라하기
2. **중급**: OpenPCDet으로 실제 모델 학습
3. **고급**: 센서 융합 알고리즘 구현

### 📚 추가 학습 자료

- [CARLA 공식 문서](https://carla.readthedocs.io/)
- [OpenPCDet GitHub](https://github.com/open-mmlab/OpenPCDet)
- [Awesome Autonomous Driving](https://github.com/autonomousdrivingkr/awesome-autonomous-driving)

---

*자율주행의 세계에 오신 것을 환영합니다! 🚗*

*"천 리 길도 한 걸음부터" - 오늘 배운 내용이 여러분의 첫 걸음이 되기를 바랍니다!*

---

## 💡 실전 팁 모음

### 🔧 개발 시 주의사항

1. **데이터 품질이 모델 성능의 90%**
   - 다양한 시나리오 포함
   - 정확한 라벨링 필수
   - 불균형 데이터 주의

2. **실시간 처리를 위한 최적화**
   - 모델 경량화 (Pruning, Quantization)
   - 효율적인 전처리 파이프라인
   - GPU 메모리 관리

3. **안전을 최우선으로**
   - Fail-safe 메커니즘
   - 센서 이중화
   - 보수적인 임계값 설정

### 🎯 프로젝트 아이디어

#### 입문 프로젝트
1. **주차장 빈자리 찾기**
   - CARLA에서 주차장 시뮬레이션
   - 라이다로 빈 공간 검출
   - 난이도: ⭐⭐

2. **보행자 카운팅**
   - 횡단보도 시뮬레이션
   - 통행량 측정
   - 난이도: ⭐⭐⭐

#### 중급 프로젝트
1. **차선 변경 보조 시스템**
   - 주변 차량 추적
   - 안전 거리 계산
   - 난이도: ⭐⭐⭐⭐

2. **교차로 행동 예측**
   - 다중 객체 추적
   - 궤적 예측
   - 난이도: ⭐⭐⭐⭐⭐

### 📊 성능 벤치마크 참고

```
일반적인 3D 객체 검출 성능 (KITTI 데이터셋 기준):

모델명          | mAP  | FPS | GPU메모리
----------------|------|-----|----------
PointPillars    | 82.5 | 62  | 4GB
SECOND          | 83.9 | 20  | 6GB
PV-RCNN         | 90.2 | 5   | 12GB
CenterPoint     | 85.2 | 30  | 8GB

* FPS는 NVIDIA RTX 3090 기준
```

---

## 🤝 커뮤니티와 함께 성장하기

### 오픈소스 기여
- 데이터셋 공유
- 모델 개선 PR
- 문서 번역

### 질문하기 좋은 곳
- Stack Overflow (태그: autonomous-driving)
- ROS Discourse
- CARLA Forum

### 대회 참여
- Waymo Open Dataset Challenge
- nuScenes Detection Challenge
- Argoverse Competition

---

*자율주행 개발자로의 여정을 응원합니다! 🎉*

*궁금한 점이 있다면 언제든 GitHub Issue로 문의해주세요.*
