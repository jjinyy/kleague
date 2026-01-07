# K리그 최종 패스 좌표 예측 AI 모델

K리그 경기 내 주어진 플레이 시퀀스의 마지막 패스 도착 좌표(X,Y)를 예측하는 AI 모델입니다.

## 프로젝트 구조

```
.
├── config.py                 # 설정 파일
├── train.py                  # 모델 학습 코드
├── inference.py              # 추론 코드
├── requirements.txt          # 필요한 라이브러리 목록
├── README.md                 # 프로젝트 설명서
├── models/                   # 모델 정의
│   ├── __init__.py
│   └── model.py             # PassPredictor, TransformerPassPredictor
├── utils/                    # 유틸리티 함수
│   ├── __init__.py
│   ├── data_loader.py       # 데이터 로더 및 전처리
│   └── train_utils.py       # 학습 유틸리티 함수
└── output/                   # 출력 디렉토리 (자동 생성)
    ├── models/              # 학습된 모델 가중치
    ├── submissions/         # 제출 파일
    └── logs/                # 로그 파일
```

## 설치 방법

1. 필요한 라이브러리 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 모델 학습

```bash
python train.py
```

학습된 모델은 `output/models/best_model.pt`에 저장됩니다.

### 2. 추론 및 제출 파일 생성

```bash
python inference.py
```

제출 파일은 `output/submissions/submission.csv`에 생성됩니다.

## 모델 아키텍처

- **PassPredictor**: LSTM 기반 시퀀스 모델
  - 양방향 LSTM 레이어
  - Multi-head Attention 메커니즘
  - 완전 연결 레이어를 통한 최종 좌표 예측

- **TransformerPassPredictor**: Transformer 기반 모델 (대안)
  - Transformer Encoder 레이어
  - 위치 인코딩 포함

## 데이터 형식

### 입력 데이터
- 각 에피소드는 패스 시퀀스로 구성됨
- 각 패스는 다음 정보를 포함:
  - 시작/종료 좌표 (start_x, start_y, end_x, end_y)
  - 시간 정보 (time_seconds)
  - 액션 타입 및 결과 (type_name, result_name)
  - 선수 및 팀 정보

### 출력 데이터
- 각 에피소드의 마지막 패스 도착 좌표 (end_x, end_y)

## 평가 지표

- **유클리드 거리 (Euclidean Distance)**: 예측 좌표와 실제 좌표 간의 거리

## 주요 특징

1. **시퀀스 모델링**: 패스 시퀀스의 시간적 패턴 학습
2. **맥락 정보 활용**: 선수 배치, 시간, 액션 결과 등 다양한 피처 활용
3. **어텐션 메커니즘**: 중요한 패스에 더 많은 가중치 부여
4. **Early Stopping**: 과적합 방지
5. **Layer Normalization**: 학습 안정성 향상
6. **Gradient Clipping**: Gradient explosion 방지
7. **Huber Loss**: Outlier에 robust한 손실 함수
8. **가중치 초기화**: Xavier 초기화로 학습 안정성 향상

## 하이퍼파라미터 설정

`config.py`의 `ModelConfig` 클래스에서 다음 파라미터를 조정할 수 있습니다:

- `hidden_dim`: 은닉층 차원 (기본값: 256)
- `num_layers`: 레이어 수 (기본값: 3)
- `dropout`: 드롭아웃 비율 (기본값: 0.3)
- `batch_size`: 배치 크기 (기본값: 32)
- `learning_rate`: 학습률 (기본값: 0.001)
- `max_sequence_length`: 최대 시퀀스 길이 (기본값: 100)

## 주의사항

- 모든 코드는 UTF-8 인코딩으로 작성되었습니다.
- 데이터 경로는 상대 경로로 설정되어 있습니다.
- 학습과 추론 코드는 분리되어 있습니다.
- 모델 가중치 파일은 필수로 포함되어야 합니다.

## 라이선스

본 프로젝트는 데이콘 경진대회 참가를 위한 코드입니다.

