# -*- coding: utf-8 -*-
"""
설정 파일
"""
import os

# 데이터 경로
DATA_DIR = "./open_track1"  # 사용자가 open__track1에 데이터를 넣었다고 했지만 실제로는 open_track1
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
MATCH_INFO_PATH = os.path.join(DATA_DIR, "match_info.csv")

# 출력 경로
OUTPUT_DIR = "./output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 모델 하이퍼파라미터
class ModelConfig:
    # 데이터 경로
    DATA_DIR = DATA_DIR
    TRAIN_DATA_PATH = TRAIN_DATA_PATH
    TEST_DATA_PATH = TEST_DATA_PATH
    SAMPLE_SUBMISSION_PATH = SAMPLE_SUBMISSION_PATH
    MATCH_INFO_PATH = MATCH_INFO_PATH
    
    # 출력 경로
    OUTPUT_DIR = OUTPUT_DIR
    MODEL_DIR = MODEL_DIR
    SUBMISSION_DIR = SUBMISSION_DIR
    LOG_DIR = LOG_DIR
    
    # 시퀀스 모델 설정 (성능 향상을 위해 증가)
    hidden_dim = 512  # 256 -> 512로 증가
    num_layers = 4  # 3 -> 4로 증가
    dropout = 0.2  # 0.3 -> 0.2로 감소 (더 큰 모델에 맞춤)
    bidirectional = True
    
    # 입력 피처 설정
    feature_dim = 34  # 실제 피처 개수: 34개
    output_dim = 2  # end_x, end_y
    
    # 학습 설정 (성능 향상)
    batch_size = 64  # 32 -> 64로 증가 (더 안정적인 학습)
    learning_rate = 0.0005  # 0.001 -> 0.0005로 감소 (더 안정적인 학습)
    num_epochs = 100  # 50 -> 100으로 증가
    early_stopping_patience = 15  # 10 -> 15로 증가
    max_grad_norm = 1.0  # Gradient clipping
    warmup_epochs = 5  # Learning rate warmup
    
    # 시퀀스 길이
    max_sequence_length = 100  # 최대 시퀀스 길이
    
    # 기타
    device = "cuda"  # "cuda" or "cpu"
    seed = 42
    
    # 로깅
    verbose = True  # 상세 로그 출력 여부

