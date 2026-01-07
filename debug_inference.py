# -*- coding: utf-8 -*-
"""추론 디버깅"""
import torch
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, MODEL_DIR, DATA_DIR
from utils.data_loader import load_test_data, PassSequenceDataset
from models.model import PassPredictor
from torch.utils.data import DataLoader

config = ModelConfig()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# 모델 로드
model_path = os.path.join(MODEL_DIR, "best_model.pt")
checkpoint = torch.load(model_path, map_location=device)
model = PassPredictor(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 테스트 데이터 로드
test_df = load_test_data(config)
print(f"\n테스트 데이터 크기: {len(test_df)}")

# 테스트 데이터셋 생성
test_dataset = PassSequenceDataset(test_df, config, is_train=False)
print(f"테스트 데이터셋 크기: {len(test_dataset)}")

if len(test_dataset) > 0:
    # 첫 번째 샘플 확인
    sample = test_dataset[0]
    print(f"\n첫 번째 샘플:")
    print(f"  game_episode: {sample['game_episode']}")
    print(f"  features shape: {sample['features'].shape}")
    print(f"  mask shape: {sample['mask'].shape}")
    print(f"  sequence_length: {sample['sequence_length']}")
    
    # 예측 테스트
    features = sample['features'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(features, mask)
        print(f"\n예측 결과:")
        print(f"  output: {output}")
        print(f"  output shape: {output.shape}")
        print(f"  end_x: {output[0, 0].item():.2f}")
        print(f"  end_y: {output[0, 1].item():.2f}")
        
        # NaN 체크
        if torch.isnan(output).any():
            print("  [경고] 예측값에 NaN이 포함되어 있습니다!")
        else:
            print("  [정상] 예측값이 정상입니다.")

# 샘플 제출 파일 확인
sample_submission = pd.read_csv(
    os.path.join(DATA_DIR, "sample_submission.csv"),
    encoding='utf-8'
)
print(f"\n샘플 제출 파일 에피소드 수: {len(sample_submission)}")
print(f"테스트 데이터셋 에피소드 수: {len(test_dataset)}")

if len(test_dataset) != len(sample_submission):
    print(f"\n[경고] 에피소드 수가 일치하지 않습니다!")
    print(f"  차이: {len(sample_submission) - len(test_dataset)}개")

