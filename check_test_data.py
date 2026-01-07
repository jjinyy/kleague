# -*- coding: utf-8 -*-
"""테스트 데이터 확인"""
import torch
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from utils.data_loader import load_test_data, PassSequenceDataset

config = ModelConfig()

# 테스트 데이터 로드
test_df = load_test_data(config)
print(f"테스트 데이터 크기: {len(test_df)}")

# 테스트 데이터셋 생성
test_dataset = PassSequenceDataset(test_df, config, is_train=False)
print(f"테스트 데이터셋 크기: {len(test_dataset)}")

if len(test_dataset) > 0:
    # 여러 샘플 확인
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        features = sample['features']
        mask = sample['mask']
        
        print(f"\n샘플 {i+1} (game_episode: {sample['game_episode']}):")
        print(f"  Features shape: {features.shape}")
        print(f"  Features range: [{features.min().item():.2f}, {features.max().item():.2f}]")
        print(f"  Features has NaN: {torch.isnan(features).any().item()}")
        print(f"  Features has Inf: {torch.isinf(features).any().item()}")
        print(f"  Mask sum: {mask.sum().item()}")
        print(f"  Sequence length: {sample['sequence_length']}")
        
        if torch.isnan(features).any():
            print(f"  [경고] NaN 발견!")
            nan_indices = torch.isnan(features).any(dim=1)
            print(f"  NaN이 있는 타임스텝: {nan_indices.sum().item()}개")
        
        if torch.isinf(features).any():
            print(f"  [경고] Inf 발견!")
            inf_indices = torch.isinf(features).any(dim=1)
            print(f"  Inf가 있는 타임스텝: {inf_indices.sum().item()}개")

