# -*- coding: utf-8 -*-
"""모델 상태 확인"""
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, MODEL_DIR
from models.model import PassPredictor

config = ModelConfig()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# 모델 로드
model_path = os.path.join(MODEL_DIR, "best_model.pt")
checkpoint = torch.load(model_path, map_location=device)
model = PassPredictor(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("모델 상태 확인:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Loss: {checkpoint['loss']:.4f}")

# 더미 입력으로 테스트
dummy_input = torch.randn(1, 100, 20).to(device)
dummy_mask = torch.ones(1, 100).to(device)

print(f"\n더미 입력 테스트:")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Input range: [{dummy_input.min().item():.2f}, {dummy_input.max().item():.2f}]")
print(f"  Input has NaN: {torch.isnan(dummy_input).any().item()}")
print(f"  Input has Inf: {torch.isinf(dummy_input).any().item()}")

with torch.no_grad():
    try:
        output = model(dummy_input, dummy_mask)
        print(f"\n출력:")
        print(f"  Output: {output}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output has NaN: {torch.isnan(output).any().item()}")
        print(f"  Output has Inf: {torch.isinf(output).any().item()}")
        
        if torch.isnan(output).any():
            print("\n[문제] 모델이 NaN을 출력합니다!")
            print("가능한 원인:")
            print("1. 모델 가중치에 NaN이 포함되어 있음")
            print("2. 모델 구조에 문제가 있음")
            
            # 가중치 확인
            has_nan = False
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"  - {name}: NaN 발견!")
                    has_nan = True
            if not has_nan:
                print("  - 모든 가중치가 정상입니다.")
        else:
            print("\n[정상] 모델이 정상적으로 작동합니다.")
    except Exception as e:
        print(f"\n[오류] 모델 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

