# -*- coding: utf-8 -*-
"""
추론 코드
"""
import torch
import pandas as pd
import numpy as np
import os
import sys

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, MODEL_DIR, SUBMISSION_DIR, DATA_DIR
from utils.data_loader import load_test_data, PassSequenceDataset
from models.model import PassPredictor, TransformerPassPredictor
from torch.utils.data import DataLoader


def load_model(model_path, config, device):
    """모델 로드"""
    model = PassPredictor(config).to(device)
    # 또는 Transformer 모델 사용 시:
    # model = TransformerPassPredictor(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"모델 로드 완료: {model_path}")
    return model


def predict(model, test_loader, device):
    """예측 수행"""
    predictions = []
    game_episodes = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            episodes = batch['game_episode']
            
            outputs = model(features, mask)
            
            # CPU로 이동 및 numpy 변환
            preds = outputs.cpu().numpy()
            
            predictions.extend(preds)
            game_episodes.extend(episodes)
    
    return np.array(predictions), game_episodes


def main():
    # 설정
    config = ModelConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    model_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 train.py를 실행하여 모델을 학습시켜주세요.")
        return
    
    model = load_model(model_path, config, device)
    
    # 테스트 데이터 로드
    test_df = load_test_data(config)
    
    if len(test_df) == 0:
        print("테스트 데이터가 없습니다.")
        return
    
    # 테스트 데이터셋 생성
    test_dataset = PassSequenceDataset(test_df, config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 예측 수행
    print("\n예측 시작...")
    predictions, game_episodes = predict(model, test_loader, device)
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'game_episode': game_episodes,
        'end_x': predictions[:, 0],
        'end_y': predictions[:, 1]
    })
    
    # 경기장 범위 내로 클리핑 (0-105, 0-68)
    submission_df['end_x'] = submission_df['end_x'].clip(0, 105)
    submission_df['end_y'] = submission_df['end_y'].clip(0, 68)
    
    # 샘플 제출 파일과 동일한 순서로 정렬
    sample_submission = pd.read_csv(
        os.path.join(DATA_DIR, "sample_submission.csv"),
        encoding='utf-8'
    )
    
    # game_episode 기준으로 정렬
    submission_df = submission_df.set_index('game_episode')
    submission_df = submission_df.reindex(sample_submission['game_episode'])
    submission_df = submission_df.reset_index()
    
    # 빈 값 처리 (경기장 중앙 좌표로 채움: 105/2=52.5, 68/2=34.0)
    # 또는 마지막 패스의 시작 좌표를 사용할 수도 있음
    empty_mask = submission_df['end_x'].isna() | submission_df['end_y'].isna()
    if empty_mask.sum() > 0:
        print(f"\n빈 값 {empty_mask.sum()}개를 기본값으로 채웁니다.")
        # 경기장 중앙 좌표 사용
        submission_df.loc[empty_mask, 'end_x'] = 52.5
        submission_df.loc[empty_mask, 'end_y'] = 34.0
    
    # 제출 파일 저장
    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n제출 파일 저장 완료: {output_path}")
    print(f"예측된 에피소드 수: {len(submission_df)}")
    print(f"\n예측 결과 샘플:")
    print(submission_df.head(10))


if __name__ == "__main__":
    main()

