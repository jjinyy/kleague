# -*- coding: utf-8 -*-
"""
모델 학습 코드
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import random
import os
import sys

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, MODEL_DIR, LOG_DIR
from utils.data_loader import load_train_data, create_data_loaders
from utils.train_utils import train_epoch, validate, save_checkpoint
from models.model import PassPredictor, TransformerPassPredictor


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 설정
    config = ModelConfig()
    set_seed(config.seed)
    
    # 디바이스 설정
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 데이터 로드
    train_df = load_train_data(config)
    
    # 데이터 로더 생성
    train_loader, val_loader = create_data_loaders(train_df, config, val_split=0.2)
    
    # 모델 생성
    model = PassPredictor(config).to(device)
    # 또는 Transformer 모델 사용 시:
    # model = TransformerPassPredictor(config).to(device)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss 함수 및 Optimizer
    # 유클리드 거리와 직접적인 손실 함수 조합
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.huber = nn.HuberLoss(delta=1.0)
            self.mse = nn.MSELoss()
        
        def forward(self, pred, target):
            # Huber Loss와 MSE의 조합 (가중 평균)
            huber_loss = self.huber(pred, target)
            mse_loss = self.mse(pred, target)
            # 유클리드 거리도 포함
            euclidean_dist = torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()
            # 조합된 손실 (Huber 70%, MSE 20%, Euclidean 10%)
            return 0.7 * huber_loss + 0.2 * mse_loss + 0.1 * euclidean_dist
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler (Cosine Annealing + ReduceLROnPlateau 조합)
    scheduler_plateau = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01
    )
    
    # 학습 루프
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("\n학습 시작...")
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 50)
        
        # 학습
        train_loss, train_dist = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        
        # 검증
        val_loss, val_dist = validate(
            model, val_loader, criterion, device, config
        )
        
        # Learning rate warmup
        if epoch <= config.warmup_epochs:
            warmup_factor = epoch / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * warmup_factor
        
        # 학습률 스케줄러 업데이트
        scheduler_plateau.step(val_loss)
        if epoch > config.warmup_epochs:
            scheduler_cosine.step()
        
        # 학습 히스토리 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch} 결과:")
        print(f"  Train Loss: {train_loss:.4f}, Train Distance: {train_dist:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Distance: {val_dist:.4f}")
        
        # 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"  [최고 모델 저장] (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping patience: {patience_counter}/{config.early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        # 주기적 체크포인트 저장
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    print("\n학습 완료!")
    print(f"최고 검증 Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

