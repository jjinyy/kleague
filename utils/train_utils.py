# -*- coding: utf-8 -*-
"""
학습 유틸리티 함수
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os


def euclidean_distance(pred, target):
    """유클리드 거리 계산 (평가 지표)"""
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1))


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """한 에폭 학습"""
    model.train()
    total_loss = 0.0
    total_distance = 0.0
    num_batches = 0
    
    # Gradient clipping을 위한 max_norm 설정
    max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, mask)
        
        # Loss 계산
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (gradient explosion 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 통계
        with torch.no_grad():
            distance = euclidean_distance(outputs, targets).mean().item()
            total_loss += loss.item()
            total_distance += distance
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dist': f'{distance:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_distance = total_distance / num_batches
    
    return avg_loss, avg_distance


def validate(model, val_loader, criterion, device, config):
    """검증"""
    model.eval()
    total_loss = 0.0
    total_distance = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(features, mask)
            loss = criterion(outputs, targets)
            distance = euclidean_distance(outputs, targets).mean().item()
            
            total_loss += loss.item()
            total_distance += distance
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dist': f'{distance:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_distance = total_distance / num_batches
    
    return avg_loss, avg_distance


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"체크포인트 저장: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """체크포인트 로드"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"체크포인트 로드: {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss

