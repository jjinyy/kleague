# -*- coding: utf-8 -*-
"""빠른 상태 확인"""
import os
import time
from datetime import datetime

checkpoint_dir = 'output/models'
model_path = 'output/models/best_model.pt'

print("=" * 60)
print("학습 진행 상황 빠른 확인")
print("=" * 60)
print(f"확인 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 최신 파일 찾기
latest_file = None
latest_time = 0
latest_size = 0

if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            if mtime > latest_time:
                latest_time = mtime
                latest_file = file
                latest_size = os.path.getsize(file_path) / 1024 / 1024

if latest_file:
    elapsed_minutes = (time.time() - latest_time) / 60
    elapsed_seconds = (time.time() - latest_time)
    
    print(f"최신 파일: {latest_file}")
    print(f"파일 크기: {latest_size:.2f} MB")
    print(f"마지막 업데이트: {datetime.fromtimestamp(latest_time).strftime('%H:%M:%S')}")
    
    if elapsed_seconds < 60:
        print(f"경과 시간: {elapsed_seconds:.0f}초 전")
        print(f"\n[진행 중] 학습이 진행 중입니다!")
    elif elapsed_minutes < 5:
        print(f"경과 시간: {elapsed_minutes:.1f}분 전")
        print(f"\n[진행 중] 학습이 진행 중일 가능성이 높습니다!")
    elif elapsed_minutes < 60:
        print(f"경과 시간: {elapsed_minutes:.1f}분 전")
        print(f"\n[대기 중] 학습이 완료되었거나 잠시 멈춘 상태일 수 있습니다.")
    else:
        print(f"경과 시간: {elapsed_minutes:.1f}분 전 ({elapsed_minutes/60:.1f}시간 전)")
        print(f"\n[완료/중단] 학습이 완료되었거나 중단되었을 가능성이 높습니다.")
        
        if elapsed_minutes > 1000:  # 약 16시간 이상
            print(f"\n[경고] 오래 전에 생성된 파일입니다.")
            print(f"   새로운 학습을 시작해야 할 수 있습니다.")
else:
    print("모델 파일이 없습니다.")

# 체크포인트 개수 확인
if os.path.exists(checkpoint_dir):
    checkpoint_count = len([f for f in os.listdir(checkpoint_dir) if 'checkpoint_epoch_' in f])
    print(f"\n체크포인트 개수: {checkpoint_count}개")
    
    if checkpoint_count > 0:
        epochs = []
        for file in os.listdir(checkpoint_dir):
            if 'checkpoint_epoch_' in file:
                epoch = int(file.replace('checkpoint_epoch_', '').replace('.pt', ''))
                epochs.append(epoch)
        if epochs:
            print(f"최고 에폭: {max(epochs)}")

print(f"\n{'='*60}")

