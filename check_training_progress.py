# -*- coding: utf-8 -*-
"""학습 진행 상황 확인"""
import os
import time
from datetime import datetime

model_path = 'output/models/best_model.pt'
checkpoint_dir = 'output/models'

print("=" * 60)
print("학습 진행 상황 확인")
print("=" * 60)

# 모델 파일 확인
if os.path.exists(model_path):
    mtime = os.path.getmtime(model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    elapsed = time.time() - mtime
    
    print(f"\n모델 파일:")
    print(f"  존재: 예")
    print(f"  크기: {size_mb:.2f} MB")
    print(f"  마지막 수정: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if elapsed < 60:
        print(f"  경과 시간: {elapsed:.0f}초 전")
        print(f"  상태: 학습 진행 중일 가능성 높음")
    elif elapsed < 300:  # 5분 이내
        print(f"  경과 시간: {elapsed/60:.1f}분 전")
        print(f"  상태: 학습 진행 중일 수 있음")
    else:
        print(f"  경과 시간: {elapsed/60:.1f}분 전")
        print(f"  상태: 학습이 완료되었거나 중단되었을 수 있음")
else:
    print(f"\n모델 파일: 아직 생성되지 않음 (학습 시작 대기 중)")

# 체크포인트 확인
if os.path.exists(checkpoint_dir):
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
            epoch_num = int(file.replace('checkpoint_epoch_', '').replace('.pt', ''))
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            elapsed = time.time() - mtime
            checkpoints.append((epoch_num, elapsed))
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        print(f"\n체크포인트:")
        for epoch, elapsed in checkpoints[-5:]:  # 최근 5개만 표시
            if elapsed < 60:
                print(f"  Epoch {epoch}: {elapsed:.0f}초 전")
            else:
                print(f"  Epoch {epoch}: {elapsed/60:.1f}분 전")
        
        latest_epoch = checkpoints[-1][0]
        latest_elapsed = checkpoints[-1][1]
        
        if latest_elapsed < 300:  # 5분 이내
            print(f"\n최신 체크포인트: Epoch {latest_epoch}")
            print(f"학습이 진행 중인 것으로 보입니다!")
        else:
            print(f"\n최신 체크포인트: Epoch {latest_epoch}")
            print(f"학습이 완료되었거나 중단되었을 수 있습니다.")
    else:
        print(f"\n체크포인트: 아직 생성되지 않음")

print(f"\n{'='*60}")

