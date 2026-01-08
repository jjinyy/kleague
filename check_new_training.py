# -*- coding: utf-8 -*-
"""새로운 학습 진행 상황 확인"""
import os
import time
from datetime import datetime

model_path = 'output/models/best_model.pt'
checkpoint_dir = 'output/models'

print("=" * 70)
print("학습 진행 상황 확인")
print("=" * 70)

# 최근 체크포인트 확인
if os.path.exists(checkpoint_dir):
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
            epoch_num = int(file.replace('checkpoint_epoch_', '').replace('.pt', ''))
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            checkpoints.append((epoch_num, mtime, file_path))
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[1], reverse=True)  # 최신순
        print(f"\n최근 체크포인트 (최신순):")
        for epoch, mtime, path in checkpoints[:5]:
            elapsed = (time.time() - mtime) / 60
            size_mb = os.path.getsize(path) / 1024 / 1024
            if elapsed < 60:
                print(f"  Epoch {epoch}: {elapsed:.0f}초 전 (크기: {size_mb:.2f} MB)")
            else:
                print(f"  Epoch {epoch}: {elapsed:.1f}분 전 (크기: {size_mb:.2f} MB)")
        
        latest = checkpoints[0]
        latest_elapsed = (time.time() - latest[1]) / 60
        
        if latest_elapsed < 5:  # 5분 이내
            print(f"\n[진행 중] 최신 체크포인트가 {latest_elapsed:.1f}분 전에 생성되었습니다.")
            print(f"학습이 진행 중인 것으로 보입니다!")
        elif latest_elapsed < 60:
            print(f"\n[대기 중] 최신 체크포인트가 {latest_elapsed:.1f}분 전에 생성되었습니다.")
            print(f"학습이 완료되었거나 잠시 멈춘 상태일 수 있습니다.")
        else:
            print(f"\n[완료/중단] 최신 체크포인트가 {latest_elapsed:.1f}분 전에 생성되었습니다.")
            print(f"학습이 완료되었거나 중단되었을 가능성이 높습니다.")
    else:
        print("\n체크포인트가 아직 생성되지 않았습니다.")
        print("학습이 막 시작되었을 수 있습니다.")

# 모델 파일 확인
if os.path.exists(model_path):
    mtime = os.path.getmtime(model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    elapsed = (time.time() - mtime) / 60
    
    print(f"\n모델 파일:")
    print(f"  크기: {size_mb:.2f} MB")
    if elapsed < 60:
        print(f"  마지막 업데이트: {elapsed:.0f}초 전")
    else:
        print(f"  마지막 업데이트: {elapsed:.1f}분 전")
    
    # 크기 비교 (개선된 모델은 더 클 것으로 예상)
    if size_mb > 100:
        print(f"  [개선된 모델] 모델 크기가 큽니다 ({size_mb:.2f} MB)")
    elif size_mb > 50:
        print(f"  [일반 모델] 모델 크기: {size_mb:.2f} MB")

print(f"\n{'='*70}")

