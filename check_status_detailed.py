# -*- coding: utf-8 -*-
"""상세 학습 상태 확인"""
import os
import time
from datetime import datetime

checkpoint_dir = 'output/models'
model_path = 'output/models/best_model.pt'

print("=" * 70)
print("학습 진행 상황 상세 확인")
print("=" * 70)

# 모든 체크포인트 확인
checkpoints = []
if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            elapsed_minutes = (time.time() - mtime) / 60
            
            if 'epoch' in file:
                epoch = int(file.replace('checkpoint_epoch_', '').replace('.pt', ''))
                checkpoints.append(('checkpoint', epoch, mtime, size_mb, elapsed_minutes))
            elif 'best' in file:
                checkpoints.append(('best', 0, mtime, size_mb, elapsed_minutes))

if checkpoints:
    checkpoints.sort(key=lambda x: x[2], reverse=True)  # 최신순
    
    print(f"\n모델 파일 목록 (최신순):")
    for file_type, epoch, mtime, size_mb, elapsed in checkpoints:
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        if file_type == 'checkpoint':
            print(f"  Epoch {epoch:2d}: {time_str} ({elapsed:.1f}분 전, {size_mb:.2f} MB)")
        else:
            print(f"  Best Model: {time_str} ({elapsed:.1f}분 전, {size_mb:.2f} MB)")
    
    latest = checkpoints[0]
    latest_elapsed = latest[4]
    
    print(f"\n최신 파일:")
    if latest[0] == 'checkpoint':
        print(f"  Epoch {latest[1]} 체크포인트")
    else:
        print(f"  Best Model")
    print(f"  생성 시간: {datetime.fromtimestamp(latest[2]).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  경과 시간: {latest_elapsed:.1f}분 전")
    print(f"  파일 크기: {latest[3]:.2f} MB")
    
    # 학습 상태 판단
    if latest_elapsed < 5:
        print(f"\n[진행 중] 학습이 진행 중인 것으로 보입니다!")
    elif latest_elapsed < 60:
        print(f"\n[대기 중] 최근에 업데이트되었습니다. 학습이 완료되었을 수 있습니다.")
    else:
        print(f"\n[완료/중단] 오래 전에 생성되었습니다.")
        print(f"  - 학습이 완료되었거나")
        print(f"  - 학습이 중단되었을 수 있습니다")
        print(f"  - 새로운 학습을 시작해야 할 수 있습니다")
else:
    print("\n모델 파일이 없습니다.")

# Python 프로세스 확인
import subprocess
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                          capture_output=True, text=True)
    python_count = result.stdout.count('python.exe')
    print(f"\n실행 중인 Python 프로세스: {python_count}개")
    if python_count > 0:
        print("  학습이 진행 중일 수 있습니다.")
    else:
        print("  실행 중인 학습 프로세스가 없습니다.")
except:
    print("\n프로세스 확인 실패")

print(f"\n{'='*70}")

