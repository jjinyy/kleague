# -*- coding: utf-8 -*-
"""실시간 학습 모니터링"""
import os
import time
from datetime import datetime

model_path = 'output/models/best_model.pt'
checkpoint_dir = 'output/models'

print("=" * 70)
print("실시간 학습 모니터링")
print("=" * 70)
print(f"시작 시간: {datetime.now().strftime('%H:%M:%S')}\n")

initial_best_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else None
initial_checkpoints = set()

if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            initial_checkpoints.add(file)

print("초기 상태:")
if initial_best_mtime:
    print(f"  Best Model: {datetime.fromtimestamp(initial_best_mtime).strftime('%H:%M:%S')}")
print(f"  체크포인트 파일 수: {len(initial_checkpoints)}")
print(f"\n변경 감지 중... (30초마다 확인)\n")

check_count = 0
while True:
    check_count += 1
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Best Model 확인
    if os.path.exists(model_path):
        current_mtime = os.path.getmtime(model_path)
        if initial_best_mtime and current_mtime != initial_best_mtime:
            elapsed = time.time() - current_mtime
            print(f"[{current_time}] ✓ Best Model 업데이트됨! ({elapsed:.0f}초 전)")
            initial_best_mtime = current_mtime
        elif not initial_best_mtime:
            initial_best_mtime = current_mtime
            print(f"[{current_time}] Best Model 파일 발견")
    
    # 새로운 체크포인트 확인
    if os.path.exists(checkpoint_dir):
        current_checkpoints = set()
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt'):
                current_checkpoints.add(file)
        
        new_checkpoints = current_checkpoints - initial_checkpoints
        if new_checkpoints:
            for cp in new_checkpoints:
                cp_path = os.path.join(checkpoint_dir, cp)
                mtime = os.path.getmtime(cp_path)
                elapsed = time.time() - mtime
                print(f"[{current_time}] ✓ 새로운 체크포인트 생성: {cp} ({elapsed:.0f}초 전)")
            initial_checkpoints = current_checkpoints
    
    # 진행 상황 출력 (5번마다)
    if check_count % 5 == 0:
        print(f"[{current_time}] 모니터링 중... (체크 {check_count}회)")
    
    time.sleep(30)

