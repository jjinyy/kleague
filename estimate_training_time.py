# -*- coding: utf-8 -*-
"""학습 완료 예상 시간 계산"""
import os
import time
from datetime import datetime, timedelta

model_path = 'output/models/best_model.pt'
checkpoint_dir = 'output/models'

# 현재 상태 확인
if os.path.exists(model_path):
    model_mtime = os.path.getmtime(model_path)
    elapsed_minutes = (time.time() - model_mtime) / 60
    print(f"모델 파일 마지막 업데이트: {elapsed_minutes:.1f}분 전")
else:
    print("모델 파일이 아직 생성되지 않았습니다.")
    elapsed_minutes = 0

# 체크포인트 확인
checkpoints = []
if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
            epoch_num = int(file.replace('checkpoint_epoch_', '').replace('.pt', ''))
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            checkpoints.append((epoch_num, mtime))

if checkpoints:
    checkpoints.sort(key=lambda x: x[0])
    latest_epoch = checkpoints[-1][0]
    latest_mtime = checkpoints[-1][0]
    
    print(f"\n생성된 체크포인트:")
    for epoch, mtime in checkpoints:
        elapsed = (time.time() - mtime) / 60
        print(f"  Epoch {epoch}: {elapsed:.1f}분 전")
    
    # 에폭 간 시간 계산
    if len(checkpoints) >= 2:
        time_diffs = []
        for i in range(1, len(checkpoints)):
            epoch_diff = checkpoints[i][0] - checkpoints[i-1][0]
            time_diff = checkpoints[i][1] - checkpoints[i-1][1]
            if epoch_diff > 0:
                time_per_epoch = time_diff / epoch_diff
                time_diffs.append(time_per_epoch)
        
        if time_diffs:
            avg_time_per_epoch = sum(time_diffs) / len(time_diffs)
            print(f"\n평균 1 에폭당 소요 시간: {avg_time_per_epoch:.1f}초")
            
            # 남은 에폭 추정
            # Early Stopping patience가 15이므로, 최악의 경우 현재 에폭 + 15
            # 하지만 일반적으로는 더 빨리 완료됨
            current_epoch = latest_epoch
            max_epochs = 100
            early_stopping_patience = 15
            
            # 최악의 경우
            worst_case_epochs = min(current_epoch + early_stopping_patience, max_epochs)
            worst_case_time = (worst_case_epochs - current_epoch) * avg_time_per_epoch / 60
            
            # 일반적인 경우 (현재 에폭 + 5-10 정도)
            typical_epochs = current_epoch + 8
            typical_time = (typical_epochs - current_epoch) * avg_time_per_epoch / 60
            
            print(f"\n예상 완료 시간:")
            print(f"  현재 에폭: {current_epoch}")
            print(f"  일반적인 경우: 약 {typical_time:.1f}분 후 (에폭 {typical_epochs} 정도)")
            print(f"  최악의 경우: 약 {worst_case_time:.1f}분 후 (에폭 {worst_case_epochs} 정도)")
            
            # 모델이 더 크므로 시간이 더 걸릴 수 있음
            print(f"\n참고:")
            print(f"  - 개선된 모델은 더 크므로 실제 시간이 더 걸릴 수 있습니다")
            print(f"  - 배치 크기가 커서 속도 향상 효과가 있을 수 있습니다")
            print(f"  - Early Stopping으로 인해 실제로는 더 빨리 완료될 수 있습니다")
        else:
            print("\n에폭 간 시간을 계산할 수 없습니다.")
    else:
        print(f"\n현재 에폭: {latest_epoch}")
        print("에폭 간 시간을 계산하려면 최소 2개의 체크포인트가 필요합니다.")
else:
    print("\n체크포인트가 아직 생성되지 않았습니다.")

