# -*- coding: utf-8 -*-
"""학습 완료 예상 시간 계산"""
import os
import time
from datetime import datetime, timedelta

checkpoint_dir = 'output/models'
model_path = 'output/models/best_model.pt'

print("=" * 70)
print("학습 완료 예상 시간 계산")
print("=" * 70)

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
    print(f"\n생성된 체크포인트:")
    for epoch, mtime in checkpoints[-5:]:  # 최근 5개만
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
            current_epoch = checkpoints[-1][0]
            
            print(f"\n현재 상태:")
            print(f"  최신 에폭: {current_epoch}")
            print(f"  평균 1 에폭당 소요 시간: {avg_time_per_epoch:.1f}초")
            
            # 개선된 모델은 더 크므로 시간이 더 걸릴 수 있음
            # 배치 크기가 커서 속도 향상 가능
            estimated_time_per_epoch = avg_time_per_epoch * 1.2  # 20% 증가 가정
            
            # Early Stopping 고려
            early_stopping_patience = 15
            max_epochs = 100
            
            # 일반적인 경우 (현재 + 5-10 에폭)
            typical_epochs = min(current_epoch + 8, max_epochs)
            typical_time_minutes = (typical_epochs - current_epoch) * estimated_time_per_epoch / 60
            
            # 최악의 경우
            worst_epochs = min(current_epoch + early_stopping_patience, max_epochs)
            worst_time_minutes = (worst_epochs - current_epoch) * estimated_time_per_epoch / 60
            
            print(f"\n예상 완료 시간:")
            print(f"  일반적인 경우: 약 {typical_time_minutes:.1f}분 후")
            print(f"    (에폭 {typical_epochs} 정도, {typical_time_minutes*60:.0f}초)")
            print(f"  최악의 경우: 약 {worst_time_minutes:.1f}분 후")
            print(f"    (에폭 {worst_epochs} 정도, {worst_time_minutes*60:.0f}초)")
            
            # 예상 완료 시각
            now = datetime.now()
            typical_completion = now + timedelta(minutes=typical_time_minutes)
            worst_completion = now + timedelta(minutes=worst_time_minutes)
            
            print(f"\n예상 완료 시각:")
            print(f"  일반적인 경우: {typical_completion.strftime('%H:%M:%S')}")
            print(f"  최악의 경우: {worst_completion.strftime('%H:%M:%S')}")
            
            print(f"\n참고:")
            print(f"  - 개선된 모델은 더 크므로 실제 시간이 더 걸릴 수 있습니다")
            print(f"  - 배치 크기 증가로 속도 향상 효과가 있을 수 있습니다")
            print(f"  - Early Stopping으로 인해 더 빨리 완료될 수 있습니다")
        else:
            print("\n에폭 간 시간을 계산할 수 없습니다.")
    else:
        print(f"\n현재 에폭: {checkpoints[0][0]}")
        print("에폭 간 시간을 계산하려면 최소 2개의 체크포인트가 필요합니다.")
else:
    print("\n체크포인트가 아직 생성되지 않았습니다.")
    print("학습이 막 시작되었거나 진행 중입니다.")
    print("\n예상 소요 시간:")
    print("  - 첫 에폭: 약 1-2분")
    print("  - 전체 학습: 약 10-20분 (Early Stopping 고려)")

