# -*- coding: utf-8 -*-
"""학습 완료 여부 확인"""
import os
import time
from datetime import datetime

model_path = 'output/models/best_model.pt'
checkpoint_10 = 'output/models/checkpoint_epoch_10.pt'
checkpoint_20 = 'output/models/checkpoint_epoch_20.pt'
checkpoint_30 = 'output/models/checkpoint_epoch_30.pt'
checkpoint_40 = 'output/models/checkpoint_epoch_40.pt'
checkpoint_50 = 'output/models/checkpoint_epoch_50.pt'

print("=" * 50)
print("학습 진행 상황 확인")
print("=" * 50)

checkpoints = [
    (checkpoint_10, 10),
    (checkpoint_20, 20),
    (checkpoint_30, 30),
    (checkpoint_40, 40),
    (checkpoint_50, 50)
]

latest_epoch = 0
for path, epoch in checkpoints:
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        print(f"Epoch {epoch}: 존재 (업데이트: {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')})")
        latest_epoch = epoch
    else:
        print(f"Epoch {epoch}: 아직 생성되지 않음")

if os.path.exists(model_path):
    mtime = os.path.getmtime(model_path)
    elapsed = time.time() - mtime
    print(f"\n최고 모델 (best_model.pt):")
    print(f"  마지막 업데이트: {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')}")
    if elapsed < 60:
        print(f"  경과 시간: {elapsed:.0f}초 전")
    elif elapsed < 3600:
        print(f"  경과 시간: {elapsed/60:.1f}분 전")
    else:
        print(f"  경과 시간: {elapsed/3600:.1f}시간 전")
    
    # 2분 이상 업데이트가 없으면 완료된 것으로 추정
    if elapsed > 120:
        print(f"\n[추정] 학습이 완료되었을 가능성이 높습니다.")
        print(f"       (2분 이상 업데이트가 없음)")
    else:
        print(f"\n[상태] 학습이 진행 중입니다.")
else:
    print("\n모델 파일이 아직 생성되지 않았습니다.")

