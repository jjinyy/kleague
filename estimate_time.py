# -*- coding: utf-8 -*-
"""학습 시간 추정"""
import pandas as pd

# 첫 번째 에폭 기준 시간 (실제 측정값)
train_batches = 351
val_batches = 86
train_time_per_epoch = 8  # 초
val_time_per_epoch = 1  # 초
total_time_per_epoch = train_time_per_epoch + val_time_per_epoch

# 설정값
max_epochs = 50
early_stopping_patience = 10

print("=" * 50)
print("학습 시간 추정")
print("=" * 50)
print(f"\n첫 번째 에폭 기준:")
print(f"  - 학습 배치 수: {train_batches}")
print(f"  - 검증 배치 수: {val_batches}")
print(f"  - 1 에폭당 소요 시간: 약 {total_time_per_epoch}초")

print(f"\n최대 에폭: {max_epochs}")
print(f"Early Stopping Patience: {early_stopping_patience}")

print(f"\n예상 소요 시간:")
min_epochs = 10
avg_epochs = 25
max_epochs_actual = 50

print(f"  - 최소 (약 {min_epochs} 에폭): {min_epochs * total_time_per_epoch / 60:.1f}분")
print(f"  - 일반 (약 {avg_epochs} 에폭): {avg_epochs * total_time_per_epoch / 60:.1f}분")
print(f"  - 최대 ({max_epochs_actual} 에폭): {max_epochs_actual * total_time_per_epoch / 60:.1f}분")

print(f"\n참고: 학습이 진행되면서 속도가 약간 느려질 수 있습니다.")
print(f"      실제 시간은 위 추정치보다 10-20% 더 걸릴 수 있습니다.")

