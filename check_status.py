# -*- coding: utf-8 -*-
"""학습 및 추론 상태 확인"""
import os
import time
from datetime import datetime

model_path = 'output/models/best_model.pt'
submission_path = 'output/submissions/submission.csv'

print("=" * 50)
print("학습 및 추론 상태 확인")
print("=" * 50)

# 모델 확인
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
    elif elapsed < 3600:
        print(f"  경과 시간: {elapsed/60:.1f}분 전")
    else:
        print(f"  경과 시간: {elapsed/3600:.1f}시간 전")
    
    # 2분 이상 업데이트 없으면 완료로 간주
    if elapsed > 120:
        print(f"  상태: 학습 완료된 것으로 보입니다")
    else:
        print(f"  상태: 학습 진행 중일 수 있습니다")
else:
    print(f"\n모델 파일: 없음")

# 제출 파일 확인
if os.path.exists(submission_path):
    import pandas as pd
    df = pd.read_csv(submission_path)
    mtime = os.path.getmtime(submission_path)
    
    print(f"\n제출 파일:")
    print(f"  존재: 예")
    print(f"  행 수: {len(df)}")
    print(f"  마지막 수정: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 빈 값 확인
    empty = df['end_x'].isna().sum() + df['end_y'].isna().sum()
    print(f"  빈 값: {empty}개")
    
    if empty == 0:
        print(f"  상태: 정상 (모든 값이 채워짐)")
    else:
        print(f"  상태: 경고 (빈 값 존재)")
else:
    print(f"\n제출 파일: 없음")

