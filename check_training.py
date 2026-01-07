# -*- coding: utf-8 -*-
"""학습 상태 확인 스크립트"""
import os
import time
from datetime import datetime

model_path = 'output/models/best_model.pt'

if os.path.exists(model_path):
    mtime = os.path.getmtime(model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"모델 파일 존재: 예")
    print(f"최종 수정 시간: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"파일 크기: {size_mb:.2f} MB")
    print(f"\n현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    elapsed = time.time() - mtime
    if elapsed < 60:
        print(f"마지막 업데이트로부터: {elapsed:.0f}초 전")
    elif elapsed < 3600:
        print(f"마지막 업데이트로부터: {elapsed/60:.1f}분 전")
    else:
        print(f"마지막 업데이트로부터: {elapsed/3600:.1f}시간 전")
else:
    print("모델 파일이 아직 생성되지 않았습니다.")

