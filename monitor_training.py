# -*- coding: utf-8 -*-
"""학습 모니터링 스크립트"""
import os
import time
from datetime import datetime

model_path = 'output/models/best_model.pt'
check_interval = 30  # 30초마다 확인

print("=" * 60)
print("학습 모니터링 시작")
print("=" * 60)
print(f"모델 파일: {model_path}")
print(f"체크 간격: {check_interval}초\n")

last_mtime = None
stable_count = 0
required_stable = 3  # 3번 연속 변경 없으면 완료로 간주

while True:
    if os.path.exists(model_path):
        current_mtime = os.path.getmtime(model_path)
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        
        if last_mtime is None:
            last_mtime = current_mtime
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일 발견")
            print(f"  크기: {size_mb:.2f} MB")
            print(f"  변경 감지 중...\n")
        elif current_mtime == last_mtime:
            stable_count += 1
            elapsed = time.time() - current_mtime
            if stable_count >= required_stable:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 학습이 완료된 것으로 보입니다.")
                print(f"  마지막 업데이트로부터 {elapsed/60:.1f}분 경과")
                print(f"  모델 크기: {size_mb:.2f} MB")
                break
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일 안정화 중... ({stable_count}/{required_stable})")
        else:
            last_mtime = current_mtime
            stable_count = 0
            elapsed = time.time() - current_mtime
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일 업데이트됨")
            print(f"  크기: {size_mb:.2f} MB")
            print(f"  경과 시간: {elapsed/60:.1f}분\n")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일을 기다리는 중...")
    
    time.sleep(check_interval)

print("\n" + "=" * 60)
print("학습 완료! 추론을 시작합니다...")
print("=" * 60 + "\n")

