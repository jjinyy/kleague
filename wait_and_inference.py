# -*- coding: utf-8 -*-
"""학습 완료 대기 후 추론 실행"""
import time
import os
import subprocess
from datetime import datetime

model_path = 'output/models/best_model.pt'
check_interval = 30  # 30초마다 확인

print("학습 완료를 기다리는 중...")
print(f"모델 파일: {model_path}")
print(f"체크 간격: {check_interval}초\n")

last_mtime = None
stable_count = 0
required_stable = 2  # 2번 연속 변경 없으면 완료로 간주

while True:
    if os.path.exists(model_path):
        current_mtime = os.path.getmtime(model_path)
        
        if last_mtime is None:
            last_mtime = current_mtime
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일 발견, 변경 감지 중...")
        elif current_mtime == last_mtime:
            stable_count += 1
            if stable_count >= required_stable:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 학습이 완료된 것으로 보입니다.")
                print("추론을 시작합니다...\n")
                break
        else:
            last_mtime = current_mtime
            stable_count = 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일 업데이트됨...")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일을 기다리는 중...")
    
    time.sleep(check_interval)

# 추론 실행
print("=" * 50)
print("추론 실행")
print("=" * 50)
subprocess.run(['python', 'inference.py'])

