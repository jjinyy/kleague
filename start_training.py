# -*- coding: utf-8 -*-
"""학습 시작 및 모니터링"""
import subprocess
import sys
import os
import time
from datetime import datetime

print("=" * 70)
print("개선된 모델 학습 시작")
print("=" * 70)
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 학습 시작
print("학습 프로세스 시작...")
process = subprocess.Popen(
    [sys.executable, 'train.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print("학습이 시작되었습니다.")
print("진행 상황을 모니터링합니다...\n")

# 출력 모니터링
try:
    for line in process.stdout:
        print(line.rstrip())
        sys.stdout.flush()
except KeyboardInterrupt:
    print("\n학습이 중단되었습니다.")
    process.terminate()

process.wait()
print(f"\n학습 프로세스 종료 코드: {process.returncode}")

