# -*- coding: utf-8 -*-
"""
학습 완료 대기 후 추론 및 Git 커밋 자동화 스크립트
"""
import os
import time
import subprocess
import sys
from datetime import datetime

model_path = 'output/models/best_model.pt'
check_interval = 30  # 30초마다 확인
required_stable_minutes = 3  # 3분 동안 변경 없으면 완료로 간주

print("=" * 70)
print("자동 학습 완료 대기 및 추론 실행")
print("=" * 70)
print(f"모델 파일: {model_path}")
print(f"체크 간격: {check_interval}초")
print(f"완료 판단: {required_stable_minutes}분 동안 변경 없으면 완료로 간주\n")

last_mtime = None
stable_start_time = None

print(f"[{datetime.now().strftime('%H:%M:%S')}] 학습 완료를 기다리는 중...\n")

while True:
    if os.path.exists(model_path):
        current_mtime = os.path.getmtime(model_path)
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        
        if last_mtime is None:
            last_mtime = current_mtime
            stable_start_time = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일 발견")
            print(f"  크기: {size_mb:.2f} MB")
            print(f"  변경 감지 중...\n")
        elif current_mtime == last_mtime:
            # 변경 없음
            elapsed_stable = time.time() - stable_start_time
            if elapsed_stable >= required_stable_minutes * 60:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 학습 완료 확인!")
                print(f"  {required_stable_minutes}분 이상 변경 없음")
                print(f"  모델 크기: {size_mb:.2f} MB")
                break
            else:
                remaining = required_stable_minutes * 60 - elapsed_stable
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 안정화 중... ({remaining/60:.1f}분 남음)")
        else:
            # 변경 감지됨
            last_mtime = current_mtime
            stable_start_time = time.time()
            elapsed = time.time() - current_mtime
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일 업데이트됨")
            print(f"  크기: {size_mb:.2f} MB")
            print(f"  경과 시간: {elapsed/60:.1f}분\n")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델 파일을 기다리는 중...")
        stable_start_time = time.time()
    
    time.sleep(check_interval)

# 추론 실행
print("\n" + "=" * 70)
print("추론 실행")
print("=" * 70 + "\n")

result = subprocess.run([sys.executable, 'inference.py'], 
                       capture_output=False, 
                       text=True)

if result.returncode == 0:
    print("\n추론 완료!")
    
    # Git 커밋
    print("\n" + "=" * 70)
    print("Git에 커밋")
    print("=" * 70 + "\n")
    
    # 변경사항 추가
    subprocess.run(['git', 'add', '-A'], check=True)
    
    # 커밋 메시지
    commit_msg = f"Update: 개선된 모델 학습 완료 및 최신 제출 파일\n\n- 피처 강화: 33개 피처 (선수/액션 정보, 시퀀스 진행률 추가)\n- 모델 개선: 어텐션 헤드 증가, Residual connection 강화\n- 손실 함수: Combined Loss (Huber + MSE + Euclidean)\n- 데이터 분할: 게임 ID 기준 분할로 데이터 누수 방지\n- 학습 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
    
    # 푸시
    subprocess.run(['git', 'push'], check=True)
    
    print("\n" + "=" * 70)
    print("모든 작업 완료!")
    print("=" * 70)
    print(f"제출 파일: output/submissions/submission.csv")
    print(f"GitHub에 푸시 완료")
else:
    print("\n추론 중 오류 발생!")

