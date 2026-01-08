# -*- coding: utf-8 -*-
"""학습 완료 확인 및 추론 실행"""
import os
import time
import subprocess
import sys
from datetime import datetime

model_path = 'output/models/best_model.pt'
checkpoint_dir = 'output/models'

print("=" * 70)
print("학습 완료 확인")
print("=" * 70)

# 최신 파일 확인
latest_file = None
latest_time = 0

if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            if mtime > latest_time:
                latest_time = mtime
                latest_file = file

if latest_file:
    elapsed_minutes = (time.time() - latest_time) / 60
    
    print(f"\n최신 파일: {latest_file}")
    print(f"마지막 업데이트: {datetime.fromtimestamp(latest_time).strftime('%H:%M:%S')}")
    print(f"경과 시간: {elapsed_minutes:.1f}분")
    
    # 10분 이상 변경 없으면 완료로 간주
    if elapsed_minutes >= 10:
        print(f"\n[완료] 학습이 완료된 것으로 보입니다.")
        print(f"추론을 시작합니다...\n")
        
        # 추론 실행
        result = subprocess.run([sys.executable, 'inference.py'], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\n추론 완료!")
            
            # Git 커밋
            print("\n" + "=" * 70)
            print("Git에 커밋")
            print("=" * 70 + "\n")
            
            subprocess.run(['git', 'add', '-A'], check=True)
            
            commit_msg = f"Update: 개선된 모델 학습 완료 (Epoch 30) 및 최신 제출 파일\n\n- 피처: 34개 (선수/액션 정보, 시퀀스 진행률 추가)\n- 모델: hidden_dim 512, 4 layers, 어텐션 헤드 16개\n- 손실 함수: Combined Loss (Huber + MSE + Euclidean)\n- 데이터 분할: 게임 ID 기준 분할\n- 학습 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            subprocess.run(['git', 'push'], check=True)
            
            print("\n모든 작업 완료!")
        else:
            print("\n추론 중 오류 발생!")
    else:
        print(f"\n[진행 중] 학습이 아직 진행 중일 수 있습니다.")
        print(f"{10 - elapsed_minutes:.1f}분 후 다시 확인하세요.")
else:
    print("\n모델 파일이 없습니다.")

