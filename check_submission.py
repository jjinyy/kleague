# -*- coding: utf-8 -*-
"""제출 파일 확인"""
import pandas as pd

submission = pd.read_csv('output/submissions/submission.csv')
sample = pd.read_csv('open_track1/sample_submission.csv')

print("=" * 50)
print("제출 파일 검증")
print("=" * 50)

print(f"\n제출 파일:")
print(f"  총 행 수: {len(submission)}")
print(f"  샘플 파일 행 수: {len(sample)}")

# 빈 값 확인
empty_x = submission['end_x'].isna() | (submission['end_x'] == '')
empty_y = submission['end_y'].isna() | (submission['end_y'] == '')
empty_both = empty_x | empty_y

print(f"\n빈 값 확인:")
print(f"  end_x 빈 값: {empty_x.sum()}개")
print(f"  end_y 빈 값: {empty_y.sum()}개")
print(f"  둘 다 빈 값: {empty_both.sum()}개")

if empty_both.sum() > 0:
    print(f"\n빈 값이 있는 에피소드:")
    empty_episodes = submission[empty_both]['game_episode'].tolist()
    for i, ep in enumerate(empty_episodes[:10]):
        print(f"  {ep}")
    if len(empty_episodes) > 10:
        print(f"  ... 외 {len(empty_episodes) - 10}개")

# 값 범위 확인
valid_x = pd.to_numeric(submission['end_x'], errors='coerce')
valid_y = pd.to_numeric(submission['end_y'], errors='coerce')

print(f"\n값 범위:")
print(f"  end_x: [{valid_x.min():.2f}, {valid_x.max():.2f}]")
print(f"  end_y: [{valid_y.min():.2f}, {valid_y.max():.2f}]")

# 경기장 범위 체크 (0-105, 0-68)
out_of_range_x = (valid_x < 0) | (valid_x > 105)
out_of_range_y = (valid_y < 0) | (valid_y > 68)
out_of_range = out_of_range_x | out_of_range_y

print(f"\n경기장 범위 초과:")
print(f"  end_x 범위 초과: {out_of_range_x.sum()}개")
print(f"  end_y 범위 초과: {out_of_range_y.sum()}개")
print(f"  전체 범위 초과: {out_of_range.sum()}개")

# 샘플 파일과 비교
print(f"\n샘플 파일과 비교:")
if len(submission) != len(sample):
    print(f"  [경고] 행 수가 다릅니다!")
else:
    print(f"  행 수 일치: OK")

if not submission['game_episode'].equals(sample['game_episode']):
    print(f"  [경고] game_episode 순서가 다릅니다!")
else:
    print(f"  game_episode 순서 일치: OK")

