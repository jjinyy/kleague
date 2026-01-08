# -*- coding: utf-8 -*-
"""
데이터 로더 및 전처리 모듈
"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, List, Tuple
import os


class PassSequenceDataset(Dataset):
    """패스 시퀀스 데이터셋"""
    
    def __init__(self, data: pd.DataFrame, config, is_train=True):
        self.data = data
        self.config = config
        self.is_train = is_train
        self.sequences = self._prepare_sequences()
        
    def _prepare_sequences(self) -> List[Dict]:
        """에피소드별로 시퀀스 데이터 준비"""
        sequences = []
        
        for game_episode, group in self.data.groupby('game_episode'):
            # 패스만 필터링 (최종 패스 예측이므로)
            passes = group[group['type_name'] == 'Pass'].copy()
            
            if len(passes) == 0:
                continue
                
            # 시간 순으로 정렬
            passes = passes.sort_values('time_seconds').reset_index(drop=True)
            
            # 마지막 패스의 end_x, end_y가 타겟
            if self.is_train:
                # 학습 데이터: 마지막 패스의 end_x, end_y가 정답
                # 최소 2개 이상의 패스가 있어야 함 (1개는 타겟, 나머지는 입력)
                if len(passes) < 2:
                    continue
                # 마지막 패스의 end_x, end_y가 유효한지 확인
                last_pass = passes.iloc[-1]
                if pd.isna(last_pass['end_x']) or pd.isna(last_pass['end_y']):
                    continue
                target = np.array([last_pass['end_x'], last_pass['end_y']], dtype=np.float32)
                # 마지막 패스를 제외한 시퀀스
                sequence = passes.iloc[:-1]
            else:
                # 테스트 데이터: 전체 시퀀스 사용하여 마지막 패스 예측
                # 최소 1개 이상의 패스가 있어야 함
                if len(passes) < 1:
                    continue
                # 마지막 패스의 end_x, end_y가 비어있어야 함 (예측 대상)
                last_pass = passes.iloc[-1]
                # end_x, end_y가 비어있거나 NaN인 경우만 예측 대상으로 간주
                if not (pd.isna(last_pass['end_x']) or pd.isna(last_pass['end_y']) or 
                       last_pass['end_x'] == '' or last_pass['end_y'] == ''):
                    # end_x, end_y가 있으면 마지막 패스를 제외하고 예측
                    if len(passes) < 2:
                        continue
                    sequence = passes.iloc[:-1]
                else:
                    # end_x, end_y가 비어있으면 전체 시퀀스 사용
                    sequence = passes
                target = None
                
            # 피처 추출
            features = self._extract_features(sequence, group)
            
            # 빈 피처 배열 체크
            if len(features) == 0:
                continue
            
            sequences.append({
                'game_episode': game_episode,
                'features': features,
                'target': target,
                'sequence_length': len(features)  # 실제 피처 개수 사용
            })
            
        return sequences
    
    def _extract_features(self, sequence: pd.DataFrame, full_episode: pd.DataFrame) -> np.ndarray:
        """
        시퀀스에서 피처 추출 (개선된 버전)
        
        Args:
            sequence: 패스 시퀀스 DataFrame
            full_episode: 전체 에피소드 DataFrame
        
        Returns:
            피처 배열 (n_features, feature_dim)
        """
        features_list = []
        
        # 시퀀스 전체 통계 계산 (맥락 정보)
        valid_passes = sequence[
            (~sequence['start_x'].isna()) & 
            (~sequence['start_y'].isna()) & 
            (~sequence['end_x'].isna()) & 
            (~sequence['end_y'].isna())
        ]
        
        if len(valid_passes) > 0:
            seq_mean_x = valid_passes['start_x'].mean()
            seq_mean_y = valid_passes['start_y'].mean()
            seq_std_x = valid_passes['start_x'].std() if len(valid_passes) > 1 else 0.0
            seq_std_y = valid_passes['start_y'].std() if len(valid_passes) > 1 else 0.0
            seq_max_x = valid_passes['start_x'].max()
            seq_min_x = valid_passes['start_x'].min()
            seq_max_y = valid_passes['start_y'].max()
            seq_min_y = valid_passes['start_y'].min()
        else:
            seq_mean_x = seq_mean_y = 52.5
            seq_std_x = seq_std_y = 0.0
            seq_max_x = seq_max_y = 105.0
            seq_min_x = seq_min_y = 0.0
        
        for idx, row in sequence.iterrows():
            # 기본 좌표 정보
            start_x = row['start_x']
            start_y = row['start_y']
            end_x = row['end_x']
            end_y = row['end_y']
            
            # NaN 체크 및 처리
            if pd.isna(start_x) or pd.isna(start_y):
                # 시작 좌표가 없으면 스킵
                continue
            if pd.isna(end_x) or pd.isna(end_y) or end_x == '' or end_y == '':
                # 종료 좌표가 없으면 이전 패스의 종료 좌표 사용 또는 스킵
                if idx > 0:
                    prev_row = sequence.iloc[idx-1]
                    if not (pd.isna(prev_row['end_x']) or pd.isna(prev_row['end_y'])):
                        end_x = prev_row['end_x']
                        end_y = prev_row['end_y']
                    else:
                        continue
                else:
                    continue
            
            # 패스 거리 및 각도
            pass_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            pass_angle = np.arctan2(end_y - start_y, end_x - start_x)
            
            # 시간 정보
            time_seconds = row['time_seconds']
            
            # 상대적 시간 (에피소드 시작 기준)
            if idx > 0:
                time_delta = time_seconds - sequence.iloc[idx-1]['time_seconds']
            else:
                time_delta = 0.0
            
            # 결과 정보 (성공/실패)
            result_successful = 1 if row['result_name'] == 'Successful' else 0
            
            # 팀 정보
            is_home = 1 if row['is_home'] else 0
            
            # 선수 및 액션 정보 (정규화)
            player_id = row['player_id']
            action_id = row['action_id']
            # 선수 ID를 해시하여 정규화 (선수 수가 많으므로)
            player_id_hash = hash(str(player_id)) % 10000 / 10000.0  # 0-1 범위로 정규화
            action_id_norm = action_id / 10000.0  # 액션 ID 정규화
            
            # 이전 패스와의 관계
            if idx > 0:
                prev_row = sequence.iloc[idx-1]
                prev_end_x = prev_row['end_x']
                prev_end_y = prev_row['end_y']
                
                # 연속성 (이전 패스 종료점에서 현재 패스 시작점까지의 거리)
                continuity = np.sqrt((start_x - prev_end_x)**2 + (start_y - prev_end_y)**2)
                
                # 방향 변화
                prev_angle = np.arctan2(prev_end_y - prev_row['start_y'], 
                                       prev_end_x - prev_row['start_x'])
                angle_change = pass_angle - prev_angle
            else:
                continuity = 0.0
                angle_change = 0.0
            
            # 경기장 위치 정보 (정규화)
            normalized_start_x = start_x / 105.0
            normalized_start_y = start_y / 68.0
            normalized_end_x = end_x / 105.0
            normalized_end_y = end_y / 68.0
            
            # 시퀀스 내 상대적 위치 (맥락 정보)
            rel_to_mean_x = (start_x - seq_mean_x) / (seq_std_x + 1e-8) if seq_std_x > 0 else 0.0
            rel_to_mean_y = (start_y - seq_mean_y) / (seq_std_y + 1e-8) if seq_std_y > 0 else 0.0
            rel_to_max_x = (start_x - seq_max_x) / 105.0
            rel_to_max_y = (start_y - seq_max_y) / 68.0
            
            # 시퀀스 진행률 (현재 패스가 시퀀스의 어느 위치인지)
            sequence_progress = idx / max(len(sequence) - 1, 1)  # 0-1 범위
            
            # 경기장 영역 정보 (공격/수비/중앙)
            attack_zone = 1.0 if start_x > 70 else (0.5 if start_x > 35 else 0.0)
            defensive_zone = 1.0 if start_x < 35 else (0.5 if start_x < 70 else 0.0)
            center_zone = 1.0 if 30 < start_y < 38 else 0.0
            
            # 패스 방향성 (전진/후진/횡)
            forward_pass = 1.0 if (end_x - start_x) > 5 else 0.0
            backward_pass = 1.0 if (end_x - start_x) < -5 else 0.0
            lateral_pass = 1.0 if abs(end_x - start_x) < 5 else 0.0
            
            # 피처 벡터 구성 (강화된 버전 - 30개 피처)
            feature = np.array([
                # 기본 좌표 (4)
                normalized_start_x,
                normalized_start_y,
                normalized_end_x,
                normalized_end_y,
                # 패스 특성 (3)
                pass_distance / 105.0,  # 정규화된 거리
                pass_angle,
                pass_angle / np.pi,  # 정규화된 각도
                # 시간 정보 (2)
                time_seconds / 3600.0,  # 정규화된 시간
                time_delta / 10.0,  # 정규화된 시간 델타
                # 카테고리 정보 (2)
                result_successful,
                is_home,
                # 연속성 정보 (2)
                continuity / 105.0,
                angle_change,
                # 삼각함수 변환 (2)
                np.sin(pass_angle),
                np.cos(pass_angle),
                # 중앙선 기준 상대 위치 (4)
                start_x - 52.5,
                start_y - 34.0,
                end_x - 52.5,
                end_y - 34.0,
                # 방향 성분 (2)
                pass_distance * np.cos(pass_angle),
                pass_distance * np.sin(pass_angle),
                # 시퀀스 맥락 정보 (4)
                rel_to_mean_x,
                rel_to_mean_y,
                rel_to_max_x,
                rel_to_max_y,
                # 경기장 영역 정보 (3)
                attack_zone,
                defensive_zone,
                center_zone,
                # 패스 방향성 (3)
                forward_pass,
                backward_pass,
                lateral_pass,
                # 선수 및 액션 정보 (2)
                player_id_hash,
                action_id_norm,
                # 시퀀스 진행률 (1)
                sequence_progress,
            ], dtype=np.float32)
            
            features_list.append(feature)
        
        return np.array(features_list)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        features = seq_data['features']
        sequence_length = seq_data['sequence_length']
        
        # 패딩 처리
        if len(features) < self.config.max_sequence_length:
            padding = np.zeros((self.config.max_sequence_length - len(features), 
                              features.shape[1]), dtype=np.float32)
            features = np.vstack([features, padding])
            mask = np.ones(self.config.max_sequence_length, dtype=np.float32)
            mask[len(seq_data['features']):] = 0
        else:
            # 시퀀스가 너무 길면 자르기
            features = features[-self.config.max_sequence_length:]
            mask = np.ones(self.config.max_sequence_length, dtype=np.float32)
            sequence_length = self.config.max_sequence_length
        
        features_tensor = torch.FloatTensor(features)
        mask_tensor = torch.FloatTensor(mask)
        
        if self.is_train:
            target = torch.FloatTensor(seq_data['target'])
            return {
                'features': features_tensor,
                'mask': mask_tensor,
                'target': target,
                'game_episode': seq_data['game_episode'],
                'sequence_length': sequence_length
            }
        else:
            return {
                'features': features_tensor,
                'mask': mask_tensor,
                'game_episode': seq_data['game_episode'],
                'sequence_length': sequence_length
            }


def load_train_data(config) -> pd.DataFrame:
    """학습 데이터 로드"""
    print(f"학습 데이터 로딩: {config.TRAIN_DATA_PATH}")
    df = pd.read_csv(config.TRAIN_DATA_PATH, encoding='utf-8')
    print(f"학습 데이터 크기: {len(df)} 행")
    return df


def load_test_data(config) -> pd.DataFrame:
    """테스트 데이터 로드"""
    print(f"테스트 데이터 로딩: {config.TEST_DATA_PATH}")
    test_info = pd.read_csv(config.TEST_DATA_PATH, encoding='utf-8')
    
    # 각 에피소드 CSV 파일 로드
    all_episodes = []
    for _, row in test_info.iterrows():
        episode_path = os.path.join(config.DATA_DIR, row['path'].replace('./', ''))
        if os.path.exists(episode_path):
            episode_df = pd.read_csv(episode_path, encoding='utf-8')
            all_episodes.append(episode_df)
    
    if len(all_episodes) > 0:
        test_df = pd.concat(all_episodes, ignore_index=True)
        print(f"테스트 데이터 크기: {len(test_df)} 행")
        return test_df
    else:
        print("테스트 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()


def create_data_loaders(train_df: pd.DataFrame, config, val_split=0.2):
    """데이터 로더 생성"""
    # 학습/검증 분할 (게임 ID 기준으로 분할하여 데이터 누수 방지)
    # 같은 게임의 에피소드가 학습/검증에 동시에 들어가지 않도록
    game_ids = train_df['game_id'].unique()
    np.random.seed(config.seed)
    np.random.shuffle(game_ids)
    
    split_idx = int(len(game_ids) * (1 - val_split))
    train_game_ids = game_ids[:split_idx]
    val_game_ids = game_ids[split_idx:]
    
    # 게임 ID 기준으로 에피소드 분할
    train_episodes = train_df[train_df['game_id'].isin(train_game_ids)]['game_episode'].unique()
    val_episodes = train_df[train_df['game_id'].isin(val_game_ids)]['game_episode'].unique()
    
    train_data = train_df[train_df['game_episode'].isin(train_episodes)]
    val_data = train_df[train_df['game_episode'].isin(val_episodes)]
    
    print(f"학습 에피소드 수: {len(train_episodes)}")
    print(f"검증 에피소드 수: {len(val_episodes)}")
    
    train_dataset = PassSequenceDataset(train_data, config, is_train=True)
    val_dataset = PassSequenceDataset(val_data, config, is_train=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

