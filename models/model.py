# -*- coding: utf-8 -*-
"""
패스 좌표 예측 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PassPredictor(nn.Module):
    """LSTM 기반 패스 좌표 예측 모델"""
    
    def __init__(self, config):
        super(PassPredictor, self).__init__()
        self.config = config
        
        # 입력 피처 차원 (utils/data_loader.py의 _extract_features에서 정의)
        self.input_dim = 20
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # LSTM 출력 차원 계산
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        
        # 어텐션 메커니즘 (선택적)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 출력 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)  # end_x, end_y
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) - 패딩 마스크
        Returns:
            output: (batch_size, output_dim) - 예측된 end_x, end_y
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM 통과
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        # 마스크 적용 (패딩 부분 제외)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            lstm_out = lstm_out * mask
        
        # 어텐션 적용
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 마지막 타임스텝 사용 (또는 평균 풀링)
        if mask is not None:
            # 마스크된 부분을 제외하고 평균
            masked_attn = attn_out * mask
            seq_lengths = mask.sum(dim=1)  # (batch_size, 1)
            pooled = masked_attn.sum(dim=1) / (seq_lengths + 1e-8)
        else:
            # 마지막 타임스텝 사용
            pooled = attn_out[:, -1, :]
        
        # 최종 출력
        output = self.fc_layers(pooled)
        
        return output


class TransformerPassPredictor(nn.Module):
    """Transformer 기반 패스 좌표 예측 모델 (대안)"""
    
    def __init__(self, config):
        super(TransformerPassPredictor, self).__init__()
        self.config = config
        self.input_dim = 20
        
        # 입력 임베딩
        self.input_projection = nn.Linear(self.input_dim, config.hidden_dim)
        
        # 위치 인코딩
        self.pos_encoder = nn.Parameter(
            torch.randn(1, config.max_sequence_length, config.hidden_dim)
        )
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # 출력 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) - 패딩 마스크
        """
        batch_size, seq_len, _ = x.size()
        
        # 입력 프로젝션
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # 위치 인코딩 추가
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # 패딩 마스크를 attention mask로 변환
        if mask is not None:
            # Transformer는 True가 마스크된 부분
            attn_mask = (1 - mask).bool()
        else:
            attn_mask = None
        
        # Transformer 인코더 통과
        encoded = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        
        # 마지막 타임스텝 또는 평균 풀링
        if mask is not None:
            masked_encoded = encoded * mask.unsqueeze(-1)
            seq_lengths = mask.sum(dim=1, keepdim=True)
            pooled = masked_encoded.sum(dim=1) / (seq_lengths + 1e-8)
        else:
            pooled = encoded[:, -1, :]
        
        # 최종 출력
        output = self.fc_layers(pooled)
        
        return output

