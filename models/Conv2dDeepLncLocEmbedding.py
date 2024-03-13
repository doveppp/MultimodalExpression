import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DataEmbedding import DataEmbedding


class Conv2dDeepLncLocEmbedding(nn.Module):
    def __init__(self, hidden_size=256) -> None:
        super().__init__()
        self.seq_type_embedding = nn.Parameter(torch.rand(1, 1, hidden_size))
        self.seq_embedding = DataEmbedding(
            hidden_size,
            length=500,
            norm_type="bn",
            pos_embedding="sincos",
            upsample=False,
            dropout=0.0,
            type_embedding=False,
            cls_token=True,
        )
        self.seq_deeplncloc_conv2d = nn.Sequential(
            nn.Conv2d(5, 256, (1, 3), (1, 1)),  # (N, C_{in}, H_{in}, W_{in})
            nn.MaxPool2d((1, 3), (1, 3)),  # (N, C_{out}, H_{out}, W_{out})
            nn.Conv2d(256, hidden_size, (1, 3), (1, 1)),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.seq_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            ),
            num_layers=2,
        )

    def forward(self, seq):
        seq = self.seq_deeplncloc_conv2d(seq.permute(0, 3, 1, 2))
        seq = seq.squeeze(3).permute(0, 2, 1)
        seq = self.seq_embedding(seq)
        seq = self.seq_transformer_encoder(seq)
        seq += self.seq_type_embedding.expand(seq.shape[0], -1, -1)
        return seq
