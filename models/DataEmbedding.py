import torch.nn as nn
import torch

from models.utils import AbsolutePositionalEncoding, PositionalEncoding


class DataEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size,
        length,
        upsample=True,
        type_embedding=True,
        cls_token=True,
        pos_embedding="sincos",
        norm_type="ln",
        dropout=0.1,
    ) -> None:
        super().__init__()
        if norm_type == "ln":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm_type == "bn":
            self.norm = nn.BatchNorm1d(hidden_size)
        else:
            self.norm = nn.Identity()
        
        length = length + 1 if cls_token else length
        self.upsample = nn.Conv1d(1, hidden_size, kernel_size=1) if upsample else None
        self.token_type_embedding = nn.Parameter(torch.rand(1, 1, hidden_size))if type_embedding else None
        if pos_embedding == "abs":
            self.pos_embedding = AbsolutePositionalEncoding(hidden_size, length)
        elif pos_embedding == "sincos":
            self.pos_embedding = PositionalEncoding(hidden_size, max_len=length)
        elif pos_embedding is None or pos_embedding is False:
            self.pos_embedding = nn.Identity()
        elif pos_embedding is True:
            raise ValueError("pos_embedding must be one of 'abs', 'sincos', None, False")
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if cls_token else None
        self.dropout = nn.Dropout(dropout)
        # torch.nn.init.kaiming_normal_(self.token_type_embedding)
        # torch.nn.init.kaiming_normal_(self.cls_token)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x.unsqueeze(1)).permute(0, 2, 1)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)
        if self.token_type_embedding is not None:
            x += self.token_type_embedding.expand(x.shape[0], -1, -1)
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        elif isinstance(self.norm, nn.LayerNorm):
            x = self.norm(x)

        return self.dropout(x)
