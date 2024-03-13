import random
import math
import typing
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # 64*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 64*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 256   model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1*max_len*d_model
        self.register_buffer("pe", pe)

    def forward(self, x):  # [batch,seq,d_model]
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        nn.init.kaiming_normal_(self.pos_embed)

    def forward(self, x):
        x += self.pos_embed.expand(x.shape[0], -1, -1)
        return x


class GlobalAveragePooling1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        x = torch.mean(x, dim=self.dim)
        return x


def sliding_window(seq, window_size, stride):
    result = []
    if window_size > len(seq):
        return [seq]
    for i in range(0, len(seq) - window_size + 1, stride):
        subseq = seq[i : i + window_size]
        result.append(subseq)
    return result


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output = self.att(inputs, inputs, inputs)[0]
        # print(len(attn_output))
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


def get_data_loader(
    filenames: typing.Iterable,
    dateset_cls=None,
    truncation=None,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    promoter_embedding=None,
):
    train_filename, valid_filename, test_filename = filenames
    test_dateset = dateset_cls(
        test_filename,
        truncation=truncation,
        promoter_embedding=promoter_embedding,
    )
    valid_dataset = dateset_cls(
        valid_filename,
        truncation=truncation,
        promoter_embedding=promoter_embedding,
    )

    return (
        DataLoader(
            dateset_cls(train_filename, truncation=truncation, promoter_embedding=promoter_embedding),
            batch_size,
            shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=10,
        ),
        DataLoader(
            valid_dataset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        ),
        DataLoader(
            test_dateset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        ),
    )


def load_submodel(model, state_dict, state_dict_submodel_name, model_submodel_name):
    submodel_state_dict = {
        k.replace(state_dict_submodel_name + ".", ""): v
        for k, v in state_dict.items()
        if k.startswith(state_dict_submodel_name)
    }
    attr = getattr(model, model_submodel_name)
    if hasattr(attr, "load_state_dict"):
        attr.load_state_dict(submodel_state_dict)
    else:
        # print(type(attr), type(submodel_state_dict))
        # print(submodel_state_dict)
        attr.data = submodel_state_dict[state_dict_submodel_name]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
