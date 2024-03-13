import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions

from models.DataEmbedding import DataEmbedding


class BertDeepLncLocEmbedding(nn.Module):
    tags = ["BertDeepLncLocEmbedding"]

    def __init__(self, bert_path=None) -> None:
        super().__init__()
        self.model_type = "BertDeepLncLocEmbedding"
        if bert_path:
            self.bert = BertModel.from_pretrained(bert_path)
        else:
            self.bert = BertModel(
                BertConfig(
                    vocab_size=97630,
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=4,
                    intermediate_size=1024,
                )
            )
        hidden_size = self.bert.config.hidden_size  # 256
        self.hidden_size = hidden_size

        self.seq_embedding = DataEmbedding(
            hidden_size,
            length=210,
            norm_type="bn",
            pos_embedding="sincos",
            upsample=False,
            dropout=0.1,
            type_embedding=False,
            cls_token=True
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
        self.seq_type_embedding = nn.Parameter(torch.rand(1, 1, hidden_size))

    def forward(self, seq):
        embedding_data_ls = []
        for batch in seq:
            bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(batch, output_hidden_states=True)
            embedding_data_ls.append((bert_output.hidden_states[1] + bert_output.hidden_states[-1]).mean(dim=1))
        seq = torch.stack(embedding_data_ls, dim=0)
        seq = self.seq_embedding(seq)
        seq = self.seq_transformer_encoder(seq)
        seq += self.seq_type_embedding.expand(seq.shape[0], -1, -1)
        return seq
