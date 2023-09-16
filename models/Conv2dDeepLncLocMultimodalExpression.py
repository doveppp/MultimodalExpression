import os
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np

from models.base import Dataset, ModelBase, BaseTrainer
from models.utils import (
    get_data_loader,
    sliding_window,
)
from models.Conv2dDeepLncLocEmbedding import Conv2dDeepLncLocEmbedding
from models.DataEmbedding import DataEmbedding


class Conv2dDeepLncLocMultimodalExpressionDataset(Dataset):
    def __init__(self, filename, truncation: tuple = None, promoter_embedding=True, **kwargs) -> None:
        super().__init__()
        file = h5py.File(filename, "r", swmr=True)
        self.X_promoter = file["promoter"]
        self.X_halflife, self.tf, self.y = (
            torch.tensor(np.array(file["halflife"]), dtype=torch.float32),
            torch.tensor(np.array(file["tf"]), dtype=torch.float32),
            torch.tensor(np.array(file["label"]), dtype=torch.float32),
        )
        if truncation:
            self.X_promoter = self.X_promoter[:, truncation[0] : truncation[1]]
        if self.X_promoter.dtype == np.bool8:
            self.X_promoter = self.X_promoter.astype(np.float32)
        self.embedded_promoter = multiprocessing.Manager().dict() if promoter_embedding else None

    def __len__(self):
        return len(self.y)

    def embedding(self, onehot_sent):
        onehot_sent = np.insert(onehot_sent, 4, 1, axis=1)  # 0:A, 1:C, 2:G, 3:T, 4:none
        small_sent_ls = sliding_window(onehot_sent, 100, 50)
        return torch.tensor(np.array(small_sent_ls))

    def __getitem__(self, idx):
        promoter = self.X_promoter[idx]
        if self.embedded_promoter is not None:
            if idx in self.embedded_promoter:
                promoter = self.embedded_promoter[idx]
            else:
                promoter = self.embedding(promoter)
                self.embedded_promoter[idx] = promoter
        return promoter, self.X_halflife[idx], self.tf[idx], self.y[idx]


class Conv2dDeepLncLocMultimodalExpression(ModelBase):
    def __init__(self, feature_extract_type="mean", n_heads=4, dropout_rate=0.1, mix_layer=True) -> None:
        super().__init__()
        hidden_size = 256
        self.max_len = 210 + 181 + 8 + 3
        self.feature_extract_type = feature_extract_type
        self.seq_deeplncloc_embedding = Conv2dDeepLncLocEmbedding(hidden_size)
        self.half_embedding = DataEmbedding(
            hidden_size, length=8, norm_type="bn", pos_embedding="abs", dropout=dropout_rate, cls_token=False
        )
        self.tf_embedding = DataEmbedding(
            hidden_size, length=181, norm_type="bn", pos_embedding="abs", dropout=dropout_rate, cls_token=False
        )
        self.mixture_encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=n_heads,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True,
                ),
                num_layers=2,
            )
            if mix_layer
            else nn.Identity()
        )
        self.final_fc = nn.Sequential(
            self.gen_first_layer(seq_size=hidden_size, output_size=100),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 1),
        )

    def forward(self, input_seq, input_halflife, input_tf, predict=True):
        """
        :param input_seq: [bs, seq_len, channels]
        :param input_halflife: [bs, 8]
        :param input_tf: [bs, 181]
        """
        seq = input_seq
        halflife = input_halflife
        tf = input_tf
        all_data = None
        halflife_tf_data = None
        if self.use_seq:
            all_data = self.seq_deeplncloc_embedding(seq)
        if self.use_halflife:
            halflife = self.half_embedding(halflife)
            halflife_tf_data = halflife
        if self.use_tf:
            tf = self.tf_embedding(tf)
            halflife_tf_data = tf if halflife_tf_data is None else torch.cat([halflife_tf_data, tf], dim=1)
        if halflife_tf_data is not None:
            all_data = halflife_tf_data if all_data is None else torch.cat([all_data, halflife_tf_data], dim=1)
            all_data = self.mixture_encoder(all_data)  # [bs, seq_len/210, channels]

        if self.feature_extract_type == "mean":
            feature = torch.mean(all_data, dim=1)
        elif self.feature_extract_type == "cls":
            feature = all_data[:, 0]
        else:
            raise ValueError("feature_extract_type must be one of 'mean', 'cls'")
        if self.use_halflife:
            feature = torch.cat([feature, input_halflife], dim=1)
        if self.use_tf:
            feature = torch.cat([feature, input_tf], dim=1)
        if predict:
            return self.final_fc(feature)
        else:
            return feature


class Conv2dDeepLncLocMultimodalExpressionTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__(
            patience=20,
        )

    def get_data_loader(self):
        center = 10000  #
        upstream = center - 7000  #
        downstream = center + 3500  #
        datadir = "Dataset/dataset_aumentati"
        filename_format = "{}_tf.h5"
        trainloader, validloader, testloader = get_data_loader(
            filenames=[os.path.join(datadir, filename_format.format(i)) for i in ("train", "validation", "test")],
            batch_size=64 if os.environ.get("_USE_SEQ") == "1" else 128,
            dateset_cls=Conv2dDeepLncLocMultimodalExpressionDataset,
            truncation=(upstream, downstream),
            num_workers=16,
            promoter_embedding=True if os.environ.get("_USE_SEQ") == "1" else False,
        )
        return trainloader, validloader, testloader

    def generate_model_optimizer_criterion(self):
        model = Conv2dDeepLncLocMultimodalExpression().to(torch.device("cuda"))
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.MSELoss()
        return model, optimizer, criterion
