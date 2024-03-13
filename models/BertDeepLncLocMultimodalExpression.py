import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing

from tqdm import tqdm

from transformers import BertConfig, BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions

from models.base import *
from models.base import _USE_HALFLIFE, _USE_SEQ, _USE_TF
from models.utils import (
    get_data_loader,
    PositionalEncoding,
    GlobalAveragePooling1D,
    AbsolutePositionalEncoding,
    load_submodel,
    sliding_window,
)
from tools import cpp_utils
from models.BertDeepLncLocEmbedding import BertDeepLncLocEmbedding
from models.DataEmbedding import DataEmbedding
from models.Conv2dDeepLncLocMultimodalExpression import Conv2dDeepLncLocMultimodalExpression

BERT_PATH = "trained_models/bert/promoter_k3_to_k7/checkpoint-100000"


class BertDeepLncLocMultimodalExpressionDataset(Dataset):
    def __init__(self, filename, truncation: tuple = None, promoter_embedding=True, **kwargs) -> None:
        super().__init__()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(BERT_PATH)
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
        max_indices = np.argmax(onehot_sent, axis=1)
        char_dict = {4: "N", 0: "A", 1: "C", 2: "G", 3: "T"}
        translated_sent = "".join([char_dict[i.item()] for i in max_indices])
        small_sent_ls = cpp_utils.sliding_window(translated_sent, 100, 50)
        fragmented_sent_ls = []
        for small_sent in small_sent_ls:
            sent = " ".join(cpp_utils.fragment(4, 4, small_sent))
            fragmented_sent_ls.append(sent)
        input_ids = self.tokenizer(
            fragmented_sent_ls,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        return input_ids

    def __getitem__(self, idx):
        promoter = self.X_promoter[idx]
        if self.embedded_promoter is not None:
            if idx in self.embedded_promoter:
                promoter = self.embedded_promoter[idx]
            else:
                promoter = self.embedding(promoter)
                self.embedded_promoter[idx] = promoter
        return promoter, self.X_halflife[idx], self.tf[idx], self.y[idx]


class BertDeepLncLocMultimodalExpression(Conv2dDeepLncLocMultimodalExpression):
    def __init__(
        self,
        train_seq=True,
        train_hl=True,
        feature_extract_type="mean",
        n_heads=4,
        dropout_rate=0.1,
        mix_layer=True,
        stage="mid",
    ) -> None:
        super().__init__(feature_extract_type, n_heads, dropout_rate, mix_layer, stage)
        self.seq_deeplncloc_embedding = BertDeepLncLocEmbedding(BERT_PATH)
        self.seq_deeplncloc_embedding.requires_grad_(train_seq)
        self.half_embedding.requires_grad_(train_hl)


class BertDeepLncLocMultimodalExpressionTrainer(BaseTrainer):
    def __init__(
        selfmodel_kwargs=None,
        model_kwargs=None,
    ) -> None:
        super().__init__(
            patience=5,
            model_kwargs=model_kwargs,
        )

    def get_data_loader(self):
        center = 10000
        upstream = center - 7000
        downstream = center + 3500
        datadir = "Dataset/dataset_aumentati"
        filename_format = "{}_tf.h5"
        trainloader, validloader, testloader = get_data_loader(
            filenames=[os.path.join(datadir, filename_format.format(i)) for i in ("train", "validation", "test")],
            batch_size=4,
            dateset_cls=BertDeepLncLocMultimodalExpressionDataset,
            truncation=(upstream, downstream),
            num_workers=12,
            promoter_embedding=True if os.environ.get("_USE_SEQ") == "1" else False,
        )
        return trainloader, validloader, testloader

    def generate_model_optimizer_criterion(self):
        model = BertDeepLncLocMultimodalExpression(
            train_seq=True, train_hl=True, **self.model_kwargs, mix_layer=True
        ).to(torch.device("cuda"))
        # seq_pretrained_model = torch.load("./trained_models/saved/BertDeepLncLocMultimodalExpression-0.650-0.341.pt")
        # load_submodel(model, seq_pretrained_model, "seq_deeplncloc_embedding", "seq_deeplncloc_embedding")
        # load_submodel(model, seq_pretrained_model, "half_embedding", "half_embedding")
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        criterion = nn.MSELoss()
        return model, optimizer, criterion
