import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing

from tqdm import tqdm

from transformers import BertTokenizer

from sklearn.metrics import precision_recall_fscore_support

from models.base import *
from models.base import _USE_HALFLIFE, _USE_SEQ, _USE_TF
from models.utils import (
    get_data_loader,
)
from models.DataEmbedding import DataEmbedding
from models.BertDeepLncLocEmbedding import BertDeepLncLocEmbedding

from tools import cpp_utils


BERT_PATH = "trained_models/bert/promoter_k3_to_k7/checkpoint-100000"


class SeqHalflifeToTFDataset(Dataset):
    def __init__(self, filename, truncation: tuple = None, promoter_embedding=True, **kwargs) -> None:
        super().__init__()
        # bert_path = "trained_models/bert/promoter_k5/checkpoint-70000"
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
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
        input_ids = torch.tensor(
            np.array(
                self.tokenizer(
                    fragmented_sent_ls,
                    add_special_tokens=False,
                )["input_ids"],
                dtype=np.int32,
            )
        )
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


class TransformerSeqHalflifeToTF(ModelBase):
    tags = ["TransformerSeqHalflifeToTF"]

    def __init__(self, n_heads=4, dropout_rate=0.1) -> None:
        super().__init__()
        self.model_type = "TransformerSeqHalflifeToTF"
        self.seq_deeplncloc_embedding = BertDeepLncLocEmbedding(BERT_PATH)
        hidden_size = 256
        self.hidden_size = hidden_size
        self.half_embedding = DataEmbedding(
            hidden_size, length=8, norm_type="bn", pos_embedding="abs", dropout=dropout_rate, cls_token=False
        )
        self.mixture_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.final_fc = nn.Sequential(
            nn.Linear(256 + 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 181),
            nn.Sigmoid(),
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
        seq = self.seq_deeplncloc_embedding(seq)

        halflife = self.half_embedding(halflife)
        all_data = torch.cat([seq, halflife], dim=1)
        all_data = self.mixture_encoder(all_data)  # [bs, seq_len/210, channels]

        feature = torch.mean(all_data, dim=1)
        feature = torch.cat([feature, input_halflife], dim=1)
        if predict:
            return self.final_fc(feature)
        else:
            return feature


class TransformerSeqHalflifeToTFTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__(
            patience=5,
        )

    @staticmethod
    def get_data_loader():
        center = 10000
        upstream = center - 7000
        downstream = center + 3500
        datadir = "Dataset/dataset_aumentati"
        filename_format = "{}_tf.h5"
        trainloader, validloader, testloader = get_data_loader(
            filenames=[os.path.join(datadir, filename_format.format(i)) for i in ("train", "validation", "test")],
            batch_size=4 if os.environ.get("_USE_SEQ") == "1" else 128,
            dateset_cls=SeqHalflifeToTFDataset,
            truncation=(upstream, downstream),
            num_workers=12,
            promoter_embedding=True if os.environ.get("_USE_SEQ") == "1" else False,
        )
        return trainloader, validloader, testloader

    def generate_model_optimizer_criterion(self):
        model = TransformerSeqHalflifeToTF().to(torch.device("cuda"))
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        criterion = nn.BCELoss()
        return model, optimizer, criterion

    def start_one_run(self, run, model, optimizer, criterion, trainloader, validloader, testloader):
        last_boost = 0
        best_model_f1 = -100
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, model, optimizer, criterion, trainloader)
            eval_loss, eval_f1 = self.evaluate(epoch, model, criterion, validloader, log_prefix="val")
            if eval_f1 > best_model_f1:
                print("Saving model")
                model_name = model.__class__.__name__
                project_name = f"SEQ{_USE_SEQ}-HALF{_USE_HALFLIFE}-TF{_USE_TF}"
                model_path = os.path.join("trained_models", model_name, project_name)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                while len(glob.glob(f"{model_path}/*.pt")) > 2:
                    os.remove(sorted(glob.glob(f"{model_path}/*.pt"))[0])
                torch.save(
                    model.state_dict(), os.path.join(model_path, f"{model_name}-{eval_f1:.3f}-{eval_loss:.3f}.pt")
                )
                self.best_model_state = copy.deepcopy(model.state_dict())
                best_model_f1 = eval_f1
                last_boost = 0
            else:
                last_boost += 1
                if last_boost > self.patience:
                    print("Early stopping. Best model f1: {:.3f}".format(best_model_f1))
                    break
        model.load_state_dict(self.best_model_state)
        test_loss, test_f1 = self.evaluate(epoch, model, criterion, testloader, log_prefix="test")
        print("Best model test f1: {:.3f} loss:{:.3f}".format(test_f1, test_loss))
        self.writer.finish()
        return test_f1, test_loss

    @torch.no_grad()
    def evaluate(self, epoch, model, criterion, dataloader, log_prefix="val"):
        model.eval()
        test_loss_ls = []
        y_pred_ls = []
        tf_true_ls = []
        for i, (X_promoter, X_halflife, X_tf, y) in enumerate(tqdm(dataloader, desc="Testing", ncols=100)):
            X_promoter, X_halflife, X_tf = (
                X_promoter.to(self.device),
                X_halflife.to(self.device),
                X_tf.to(self.device),
            )
            y_pred = model(
                X_promoter,
                X_halflife,
                X_tf,
            )
            y_pred = y_pred.squeeze().cpu()
            y_pred_ls.append(y_pred)
            tf_true_ls.append(X_tf.cpu())
            loss = criterion(y_pred, X_tf.cpu())
            test_loss_ls.append(loss.item())

        loss = np.mean(test_loss_ls)
        y_pred_ls, tf_true_ls = torch.cat(y_pred_ls), torch.cat(tf_true_ls)
        precision, recall, f1, _ = precision_recall_fscore_support(tf_true_ls, y_pred_ls > 0.5, average="micro")
        print(
            f"\033[31m{log_prefix} set: loss:{loss:.4f} precision: {precision:.4f} recall:{recall:.4f} f1:{f1:.4f} \033[0m"
        )

        return loss, f1

    def train(self, epoch, model, optimizer, criterion, trainloader):
        model.train()
        train_loss_ls = []
        for i, (X_promoter, X_halflife, X_tf, y) in enumerate(
            tqdm(trainloader, desc="Training {}".format(epoch), ncols=100)
        ):
            X_promoter, X_halflife, X_tf = (
                X_promoter.to(self.device),
                X_halflife.to(self.device),
                X_tf.to(self.device),
            )
            optimizer.zero_grad()
            y_pred = model(X_promoter, X_halflife, X_tf)
            y_pred = y_pred.squeeze().cpu()
            y = y.squeeze().cpu()
            loss = criterion(y_pred, X_tf.cpu())
            train_loss_ls.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        print(f"\033[34mTrain Epoch: {epoch} \tLoss: {np.mean(train_loss_ls):.6f} \033[0m")
        self.writer.log({"train/loss": np.mean(train_loss_ls)})
        return np.mean(train_loss_ls)
