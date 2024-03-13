import glob
import os
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from tqdm import tqdm
from sklearn.metrics import r2_score
import h5py

import os
from models import tracker_cls, _USE_HALFLIFE, _USE_SEQ, _USE_TF
from models.utils import count_parameters


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed=3)


class BaseDataset(Dataset):
    def __init__(self, filename, truncation: tuple = None, promoter_embedding=None, wv=None) -> None:
        super().__init__()
        self.promoter_embedding = promoter_embedding
        file = h5py.File(filename, "r", swmr=True)
        self.X_promoter = file["promoter"]
        self.X_halflife, self.y = (
            torch.tensor(np.array(file["halflife"]), dtype=torch.float32),
            torch.tensor(np.array(file["label"]), dtype=torch.float32),
        )
        if int == file["tf"].dtype:
            self.X_tf = torch.tensor(np.array(file["tf"]), dtype=torch.long)
        else:
            self.X_tf = torch.tensor(np.array(file["tf"]), dtype=torch.float32)
        if truncation:
            self.X_promoter = self.X_promoter[:, truncation[0] : truncation[1]]
        if self.X_promoter.dtype == np.bool8:
            self.X_promoter = self.X_promoter.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        promoter = self.X_promoter[idx]
        if self.promoter_embedding is not None:
            promoter = self.promoter_embedding(promoter)
        return promoter, self.X_halflife[idx], self.X_tf[idx], self.y[idx]


class ModelBase(nn.Module):
    tags = []
    use_seq = os.environ.get("_USE_SEQ", "0") == "1"
    use_halflife = os.environ.get("_USE_HALFLIFE", "0") == "1"
    use_tf = os.environ.get("_USE_TF", "0") == "1"

    def __init__(self, stage) -> None:
        super().__init__()
        self.stage = stage

    def cat(self, seq, halflife, tf):
        def _cat(x, y):
            if x is None:
                return y
            return torch.cat([x, y], dim=1)

        x = None
        if self.use_seq:
            x = _cat(x, seq)
        if self.use_halflife:
            x = _cat(x, halflife)
        if self.use_tf:
            x = _cat(x, tf)
        return x

    def forward(self, input_seq, input_halflife, input_tf):
        if self.stage == "mid":
            return self.forward_mid(input_seq, input_halflife, input_tf)
        elif self.stage == "late":
            return self.forward_late(input_seq, input_halflife, input_tf)
        else:
            raise NotImplementedError

    def forward_late(self, input_seq, input_halflife, input_tf):
        raise NotImplementedError

    def forward_mid(self, input_seq, input_halflife, input_tf):
        raise NotImplementedError

    def gen_first_layer(self, seq_size=None, output_size=None, half_size=8, tf_size=181):
        r = 0
        if self.use_seq:
            r += seq_size
        if self.use_halflife:
            r += half_size
        if self.use_tf:
            r += tf_size
        return nn.Linear(r, output_size)


class DumpWriger:
    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class BaseTrainer:
    def __init__(
        self,
        epochs=250,
        run_nums=5,
        writer=None,
        patience=20,
        lr_scheduler=None,
        model_kwargs=None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.patience = patience
        self._writer = writer
        self.lr_scheduler = lr_scheduler
        self.run_nums = run_nums
        self.best_model_state = None
        self.name = None
        self.model_kwargs = model_kwargs or {}

    def get_data_loader(self):
        raise NotImplementedError

    def generate_model_optimizer_criterion(self):
        raise NotImplementedError

    @property
    def writer(self):
        if os.environ.get("_USE_TRACK") is not None:
            _USE_TRACK = os.environ["_USE_TRACK"] == "1"
        else:
            _USE_TRACK = False
        if not _USE_TRACK or os.environ.get("_USE_TRACK", "0") != "1":
            return DumpWriger()
        if not self._writer:
            from datetime import datetime

            tracker = tracker_cls(
                project_name=f"SEQ{_USE_SEQ}-HALF{_USE_HALFLIFE}-TF{_USE_TF}",
                config={
                    "learning_rate": self.lr,
                    "architecture": self.name,
                    "optimizer": self.optimizer.__class__.__name__,
                    "epochs": self.epochs,
                    "use_halflife": ModelBase.use_halflife,
                    "use_tf": ModelBase.use_tf,
                    "use_seq": ModelBase.use_seq,
                },
                tags=self.model.tags,
            )
            self._writer = tracker
        return self._writer

    @property
    def lr(self):
        return self.optimizer.defaults["lr"]

    def start(self):
        runs_res = []
        for run in range(self.run_nums):
            print(f"Start run {run}")
            model, optimizer, criterion = self.generate_model_optimizer_criterion()
            if run == 0:
                print(f"Model {model.__class__} ")
                if hasattr(model, "seq_deeplncloc_embedding"):
                    print(f"seq_deeplncloc_embedding has {count_parameters(model.seq_deeplncloc_embedding)} parameters")
                print(f"total {count_parameters(model)} parameters")
            trainloader, validloader, testloader = self.get_data_loader()
            test_r2, test_loss = self.start_one_run(
                run, model, optimizer, criterion, trainloader, validloader, testloader
            )
            runs_res.append((test_r2, test_loss))
            print(runs_res)

        print(f"Average r2: {np.mean([x[0] for x in runs_res])} loss: {np.mean([x[1] for x in runs_res])}")

    def start_one_run(self, run, model, optimizer, criterion, trainloader, validloader, testloader):
        last_boost = 0
        best_model_r2 = -100
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, model, optimizer, criterion, trainloader)
            eval_loss, eval_r2 = self.evaluate(epoch, model, criterion, validloader, log_prefix="val")
            if eval_r2 > best_model_r2:
                print("Saving model")
                model_name = model.__class__.__name__
                project_name = f"SEQ{_USE_SEQ}-HALF{_USE_HALFLIFE}-TF{_USE_TF}"
                model_path = os.path.join("trained_models", model_name, project_name)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                while len(glob.glob(f"{model_path}/*.pt")) > 2:
                    os.remove(sorted(glob.glob(f"{model_path}/*.pt"))[0])
                torch.save(
                    model.state_dict(),
                    os.path.join(model_path, f"{model_name}-{eval_r2:.3f}-{eval_loss:.3f}.pt"),
                )
                self.best_model_state = copy.deepcopy(model.state_dict())
                best_model_r2 = eval_r2
                last_boost = 0
            else:
                last_boost += 1
                if last_boost > self.patience:
                    print("Early stopping. Best model r2: {:.3f}".format(best_model_r2))
                    break

            if self.lr_scheduler:
                self.lr_scheduler.step()
        model.load_state_dict(self.best_model_state)
        test_loss, test_r2 = self.evaluate(epoch, model, criterion, testloader, log_prefix="test")
        print("Best model test r2: {:.3f} loss:{:.3f}".format(test_r2, test_loss))
        self.writer.finish()
        return test_r2, test_loss

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
            loss = criterion(y_pred, y)
            train_loss_ls.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        print(f"\033[34mTrain Epoch: {epoch} \tLoss: {np.mean(train_loss_ls):.6f} \033[0m")
        self.writer.log({"train/loss": np.mean(train_loss_ls)})
        return np.mean(train_loss_ls)

    @torch.no_grad()
    def evaluate(self, epoch, model, criterion, dataloader, log_prefix="val"):
        model.eval()
        test_loss_ls = []
        y_pred_ls = []
        y_true_ls = []
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
            y_true_ls.append(y)
            loss = criterion(y_pred, y)
            test_loss_ls.append(loss.item())

        loss = np.mean(test_loss_ls)
        y_pred_ls, y_true_ls = torch.cat(y_pred_ls), torch.cat(y_true_ls)
        r2 = r2_score(y_true_ls, y_pred_ls)
        mae = F.l1_loss(y_pred_ls, y_true_ls)
        mse = F.mse_loss(y_pred_ls, y_true_ls)
        p = np.corrcoef(y_pred_ls, y_true_ls)[0][1]

        print(
            f"\033[31m{log_prefix} set: Average loss: {loss:.4f} R2 score:{r2:.4f} MAE:{mae:.4f} MSE:{mse:.4f} P:{p:.4f} \033[0m"
        )
        self.writer.log(
            {
                f"{log_prefix}/loss": loss,
                f"{log_prefix}/r2_score": r2,
                f"{log_prefix}/mae": mae,
                f"{log_prefix}/mse": mse,
            }
        )
        return loss, r2
