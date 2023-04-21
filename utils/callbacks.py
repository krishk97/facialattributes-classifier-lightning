import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCallback(Callback):
    def __init__(self):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, y = batch
        y = y.cpu()
        outputs['logits'] = outputs['logits'].cpu()
        predictions = torch.round(nn.Sigmoid()(outputs['logits']))

        self.acc_mod_train(predictions, y)
        self.F1score_mod_train(predictions, y)

    def on_train_epoch_end(self, trainer, pl_module):

        acc = self.acc_mod_train.compute()
        self.acc_mod_train.reset()
        pl_module.log("train/acc", acc)

        F1 = self.F1score_mod_train.compute()
        self.F1score_mod_train.reset()
        pl_module.log("train/F1", F1)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, y = batch
        y = y.cpu()
        outputs['logits'] = outputs['logits'].cpu()
        predictions = torch.round(nn.Sigmoid()(outputs['logits']))

        self.acc_mod_val(predictions, y)
        self.F1score_mod_val(predictions, y)

    def on_validation_epoch_end(self, trainer, pl_module):
        attr_dict = pl_module.attr_dict
        acc = self.acc_mod_val.compute()
        self.acc_mod_val.reset()
        # for i, accuracy in enumerate(acc):
        #    wandb.log({"val/accuracy-{}".format(attr_dict[i]): accuracy})
        # table = wandb.Table(data=acc, columns=attr_dict)
        pl_module.log("val/acc", acc)

        F1 = self.F1score_mod_val.compute()
        self.F1score_mod_val.reset()
        # for i, f_score in enumerate(F1):
        #    wandb.log({"val/F1-scores-{}".format(attr_dict[i]): f_score})
        # table = wandb.Table(data=F1, columns=attr_dict)
        pl_module.log("val/F1", F1)


