import time

import torch

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info

from torchmetrics import F1Score, Precision, Recall


class MetricsCallback(Callback):
    def __init__(self, num_classes, multilabel=True):
        self.precision_mod_train = Precision(task='multilabel' if multilabel else 'multiclass', num_classes=num_classes)
        self.precision_mod_val = Precision(task='multilabel' if multilabel else 'multiclass', num_classes=num_classes)

        self.recall_mod_train = Recall(task='multilabel' if multilabel else 'multiclass', num_classes=num_classes)
        self.recall_mod_val = Recall(task='multilabel' if multilabel else 'multiclass', num_classes=num_classes)

        self.f1_mod_train = F1Score(task='multilabel' if multilabel else 'multiclass', num_classes=num_classes)
        self.f1_mod_val = F1Score(task='multilabel' if multilabel else 'multiclass', num_classes=num_classes)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        logits = outputs['logits'].cpu()
        targets = outputs['targets'].cpu()

        self.precision_mod_val(logits, targets)
        self.recall_mod_val(logits, targets)
        self.f1_mod_val(logits, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        precision = self.precision_mod_val.compute()
        self.precision_mod_val.reset()
        pl_module.log("val/precision", precision)

        recall = self.recall_mod_val.compute()
        self.recall_mod_val.reset()
        pl_module.log("val/recall", recall)

        f1 = self.f1_mod_val.compute()
        self.f1_mod_val.reset()
        pl_module.log("val/f1", f1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        logits = outputs['logits'].detach().cpu()
        targets = outputs['targets'].detach().cpu()

        self.precision_mod_train(logits, targets)
        self.recall_mod_train(logits, targets)
        self.f1_mod_train(logits, targets)

    def on_train_epoch_end(self, trainer, pl_module):
        precision = self.precision_mod_train.compute()
        self.precision_mod_train.reset()
        pl_module.log("train/precision", precision)

        recall = self.recall_mod_train.compute()
        self.recall_mod_train.reset()
        pl_module.log("train/recall", recall)

        f1 = self.f1_mod_train.compute()
        self.f1_mod_train.reset()
        pl_module.log("train/f1", f1)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
