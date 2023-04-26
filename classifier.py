from typing import Optional

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision as tv
import pytorch_lightning as pl

_activation_funcs = {
    'linear': lambda x, **_: x,
    'relu': lambda x, **_: F.relu(x),
    'lrelu': lambda x, alpha, **_: F.leaky_relu(x, alpha),
    'tanh': lambda x, **_: torch.tanh(x),
    'sigmoid': lambda x, **_: torch.sigmoid(x),
    'elu': lambda x, **_: F.elu(x),
    'selu': lambda x, **_: F.selu(x),
    'gelu': lambda x, **_: F.gelu(x),
    'softplus': lambda x, beta, **_: F.softplus(x, beta),
    'swish': lambda x, **_: torch.sigmoid(x) * x,
}


class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 activation='linear',
                 activation_params=None,
                 dropout=True,
                 dropout_p=0.5):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = _activation_funcs[activation]
        self.act_params = activation_params if activation_params is not None else {}
        self.dropout = nn.Dropout(p=dropout_p) if dropout else nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x, **self.act_params)
        x = self.dropout(x)
        return x

# ----------------------------------------------------------------------------

def get_resnet_classifier(backbone, pretrained=True, num_classes=1):
    if backbone in ['r18', 'resnet18']:
        weights = tv.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv.models.resnet18(weights=weights)
        model.fc = nn.Sequential(FullyConnectedLayer(512, 512, activation='relu', dropout=True),
                                 FullyConnectedLayer(512, num_classes, activation='linear', dropout=False))
    elif backbone in ['r34', 'resnet34']:
        weights = tv.models.ResNet34_Weights.DEFAULT if pretrained else None
        model = tv.models.resnet34(weights=weights)
        model.fc = nn.Sequential(FullyConnectedLayer(512, 512, activation='relu', dropout=True),
                                 FullyConnectedLayer(512, num_classes, activation='linear', dropout=False))
    elif backbone in ['r50', 'resnet50']:
        weights = tv.models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv.models.resnet50(weights=weights)
        model.fc = nn.Sequential(FullyConnectedLayer(2048, 2048, activation='relu', dropout=True),
                                 FullyConnectedLayer(2048, 512, activation='relu', dropout=True),
                                 FullyConnectedLayer(512, num_classes, activation='linear', dropout=False))
    elif backbone in ['r101', 'resnet101']:
        weights = tv.models.ResNet101_Weights.DEFAULT if pretrained else None
        model = tv.models.resnet101(weights=weights)
        model.fc = nn.Sequential(FullyConnectedLayer(2048, 2048, activation='relu', dropout=True),
                                 FullyConnectedLayer(2048, 512, activation='relu', dropout=True),
                                 FullyConnectedLayer(512, num_classes, activation='linear', dropout=False))
    elif backbone in ['r152', 'resnet152']:
        weights = tv.models.ResNet152_Weights.DEFAULT if pretrained else None
        model = tv.models.resnet152(weights=weights)
        model.fc = nn.Sequential(FullyConnectedLayer(2048, 2048, activation='relu', dropout=True),
                                 FullyConnectedLayer(2048, 512, activation='relu', dropout=True),
                                 FullyConnectedLayer(512, num_classes, activation='linear', dropout=False))
    else:
        raise RuntimeError(f'{backbone} is not a valid backbone.')

    return model

# ----------------------------------------------------------------------------

class ResnetClassifierModule(pl.LightningModule):
    def __init__(self,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 num_classes: int = 1,
                 multilabel: bool = False,  # multilabel classification?
                 optimizer_config: Optional[dict] = None,
                 scheduler_config: Optional[dict] = None,
                 ):
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        # Create model
        self.model = get_resnet_classifier(backbone, pretrained=pretrained, num_classes=num_classes)

        # Create loss module
        self.loss = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

        # Optimizer and scheduler configs
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

    def configure_optimizers(self):
        if self.optimizer_config is None:
            print("Optimizer config not specified. Using Adam with default torch configs")
            optimizer = optim.Adam(self.model.parameters())
        else:
            if self.optimizer_config['method'] == "AdamW":
                # AdamW is Adam with a correct implementation of weight decay (see here
                # for details: https://arxiv.org/pdf/1711.05101.pdf)
                optimizer = optim.AdamW(self.model.parameters(), **self.optimizer_config['params'])
            elif self.optimizer_config['method'] == "Adam":
                optimizer = optim.Adam(self.model.parameters(), **self.optimizer_config['params'])
            elif self.optimizer_config['method'] == "SGD":
                optimizer = optim.SGD(self.model.parameters(), **self.optimizer_config['params'])
            else:
                raise RuntimeError(
                    f"{self.optimizer_config['method']} is not a valid optimizer. Choose from ['AdamW', 'Adam', 'SGD']")

        if self.scheduler_config is None:
            return optimizer
        else:
            if self.scheduler_config['method'] == "StepLR":
                scheduler = optim.lr_scheduler.StepLR(optimizer, **self.scheduler_config['params'])
            elif self.scheduler_config['method'] == "MultiStepLR":
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_config['params'])
            elif self.scheduler_config['method'] == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_config['params'])
            else:
                raise RuntimeError(
                    f"{self.scheduler_config['method']} is not a valid scheduler. Choose from ['StepLR', 'MultiStepLR', 'ReduceLROnPlateau']")
            return [optimizer], [scheduler]

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, targets = batch
        logits = self.model(imgs)
        loss = self.loss(logits, targets.float())

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        acc = (logits.argmax(dim=-1) == targets).float().mean()
        self.log("train/acc", acc, on_step=False, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True)

        return {'loss': loss, 'logits': logits, 'targets': targets}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs)

        # By default logs it per epoch (weighted average over batches)
        acc = (logits.argmax(dim=-1) == targets).float().mean()
        self.log("val/acc", acc)

        return {'logits': logits, 'targets': targets}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs)

        # By default logs it per epoch (weighted average over batches)
        acc = (logits.argmax(dim=-1) == targets).float().mean()
        self.log("test/acc", acc)

        return {'logits': logits, 'targets': targets}
