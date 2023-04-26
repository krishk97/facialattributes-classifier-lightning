import os
from datetime import datetime
import pathlib
import argparse
import yaml

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import dnnlib
from utils.callbacks import MetricsCallback, CUDACallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on a dataset"
    )
    parser.add_argument('--config',
                        required=True,
                        help="Path to config file",
                        type=pathlib.Path)
    parser.add_argument('--save',
                        default="output",
                        type=pathlib.Path,
                        help="Path to save directory",)
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help="gpu ids used")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    args.save = os.path.join(args.save, f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not os.path.exists(args.save):
        print("Creating directory to save files to at: ", args.save)
        os.makedirs(args.save)
        os.makedirs(os.path.join(args.save, "lightning_logs"))
        os.makedirs(os.path.join(args.save, "checkpoint"))

    if ('num_classes' in config['model'] and config['model']['num_classes'] is not None) and ('attr_names' in config['datamodule'] and config['datamodule']['attr_names'] is not None):
        assert config['model']['num_classes'] == len(config['datamodule']['attr_names']), 'Number of classes in model and datamodule must match'

    model = dnnlib.construct_class_by_name(**config['model'])
    datamodule = dnnlib.construct_class_by_name(**config['datamodule'])
    logger = TensorBoardLogger(save_dir=args.save)

    # # debugging limit train images
    # print("REMEMBER TO REMOVE LIMIT TRAIN IMAGES!")
    # datamodule = dnnlib.construct_class_by_name(max_train_imgs=1000, **config['datamodule'])

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(args.save, "checkpoint"), **config['checkpoint']),
        MetricsCallback(num_classes=config['model']['num_classes'], multilabel=config['model']['multilabel']),
        CUDACallback()
    ]

    trainer = Trainer(accelerator='gpu',
                      callbacks=callbacks,
                      logger=logger,
                      **config['trainer'])

    trainer.fit(model, datamodule=datamodule)
