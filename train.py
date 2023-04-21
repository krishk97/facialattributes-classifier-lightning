import os, sys
import argparse
import yaml

from pytorch_lightning import Trainer

from utils import dnnlib

if __name__ == "__main__":

    trainer = Trainer()

    yaml.load(args.config)

