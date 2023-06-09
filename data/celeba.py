from typing import Optional
import numpy as np

from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.datasets import CelebA
from torchvision.transforms import InterpolationMode

import pytorch_lightning as pl


class CelebADataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 attr_names: Optional[list] = None,
                 batch_size: int = 64,
                 max_train_imgs: Optional[int] = None, # max number of training images to use
                 **dataloader_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_train_imgs = max_train_imgs
        self.attr_names = attr_names
        self.num_classes = len(attr_names) if attr_names is not None else 40
        self.transform = tv.transforms.Compose([tv.transforms.Resize((224, 224),
                                                                     interpolation=InterpolationMode.BICUBIC,
                                                                     antialias=True),
                                                tv.transforms.ToTensor(),
                                                tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.celeba_train = self.get_celeba_dataset('train', max_imgs=self.max_train_imgs)
            self.celeba_val = self.get_celeba_dataset('valid')

        if stage == 'test' or stage is None:
            self.celeba_test = self.get_celeba_dataset('test')

    def get_celeba_dataset(self, split, max_imgs=None):
        dataset = CelebA(root=self.data_dir, split=split, target_type='attr', transform=self.transform)

        if max_imgs is not None and max_imgs < len(dataset):
            subset_indxs = np.random.choice(len(dataset), size=max_imgs, replace=False)
            dataset.filename = np.array(dataset.filename)[subset_indxs]
            dataset.attr = np.array(dataset.attr)[subset_indxs]
            dataset.identity = np.array(dataset.identity)[subset_indxs]
            dataset.bbox = np.array(dataset.bbox)[subset_indxs]
            dataset.landmarks_align = np.array(dataset.landmarks_align)[subset_indxs]

        if self.attr_names is not None:  # limit attrs to those specified
            attr_indxs = [dataset.attr_names.index(a) for a in self.attr_names]
            dataset.attr = dataset.attr[:, attr_indxs]
            dataset.attr_names = self.attr_names
        else:
            dataset.attr_names = dataset.attr_names[:-1]  # remove empty string attr

        return dataset

    def train_dataloader(self):
        return DataLoader(self.celeba_train, batch_size=self.batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.celeba_val, batch_size=self.batch_size, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.celeba_test, batch_size=self.batch_size, **self.dataloader_kwargs)
