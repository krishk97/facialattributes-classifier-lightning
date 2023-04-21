import torch
from torch.utils.data import Dataset, DataLoader

import torchvision as tv
from torchvision.datasets import CelebA
from torchvision.transforms import InterpolationMode

import pytorch_lightning as pl


class CelebADataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 attr_names=None,
                 batch_size=64,
                 **dataloader_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
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
            self.celeba_train = self.get_celeba_dataset('train')
            self.celeba_val = self.get_celeba_dataset('val')

        if stage == 'test' or stage is None:
            self.celeba_test = self.get_celeba_dataset('test')

    def get_celeba_dataset(self, split):
        dataset = CelebA(root=self.data_dir, split=split, target_type='attr', transform=self.transform)
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
