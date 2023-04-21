import pandas as pd

import torch
from torch.utils.data import DataLoader

import torchvision as tv

import pytorch_lightning as pl

from .base import ImageFolderDataset


# class FFHQAgingDataset(ImageFolderDataset):
#     def __init__(self,
#                  data_dir,  # Path to directory or zip.
#                  labels_file,  # Path to labels file.
#                  resolution=None,  # Ensure specific resolution, None = highest available.
#                  **super_kwargs  # Additional arguments for the Dataset base class.
#                  ):
#
#         self._labels_file = labels_file
#         super(FFHQAgingDataset, self).__init__(data_dir,  # Path to directory or zip.
#                                                name='FFHQ',  # Name of the dataset.
#                                                resolution=resolution,  # Ensure specific resolution, None = highest available.
#                                                use_labels=True,  # Whether to use labels.
#                                                **super_kwargs,     # Additional arguments for the Dataset base class.
#                                               )
#     def _load_raw_labels(self):
#         df = pd.read_csv(self._labels_file)
#         labels = df.to_numpy()
#         labels = torch.from_numpy(labels)
#
#         return labels
#
#
# class FFHQAgingDataModule(pl.LightningDataModule):
#     def __init__(self,
#                  data_dir,
#                  attr_names=None,
#                  batch_size=64,
#                  **dataloader_kwargs):
#         super().__init__()
#         # self.data_dir = data_dir
#         # self.batch_size = batch_size
#         # self.attr_names = attr_names
#         # self.num_classes = len(attr_names) if attr_names is not None else 40
#         # self.transform = tv.transforms.Compose([tv.transforms.Resize((224, 224),
#         #                                                              interpolation=InterpolationMode.BICUBIC,
#         #                                                              antialias=True),
#         #                                         tv.transforms.ToTensor(),
#         #                                         tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#         self.dataloader_kwargs = dataloader_kwargs
#
#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             self.ffhq_train = self.get_ffhq_dataset('train')
#             self.ffhq_val = self.get_ffhq_dataset('val')
#
#
#     def get_ffhq_dataset(self, split):
#         # TODO: implement FFHQ Aging dataset loader
#         dataset = FFHQAgingDataset()
#         return dataset
#
#     def train_dataloader(self):
#         return DataLoader(self.ffhq_train, batch_size=self.batch_size, **self.dataloader_kwargs)
#
#     def val_dataloader(self):
#         return DataLoader(self.ffhq_val, batch_size=self.batch_size, **self.dataloader_kwargs)


