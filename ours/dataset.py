from functools import cache
import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)

import os
import numpy as np
import torch
import datasets as hfdat

import karras.dnnlib as dnnlib
from karras.training.dataset import Dataset
# import pyvtorch.huggingface as vhfdat

DATASET_OPTIONS = {
    "imagenet": dict(class_name='karras.training.dataset.ImageFolderDataset'),
    "cifar10": dict(class_name='ours.dataset.HuggingFaceDataset', path="uoft-cs/cifar10", n_classes=10),
    "folder": dict(class_name='karras.training.dataset.ImageFolderDataset'),
}

class HuggingFaceDataset(Dataset):

    def __init__(self,
        path,                   # Path to Hugging Face dataset
        n_classes,              # Specify number of classes for one hot encoding
        resolution      = None, # Ensure specific resolution, None = anything goes.
        cache_dir       = dirs.DATA_HOME, # Cache dir to store the Hugging Face dataset
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path        
        self._cache_dir = cache_dir

        self._dataset = hfdat.load_dataset(path, split="train", cache_dir=self.cache_dir)
        
        self._n_classes = n_classes
        self._key_image = "img" # Change if different
        self._key_label = "label" # Change if different
        if self.key_image not in self.data.column_names:
            raise ValueError(f"Hugging Face Dataset has no column {self.key_image}")
        if self.key_label not in self.data.column_names:
            raise ValueError(f"Hugging Face Dataset has no column {self.key_label}")
        
        self.data.set_format(type="torch", columns=[self.key_image,self.key_label])

        name = "HuggingFaceDataset" #os.path.splitext(os.path.basename(self.path))[0]
        raw_shape = [len(self._dataset)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
    
    @property
    def path(self):
        return self._path
    
    @property
    def cache_dir(self):
        return self._cache_dir
    
    @property
    def data(self):
        return self._dataset
    
    @property
    def n_classes(self):
        return self._n_classes

    @property
    def label_shape(self):
        return [self.n_classes+1]
    
    @property
    def key_image(self):
        return self._key_image
    
    @property
    def key_label(self):
        return self._key_label

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        else:
            if isinstance(image, torch.Tensor):
                image = image.copy()
            else:
                image = image.detach().clone()
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return idx, image, self.get_label(idx)

    def _load_raw_image(self, raw_idx):
        raw_idx = np.array([raw_idx], dtype=int)
        # print("Image Raw Index", type(raw_idx), raw_idx)
        img = self.data[raw_idx][self.key_image].squeeze().float() # CHW
        # print("Image", type(img), img.dtype, img.shape, img)
        return img # Return float32 Torch tensor

    def _load_raw_label(self, raw_idx):
        raw_idx = np.array([raw_idx], dtype=int)
        return self.data[raw_idx][self.key_label]
    
    def get_label(self, idx):
        raw_idx = self._raw_idx[idx]
        label = self._load_raw_label(raw_idx) # Assume this is an int
        onehot = np.zeros(self.label_shape, dtype=np.float32)
        onehot[label] = 1
        return onehot.copy()

    @property
    def has_onehot_labels(self):
        return False # Assume dataset has just int labels ==> Get label converts it instead

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._load_raw_label(d.raw_idx).copy()
        return d

if __name__ == "__main__":

    # split = vhfdat.get_splits_combination("uoft-cs/cifar10", cache_dir=cache_dir)
    # dataset = hfdat.load_dataset("uoft-cs/cifar10", split="train", cache_dir=cache_dir)
    # dataset.set_format(type="torch", columns=["img","label"])
    # print(dataset[0])

    dataset = HuggingFaceDataset("uoft-cs/cifar10", 10)
    # print("Dataset Raw Index", type(dataset._raw_idx), dataset._raw_idx.shape, dataset._raw_idx)
    print(dataset[0])