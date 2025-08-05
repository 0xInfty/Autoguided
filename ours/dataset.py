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
    "img512": dict(class_name='karras.training.dataset.ImageFolderDataset', path=os.path.join(dirs.DATA_HOME, "img512.zip")),
    "img64": dict(class_name='karras.training.dataset.ImageFolderDataset', path=os.path.join(dirs.DATA_HOME, "img512-sd.zip")),
    "cifar10": dict(class_name='ours.dataset.HuggingFaceDataset', path="uoft-cs/cifar10", n_classes=10, key_image="img", key_label="label"),
    "tiny": dict(class_name='ours.dataset.TinyImageNetDataset', path="zh-plus/tiny-imagenet", n_classes=200, key_image="image", key_label="label"),
    "folder": dict(class_name='karras.training.dataset.ImageFolderDataset'),
}

def get_dataset_kwargs(dataset_name, image_path=None, use_labels=True):
    try:
        dataset_kwargs = DATASET_OPTIONS[dataset_name]
    except KeyError:
        if image_path is not None: 
            dataset_kwargs = DATASET_OPTIONS["folder"]
            dataset_kwargs["path"] = image_path
        else:
            raise ValueError("Dataset path is required for 'folder' datasets")
    dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
    try:
        dataset_kwargs.path
    except AttributeError:
        dataset_kwargs.path = image_path
    if dataset_name != "folder": 
        dataset_kwargs.name = dataset_name
    else:
        dataset_kwargs.name = dataset_kwargs.path
    dataset_kwargs.use_labels = use_labels
    return dataset_kwargs

class HuggingFaceDataset(Dataset):

    def __init__(self,
        path,                   # Path to Hugging Face dataset
        n_classes,              # Specify number of classes for one hot encoding
        key_image,              # String identifier of the images column
        key_label,              # String identifier of the labels column
        resolution      = None, # Ensure specific resolution, None = anything goes.
        name            = None, # Name of the dataset, optional
        cache_dir       = dirs.DATA_HOME, # Cache dir to store the Hugging Face dataset
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        
        self._set_up(path, n_classes, key_image, key_label, cache_dir)

        name = name or os.path.splitext(path)[-1]
        raw_shape = [len(self._dataset)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
    
    def _set_up(self, path, n_classes, key_image, key_label, cache_dir=dirs.DATA_HOME):

        self._path = path        
        self._cache_dir = cache_dir

        self._dataset = hfdat.load_dataset(path, split="train", cache_dir=self.cache_dir)
        
        self._n_classes = n_classes
        self._key_image = key_image
        self._key_label = key_label
        if self.key_image not in self.data.column_names:
            raise ValueError(f"Hugging Face Dataset has no column {self.key_image}")
        if self.key_label not in self.data.column_names:
            raise ValueError(f"Hugging Face Dataset has no column {self.key_label}")
        
        self.data.set_format(type="torch", columns=[self.key_image,self.key_label])

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
        if list(image.shape) != self._raw_shape[1:]:
            if list(image.shape) == self._raw_shape[2:]: # Grayscale image detected
                image = image.repeat(self._raw_shape[1], 1, 1) # Repeat on all channels
            else: # Something else is wrong
                raise ValueError(f"Image {idx} does not have the right shape: expected {self._raw_shape[1:]} but found {list(image.shape)}")
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
        d.raw_label = int(self._load_raw_label(d.raw_idx))
        try:
            d.name_label = self.data.features[self.key_label].names[d.raw_label]
        except:
            d.name_label = None
        return d

class TinyImageNetDataset(HuggingFaceDataset):

    def __init__(self,
        path,                   # Path to Hugging Face dataset
        n_classes,              # Specify number of classes for one hot encoding
        key_image,              # String identifier of the images column
        key_label,              # String identifier of the labels column
        resolution      = None, # Ensure specific resolution, None = anything goes.
        name            = None, # Name of the dataset, optional
        cache_dir       = dirs.DATA_HOME, # Cache dir to store the Hugging Face dataset
        names_filename  = "words.txt", # Classes' textual names (e.g. Goldfish for n01443537)
        transform       = lambda x : x,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        
        super().__init__(path, n_classes, key_image, key_label, 
                         resolution, name, cache_dir, **super_kwargs)
        
        dataset_dir = "___".join(os.path.split(path))
        names_filepath = os.path.join(self.cache_dir, dataset_dir, names_filename)
        if os.path.isfile(names_filepath):
            class_names = {k:None for k in self.data.features["label"].names}
            with open(names_filepath, "r", encoding='UTF-8') as f:
                for line in f:
                    k, name = line.rstrip().split("\t")
                    class_names[k] = name.capitalize()
            self._class_names = class_names
        else: self._class_names = None

        self.transform = transform

    @property
    def class_names(self):
        return self._class_names

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = int(self._load_raw_label(d.raw_idx))
        d.name_label = self.data.features[self.key_label].names[d.raw_label]
        if self.class_names is not None:
            d.words_label = self.class_names[d.name_label]
        return d
    
    def __getitem__(self, idx):
        return self.transform(super().__getitem__(idx))