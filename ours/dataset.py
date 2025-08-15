from functools import cache
import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)

import os
import json
import pickle
import zipfile
import numpy as np
import torch
import datasets as hfdat
import matplotlib.pyplot as plt
import PIL
from pyvtools.text import find_numbers

import karras.dnnlib as dnnlib
from karras.training.dataset import Dataset, ImageFolderDataset
from karras.training.encoders import Identity
from ours.utils import find_all_indices

DATASET_OPTIONS = {
    "img512": dict(class_name='karras.training.dataset.ImageFolderDataset', path=os.path.join(dirs.DATA_HOME, "img512.zip")),
    "img64": dict(class_name='karras.training.dataset.ImageFolderDataset', path=os.path.join(dirs.DATA_HOME, "img512-sd.zip")),
    "cifar10": dict(class_name='ours.dataset.HuggingFaceDataset', path="uoft-cs/cifar10", n_classes=10, key_image="img", key_label="label"),
    "tiny": dict(class_name='ours.dataset.TinyImageNetDataset', path="zh-plus/tiny-imagenet", n_classes=200, key_image="image", key_label="label"),
    "folder": dict(class_name='karras.training.dataset.ImageFolderDataset'),
    "generated": dict(class_name='ours.dataset.GeneratedFolderDataset'),
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
        transform       = None, # Optional image transformation
        cache_dir       = dirs.DATA_HOME, # Cache dir to store the Hugging Face dataset
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        
        self._path = path
        self._set_up_huggingface(n_classes, key_image, key_label, cache_dir)
        self._set_up_transform(transform)
        name, raw_shape = self._infer_name_and_raw_shape(path, name, resolution)
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
    
    def _set_up_huggingface(self, n_classes, key_image, key_label, cache_dir):

        self._cache_dir = cache_dir
        self._dataset = hfdat.load_dataset(self.path, split="train", cache_dir=self.cache_dir)
        
        self._n_classes = n_classes
        self._key_image = key_image
        self._key_label = key_label
        if self.key_image not in self.data.column_names:
            raise ValueError(f"Hugging Face Dataset has no column {self.key_image}")
        if self.key_label not in self.data.column_names:
            raise ValueError(f"Hugging Face Dataset has no column {self.key_label}")
        
        self.data.set_format(type="torch", columns=[self.key_image,self.key_label])
        
    def _set_up_transform(self, transform=None):

        if transform is None:
            transform = Identity()
        self.transform = transform

    def _infer_name_and_raw_shape(self, path, name=None, resolution=None):

        name = name or os.path.splitext(os.path.basename(path))[-1]
        raw_shape = [len(self._dataset)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
    
        return name, raw_shape

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
    
    def __getimg__(self, idx):
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
        return image

    def __getitem__(self, idx):
        image = self.__getimg__(idx)
        if list(image.shape) != self._raw_shape[1:]:
            if list(image.shape) == self._raw_shape[2:]: # Grayscale image detected
                image = image.repeat(self._raw_shape[1], 1, 1) # Repeat on all channels
            else: # Something else is wrong
                raise ValueError(f"Image {idx} does not have the right shape: expected {self._raw_shape[1:]} but found {list(image.shape)}")
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return idx, self.transform(image), self.get_label(idx)

    def _load_raw_image(self, raw_idx):
        raw_idx = np.array([raw_idx], dtype=int)
        # print("Image Raw Index", type(raw_idx), raw_idx)
        img = self.data[raw_idx][self.key_image].squeeze().float() # CHW
        # print("Image", type(img), img.dtype, img.shape, img)
        return img # Return float32 Torch tensor

    def _load_raw_label(self, raw_idx):
        raw_idx = np.array([raw_idx], dtype=int)
        return self.data[raw_idx][self.key_label]

    def _set_up_raw_labels(self):
        self._raw_labels = [self.data[int(raw_idx)][self.key_label] for raw_idx in self._raw_idx]

    def get_label(self, idx):
        raw_idx = self.get_raw_idx(idx)
        label = self._load_raw_label(raw_idx) # Assume this is an int
        onehot = np.zeros(self.label_shape, dtype=np.float32)
        onehot[label] = 1
        return onehot.copy()

    @property
    def has_onehot_labels(self):
        return False # Assume dataset has just int labels ==> Get label converts it instead

    @property
    def class_names(self):
        return self.data.features[self.key_label].names

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = self.get_raw_idx(idx)
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = int(self._load_raw_label(d.raw_idx))
        try:
            d.name_label = self.class_names[d.raw_label]
        except:
            d.name_label = None
        return d

    def get_raw_label(self, idx):
        raw_idx = self.get_raw_idx(idx)
        return int(self._load_raw_label(raw_idx))

    def get_name_label(self, idx):
        try:
            raw_label = self.get_raw_label(idx)
            return self.class_names[raw_label]
        except:
            raise NotImplementedError("This dataset does not have names for its labels")
    
    def get_name_from_label(self, label):
        try:
            return self.class_names[label]
        except:
            raise NotImplementedError("This dataset does not have names for its labels")
    
    def get_label_from_name(self, name):
        try:
            return self.class_names.index(name)
        except:
            raise NotImplementedError("This dataset does not have names for its labels")
    
    def visualize(self, idx):
        img = self.__getimg__(idx)
        try:
            lab = self.get_name_label(idx)
        except NotImplementedError:
            lab = self.get_raw_label(idx)
        plt.imshow( img.detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2).astype(np.float32)/255 )
        plt.title(f"Class {lab}")

class TinyImageNetDataset(HuggingFaceDataset):

    def __init__(self,
        path,                   # Path to Hugging Face dataset
        n_classes,              # Specify number of classes for one hot encoding
        key_image,              # String identifier of the images column
        key_label,              # String identifier of the labels column
        resolution      = None, # Ensure specific resolution, None = anything goes.
        name            = None, # Name of the dataset, optional
        cache_dir       = dirs.DATA_HOME, # Cache dir to store the Hugging Face dataset
        words_filename  = "words.txt", # Classes' textual names (e.g. Goldfish for n01443537)
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        super().__init__(path=path, n_classes=n_classes, key_image=key_image, key_label=key_label,
                         resolution=resolution, name=name, cache_dir=cache_dir, **super_kwargs)
        self._set_up_class_words(path, words_filename)
    
    def _set_up_class_words(self, path, words_filename):
        
        dataset_dir = "___".join(os.path.split(path))
        names_filepath = os.path.join(self.cache_dir, dataset_dir, words_filename)
        if os.path.isfile(names_filepath):
            class_words = {k:None for k in self.data.features["label"].names}
            with open(names_filepath, "r", encoding='UTF-8') as f:
                for line in f:
                    k, name = line.rstrip().split("\t")
                    class_words[k] = name.capitalize()
            self._class_words = class_words
        else: self._class_words = None

    @property
    def class_words(self):
        return self._class_words

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = int(self._load_raw_label(d.raw_idx))
        d.name_label = self.class_names[d.raw_label]
        if self.class_words is not None:
            d.words_label = self.class_words[d.name_label]
        return d

    def get_name_label(self, idx):
        raw_label = self.get_raw_label(idx)
        return self.class_names[raw_label]
    
    def get_name_from_label(self, label):
        return self.class_names[label]
    
    def get_label_from_name(self, name):
        try:
            return self.class_names.index(name)
        except:
            words_label = self.get_words_from_name(name)
            all_names = self.get_all_names_from_words_label(words_label)
            for n in all_names:
                if n in self.class_names:
                    return self.class_names.index(n)

    def get_words_from_name(self, name):
        return self.class_words[name]

    def get_words_label(self, idx):
        if self.class_words is not None:
            name_label = self.get_name_label(idx)
            return self.class_words[name_label]
        else:
            raise NotImplementedError("This dataset does not have words labels")
    
    def visualize(self, idx):
        img = self.__getimg__(idx)
        try:
            lab = self.get_words_label(idx)
            title = lab
        except NotImplementedError:
            lab = self.get_name_label(idx)
            title = f"Class {lab}"
        plt.imshow( img.detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2).astype(np.float32)/255 )
        plt.title( title )
    
    def get_words_from_label(self, raw_label):
        return self.class_words[self.get_name_from_label(raw_label)]
    
    def get_all_names_from_words_label(self, words):
        indices = find_all_indices(words, list(self.class_words.values()))
        names = [list(self.class_words.keys())[i] for i in indices] 
        return names
    
    def get_all_names_from_label(self, label):
        words = self.get_words_from_label(label)
        return self.get_all_names_from_words_label(words)
    
class GeneratedFolderDataset(TinyImageNetDataset):

    def __init__(self,
        path,                   # Path to directory or zip, instead of the Hugging Face dataset
        resolution      = None, # Ensure specific resolution, None = anything goes.
        name            = None, # Name of the dataset, optional
        transform       = None, # Optional image transformation
        use_labels  = True,     # Enable conditioning labels? False = label dimension is zero.
        names_filename  = "tiny_huggingface_names.pkl", # Classes' names (e.g. n01443537 for class 0)
        words_filename  = "words.txt", # Classes' textual names (e.g. Goldfish for n01443537)
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
        cache_dir   = dirs.DATA_HOME, # Cache dir storing the Hugging Face dataset with class words information
    ):

        self._path = path
        self._set_up_transform(transform)
        self._inspect_and_setup_folder(path)
        self._set_up_class_names(names_filename)
        name, raw_shape = self._infer_name_and_raw_shape(path, name, resolution)
        self._set_up(name, raw_shape, use_labels, max_size, xflip, random_seed, cache)
        self._cache_dir = cache_dir
        self._set_up_class_words(path, words_filename)

    def _inspect_and_setup_folder(self, path): # From karras.training.ImageFolderDataset

        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

    def _set_up_raw_labels(self): # From karras.training.Dataset
        self._raw_labels = self._load_raw_labels() if self._use_labels else None
        if self._raw_labels is None:
            self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
        assert isinstance(self._raw_labels, np.ndarray)
        assert self._raw_labels.shape[0] == self._raw_shape[0]
        assert self._raw_labels.dtype in [np.float32, np.int64]
        if self._raw_labels.dtype == np.int64:
            assert self._raw_labels.ndim == 1
            assert np.all(self._raw_labels >= 0)
    
    def _set_up_class_names(self, names_filename): # Original from this class

        names_filepath = os.path.join(dirs.DATA_HOME, names_filename)
        with open(names_filepath, 'rb') as f:
            names = pickle.load(f)
        self._class_names = names

    def _infer_name_and_raw_shape(self, path, name=None, resolution=None): # Original from this class

        name = name or os.path.splitext(os.path.basename(path))[-1]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
    
        return name, raw_shape

    @staticmethod
    def _file_ext(fname): # From karras.training.ImageFolderDataset
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self): # From karras.training.ImageFolderDataset
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname): # From karras.training.ImageFolderDataset
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self): # From karras.training.ImageFolderDataset
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __del__(self): # From karras.training.ImageFolderDataset
        try:
            self.close()
        except:
            pass

    def __getstate__(self): # From karras.training.ImageFolderDataset
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx): # From karras.training.ImageFolderDataset
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        image = torch.Tensor(image).float() # Addition to the original method
        return image

    def _load_raw_labels(self): # Original from this class
        labels = [find_numbers(fname)[-1] for fname in self._all_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_raw_label(self, raw_idx): # Original from this class
        raw_idx = np.array([raw_idx], dtype=int)
        return self.raw_labels[raw_idx]

    @property
    def label_shape(self):
        if self._label_shape is None:
            if self.raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(self.raw_labels)) + 1]
            else:
                self._label_shape = self.raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self): # From karras.training.Dataset
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def n_classes(self): # From karras.training.Dataset
        return self.label_dim