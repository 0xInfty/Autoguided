# This is an adaptation from code found at "EDM2 and Autoguidance" by Tero Karras et al
# https://github.com/NVlabs/edm2/blob/main/training/encoders.py licensed under CC BY-NC-SA 4.0
#
# Original copyright disclaimer:
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Converting between pixel and latent representations of image data."""

import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)

import os
import warnings
import numpy as np
import torch
import torchvision.transforms as transforms
import karras.torch_utils.persistence as persistence
import karras.torch_utils.misc as misc

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

PRETRAINED_HOME = os.path.join(dirs.MODELS_HOME, "Images", "00_PreTrained")
if not os.path.isdir(PRETRAINED_HOME): os.mkdir(PRETRAINED_HOME)

#----------------------------------------------------------------------------
# Simple transformations

def from_8_bit_to_0_1(img):
    return img / 255.

def from_8_bit_to_minus1_1(img):
    return img / 127.5 - 1

class From8bitTo01(torch.nn.Module):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    def forward(self, img):
        return from_8_bit_to_0_1(img)
    
class From8bitToMinus11(torch.nn.Module):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    def forward(self, img):
        return from_8_bit_to_minus1_1(img)

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # Called during preprocessing
        return x

    def encode_latents(self, x): # 8 bits [0,255] to [-1,+1] during training/inference
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # Normalized [-1,+2] to 8 bits [0,255] during training/inference
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)

#----------------------------------------------------------------------------
# Pre-trained VAE encoder from Stability AI.

@persistence.persistent_class
class StabilityVAEEncoder(Encoder):
    def __init__(self,
        vae_name    = 'stabilityai/sd-vae-ft-mse',  # Name of the VAE to use.
        raw_mean    = [5.81, 3.25, 0.12, -2.15],    # Assumed mean of the raw latents.
        raw_std     = [4.17, 4.62, 3.71, 3.28],     # Assumed standard deviation of the raw latents.
        final_mean  = 0,                            # Desired mean of the final latents.
        final_std   = 0.5,                          # Desired standard deviation of the final latents.
        batch_size  = 8,                            # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.vae_name = vae_name
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = None

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_stability_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        d = self._vae.encode(x)['latent_dist']
        return torch.cat([d.mean, d.std], dim=1)

    def _run_vae_decoder(self, x):
        return self._vae.decode(x)['sample']

    def encode_pixels(self, x): # raw pixels => raw latents
        self.init(x.device)
        x = x.to(torch.float32) / 255
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

    def encode_latents(self, x): # raw latents => final latents
        mean, std = x.to(torch.float32).chunk(2, dim=1)
        x = mean + torch.randn_like(mean) * std
        x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        return x

    def decode(self, x): # final latents => raw pixels
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = x.clamp(0, 1).mul(255).to(torch.uint8)
        return x

#----------------------------------------------------------------------------

def load_stability_vae(vae_name='stabilityai/sd-vae-ft-mse', device=torch.device('cpu')):
    
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HOME'] = PRETRAINED_HOME

    import diffusers # pip install diffusers # pyright: ignore [reportMissingImports]
    try:
        # First try with local_files_only to avoid consulting tfhub metadata if the model is already in cache.
        vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, cache_dir=PRETRAINED_HOME, local_files_only=True)
    except:
        # Could not load the model from cache; try without local_files_only.
        vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, cache_dir=PRETRAINED_HOME)
    return vae.eval().requires_grad_(False).to(device)

#----------------------------------------------------------------------------