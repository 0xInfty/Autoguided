# This is an adaptation from code found at "EDM2 and Autoguidance" by Tero Karras et al
# https://github.com/NVlabs/edm2/blob/main/generate_images.py licensed under CC BY-NC-SA 4.0
#
# Original copyright disclaimer:
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generate random images using the given model."""

import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))

import math
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import matplotlib.pyplot as plt

from karras.dnnlib.util import EasyDict, construct_class_by_name, open_url, call_func_by_name
import karras.torch_utils.distributed as dist
from karras.training.encoders import PRETRAINED_HOME
import Images.calculate_metrics as calc
import ours.visualize as vis

import logs

log = logs.create_logger("errors")

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')

#----------------------------------------------------------------------------
# Configuration presets.

DEFAULT_SAMPLER = EasyDict(dict(
    num_steps = 32,         # Number of sampling steps
    sigma_min = 0.002,      # Lowest noise level
    sigma_max = 80,         # Highest noise level
    rho = 7,                # Time step exponent
    S_churn = 0,            # Stochasticity strength
    S_min = 0,              # Stoch. min noise level
    S_max = float("inf"),   # Stoch. max noise level
    S_noise = 1,            # Stoch. noise inflation
))

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'
models_dir = os.path.join(dirs.MODELS_HOME, "Images")

config_presets = {
    'edm2-img512-xs-fid':              EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),      # fid = 3.53
    'edm2-img512-xs-dino':             EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),      # fd_dinov2 = 103.39
    'edm2-img512-s-fid':               EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),       # fid = 2.56
    'edm2-img512-s-dino':              EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),       # fd_dinov2 = 68.64
    'edm2-img512-m-fid':               EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),       # fid = 2.25
    'edm2-img512-m-dino':              EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),       # fd_dinov2 = 58.44
    'edm2-img512-l-fid':               EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),       # fid = 2.06
    'edm2-img512-l-dino':              EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),       # fd_dinov2 = 52.25
    'edm2-img512-xl-fid':              EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),      # fid = 1.96
    'edm2-img512-xl-dino':             EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),      # fd_dinov2 = 45.96
    'edm2-img512-xxl-fid':             EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'),     # fid = 1.91
    'edm2-img512-xxl-dino':            EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'),     # fd_dinov2 = 42.84
    'edm2-img64-s-fid':                EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),        # fid = 1.58
    'edm2-img64-m-fid':                EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),        # fid = 1.43
    'edm2-img64-l-fid':                EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),        # fid = 1.33
    'edm2-img64-xl-fid':               EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),       # fid = 1.33
    'edm2-img512-xs-guid-fid':         EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance_weight=1.40), # fid = 2.91
    'edm2-img512-xs-guid-dino':        EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance_weight=1.70), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-fid':          EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance_weight=1.40), # fid = 2.23
    'edm2-img512-s-guid-dino':         EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance_weight=1.90), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-fid':          EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance_weight=1.20), # fid = 2.01
    'edm2-img512-m-guid-dino':         EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance_weight=2.00), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-fid':          EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance_weight=1.20), # fid = 1.88
    'edm2-img512-l-guid-dino':         EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance_weight=1.70), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-fid':         EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance_weight=1.20), # fid = 1.85
    'edm2-img512-xl-guid-dino':        EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance_weight=1.70), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-fid':        EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance_weight=1.20), # fid = 1.81
    'edm2-img512-xxl-guid-dino':       EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance_weight=1.70), # fd_dinov2 = 33.09
    'edm2-img512-s-autog-fid':         EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.070.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.125.pkl',        guidance_weight=2.10), # fid = 1.34
    'edm2-img512-s-autog-dino':        EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.120.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.165.pkl',        guidance_weight=2.45), # fd_dinov2 = 36.67
    'edm2-img512-xxl-autog-fid':       EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.075.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.155.pkl',         guidance_weight=2.05), # fid = 1.25
    'edm2-img512-xxl-autog-dino':      EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.130.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.205.pkl',         guidance_weight=2.30), # fd_dinov2 = 24.18
    'edm2-img512-s-uncond-autog-fid':  EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.070.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.110.pkl', guidance_weight=2.85), # fid = 3.86
    'edm2-img512-s-uncond-autog-dino': EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.090.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.125.pkl', guidance_weight=2.90), # fd_dinov2 = 90.39
    'edm2-img64-s-autog-fid':          EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.110.pkl',         guidance_weight=1.70), # fid = 1.01
    'edm2-img64-s-autog-dino':         EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.105.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.175.pkl',         guidance_weight=2.20), # fd_dinov2 = 31.85
    'edm2-CIFAR10-xxs-ref':            EasyDict(net=os.path.join(models_dir, "01_CIFAR10", "Ref", "00", 'network-snapshot-0000639-0.100.pkl')),
    'edm2-CIFAR10-xxs-ajest-final':    EasyDict(net=os.path.join(models_dir, "01_CIFAR10", "AJEST", "00", 'network-snapshot-0042879-0.100.pkl')),
    'edm2-CIFAR10-xxs-ajest-nimg':     EasyDict(net=os.path.join(models_dir, "01_CIFAR10", "Ref", "00", 'network-snapshot-0000639-0.100.pkl')),
    'edm2-CIFAR10-xxs-eajest':         EasyDict(net=os.path.join(models_dir, "01_CIFAR10", 'network-snapshot-0117440-0.100.pkl')),       # fid = unknown
    'edm2-tiny-ref-0.10':              EasyDict(net=os.path.join(models_dir, "04_Tiny_LR/Ref/00/network-snapshot-0005159-0.100.pkl")),       # fid = unknown
    'edm2-tiny-ref-0.05':              EasyDict(net=os.path.join(models_dir, "04_Tiny_LR/Ref/00/network-snapshot-0005159-0.050.pkl")),       # fid = unknown
    'edm2-tiny-base-0.10':             EasyDict(net=os.path.join(models_dir, "04_Tiny_LR/Baseline/00/network-snapshot-0020479-0.100.pkl")),       # fid = unknown
    'edm2-tiny-base-0.50':             EasyDict(net=os.path.join(models_dir, "04_Tiny_LR/Baseline/00/network-snapshot-0020479-0.050.pkl")),       # fid = unknown
}

#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_full_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=DEFAULT_SAMPLER.num_steps, 
    sigma_min=DEFAULT_SAMPLER.sigma_min, sigma_max=DEFAULT_SAMPLER.sigma_max, 
    rho=DEFAULT_SAMPLER.rho, guidance=1,
    S_churn=DEFAULT_SAMPLER.S_churn, S_min=DEFAULT_SAMPLER.S_min, 
    S_max=DEFAULT_SAMPLER.S_max, S_noise=DEFAULT_SAMPLER.S_noise,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    x_trajectory = [x_next]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_trajectory.append(x_next)

    return x_next, x_trajectory

def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    
    x_next, _ = edm_full_sampler(net, noise, labels=labels, gnet=gnet, num_steps=num_steps, 
                                 sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, guidance=guidance,
                                 S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise,
                                 dtype=dtype, randn_like=randn_like)
    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = use automatic selection.
    random_class        = True,                 # Automatic selection can be uniformly random or forced exact distribution.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    dist.print0("Sampler kwargs", sampler_kwargs)
    
    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with open_url(net, verbose=(verbose and dist.get_rank() == 0),
                      cache_dir=PRETRAINED_HOME) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = construct_class_by_name(class_name='karras.training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with open_url(gnet, verbose=(verbose and dist.get_rank() == 0),
                      cache_dir=PRETRAINED_HOME) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = EasyDict(images=None, labels=None, labels_indices=None, noise=None, 
                             batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        if class_idx is not None:
                            r.labels = torch.zeros((net.label_dim,net.label_dim), device=device)
                            r.labels[:, class_idx] = 1
                            r.labels_indices = np.ones(len(r.seeds), dtype=np.uint32)*class_idx
                        elif random_class:
                            r.labels_indices = rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)
                            r.labels = torch.eye(net.label_dim, device=device)[r.labels_indices]
                        elif not random_class:
                            # If the overall seeds is a continuous range of integers, we'll get an equal number of examples per class
                            r.labels_indices = np.array([i%(net.label_dim-1) for i in r.seeds], dtype=np.uint32)
                            r.labels = torch.eye(net.label_dim, device=device)[r.labels_indices]
                            # No images are generated using unconditional generation 
                            # because we go from 0 to label_dim-1; label_dim is the null class

                    # Generate images.
                    latents = call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
                    if isinstance(latents, torch.Tensor):
                        r.images = encoder.decode(latents)
                        r.trajectories = None
                    else:
                        r.images = encoder.decode(latents[0])
                        r.trajectories = torch.stack(latents[1]).swapaxes(0,1)

                    # Save images.
                    if outdir is not None:
                        for idx, (seed, image, lab) in enumerate(zip(r.seeds, 
                                                                     r.images.permute(0, 2, 3, 1).cpu().numpy(), 
                                                                     r.labels_indices)):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}_{lab:05d}.png'))
                            if r.trajectories is not None:
                                trajs = r.trajectories[idx]
                                fig, axes = plt.subplots(ncols=math.ceil((len(trajs)-1)/4), nrows=4)
                                for t, (x_t, ax) in enumerate(zip(trajs[:-1], axes.flatten())):
                                    ax.imshow(encoder.decode(x_t).cpu().detach().numpy().swapaxes(0,1).swapaxes(1,2))
                                    ax.axis('off')
                                    ax.set_title(str(t))
                                plt.suptitle(f"Label {int(lab)}")
                                plt.tight_layout()
                                plt.show()
                                fig_dir = image_dir.replace("gen_images","gen_trajectories")
                                if fig_dir == image_dir: fig_dir = os.path.join(image_dir,"gen_trajectories")
                                os.makedirs(fig_dir, exist_ok=True)
                                plt.savefig(os.path.join(fig_dir, f'{seed:06d}_{lab:05d}.png'))
                                plt.close(fig)

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

def visualize_generated_images(generated_images_path, n_images=200, batch_size=100,
                               plot_labels=False, save_dir=None, tight_layout=False):

    dataset_kwargs = calc.get_dataset_kwargs("generated", image_path=generated_images_path)
    dataset_obj = construct_class_by_name(**dataset_kwargs, random_seed=0)

    saving = save_dir is not None
    if tight_layout: kwargs = dict(bbox_inches="tight")
    else: kwargs = dict()
    for start, end in zip(np.arange(0, n_images+batch_size, batch_size)[:-1], 
                          np.arange(0, n_images+batch_size, batch_size)[1:]):
        fig, _ = vis.visualize_images(dataset_obj, np.arange(start, end), n_cols=10)
        if plot_labels:
            fig_2, _ = vis.visualize_classes(dataset_obj, np.arange(start, end), n_cols=10)
        if saving:
            fig.savefig(os.path.join(save_dir, f"imgs-{start:07.0f}-{end:07.0f}.png"), **kwargs)
            plt.close(fig)
            if plot_labels:
                fig_2.savefig(os.path.join(save_dir, f"labs-{start:07.0f}-{end:07.0f}.png"), **kwargs)
                plt.close(fig_2)

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--gnet',                     help='Guiding network pickle filename', metavar='PATH|URL',             type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, default=None, show_default=True)
@click.option('--results/--no-results',     help='Whether to send output to Results or to Data', metavar='BOOL',    type=bool, default=False, show_default=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--random/--no-random', 'random_class', 
                                            help='Whether to generate images from random classes', metavar='BOOL',  type=bool, default=True, show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=DEFAULT_SAMPLER.num_steps, show_default=True)
@click.option('--sigma-min', "sigma_min",   help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=DEFAULT_SAMPLER.sigma_min, show_default=True)
@click.option('--sigma-max', "sigma_max",   help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=DEFAULT_SAMPLER.sigma_max, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=DEFAULT_SAMPLER.rho, show_default=True)
@click.option('--guide-weight', "guidance_weight",
                                            help='Guidance weight  [default: 1; no guidance]', metavar='FLOAT',     type=float, default=None)
@click.option('--S-churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=DEFAULT_SAMPLER.S_churn, show_default=True)
@click.option('--S-min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=DEFAULT_SAMPLER.S_min, show_default=True)
@click.option('--S-max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=DEFAULT_SAMPLER.S_max, show_default=True)
@click.option('--S-noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=DEFAULT_SAMPLER.S_noise, show_default=True)

@click.option('--guidance/--no-guidance',   help='Apply guidance, if possible?', metavar='BOOL',                    type=bool, default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance_weight is None or opts.guidance_weight == 1:
        opts.guidance_weight = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')
    
    # Make changes, if needed
    if opts.guidance_weight != 1 and not opts.guidance:
        opts.guidance_weight = 1
        log.info("Guidance deactivated due to --no-guidance flag")
    elif opts.guidance and opts.guidance_weight == 1:
        log.info("Guidance cannot be activated: no guidance weight in configuration preset")
    opts.guidance = opts.guidance_weight # Rename for `generate_images` to work
    del opts.guidance_weight
    if opts.outdir is None:
        opts.outdir = os.path.splitext(opts.net)[0].split( models_dir+os.sep )[-1]
    if opts.results:
        opts.outdir = os.path.join(dirs.RESULTS_HOME, "Images", opts.outdir)
    else: 
        opts.outdir = os.path.join(dirs.DATA_HOME, "generated", opts.outdir)
    del opts.results

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------