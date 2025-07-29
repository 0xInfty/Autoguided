# This is an adaptation from code found at "EDM2 and Autoguidance" by Tero Karras et al
# https://github.com/NVlabs/edm2/blob/main/train_edm2.py licensed under CC BY-NC-SA 4.0
#
# Original copyright disclaimer:
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Train diffusion models according to the EDM2 recipe from the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)

import os
import json
import warnings
import click
import torch
from pyvtorch.aux import get_device_number

import karras.dnnlib as dnnlib
import karras.torch_utils.distributed as dist
import karras.training.training_loop as trn
from ours.dataset import DATASET_OPTIONS, get_dataset_kwargs

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Configuration presets.

def img_to_other(n, batch_size, n_train_total):
    return int(n_train_total*n/1281167/batch_size)*batch_size

def img_to_cifar(n, batch_size=2048):
    return img_to_other(n, batch_size, 50000)

def img_to_tiny(n, batch_size=2048):
    return img_to_other(n, batch_size, 100000)

config_presets = {
    'edm2-img512-xxs':  dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=64,  lr=0.0170, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xs':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-s':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-m':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-l':    dnnlib.EasyDict(duration=1792<<20, batch=2048, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xl':   dnnlib.EasyDict(duration=1280<<20, batch=2048, channels=384, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(duration=896<<20,  batch=2048, channels=448, lr=0.0065, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img64-xs':    dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=128, lr=0.0120, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-s':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-m':     dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-l':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-xl':    dnnlib.EasyDict(duration=640<<20,  batch=2048, channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-cifar10-xxs': dnnlib.EasyDict(duration=img_to_cifar(2048<<20), batch=2048, channels=64,
                                        lr=0.0170, decay=img_to_cifar(70e3*2048)/2048, rampup=img_to_cifar(10*1e6)/1e6,
                                        dropout=0.00, P_mean=-0.4, P_std=1.0,
                                        checkpoint_period=None, snapshot_period=40),
    'edm2-tiny-xxs':    dnnlib.EasyDict(duration=img_to_tiny(2048<<20), batch=2048, channels=64, 
                                        lr=0.0170, decay=img_to_tiny(70e3*2048)/2048, rampup=img_to_tiny(10*1e6)/1e6,
                                        dropout=0.00, P_mean=-0.4, P_std=1.0,
                                        checkpoint_period=1000, snapshot_period=500),
    'edm2-tiny-xs':     dnnlib.EasyDict(duration=img_to_tiny(2048<<20), batch=2048, channels=128, ref_channels=64,
                                        lr=0.0120, decay=img_to_tiny(70e3*2048)/2048, rampup=img_to_tiny(10*1e6)/1e6,
                                        dropout=0.00, ref_dropout=0.00, P_mean=-0.4, P_std=1.0,
                                        checkpoint_period=1000, snapshot_period=500),
}
config_presets["test-training"] = config_presets["edm2-cifar10-xxs"]
config_presets["test-training"].duration = 20*2048 # Just for 20 epochs
config_presets["test-training"].checkpoint_period = 20 # Just for 20 epochs
config_presets["test-random"] = config_presets["test-training"]
config_presets["test-random"]["selection_func"] = "ours.selection.random_baseline"

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='edm2-img512-s', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset.
    c.dataset_kwargs = get_dataset_kwargs(opts.dataset, use_labels=opts.get('cond', True))
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True, but no labels found in the dataset')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='karras.training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='karras.training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Hyperparameters.
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    c.network_kwargs = dnnlib.EasyDict(class_name='karras.training.networks_edm2.Precond', model_channels=opts.channels, dropout=opts.dropout)
    c.ref_network_kwargs = dnnlib.EasyDict(class_name='karras.training.networks_edm2.Precond')
    try: c.ref_network_kwargs.model_channels = opts.ref_channels or opts.channels
    except AttributeError: c.ref_network_kwargs.model_channels = opts.channels
    try: c.ref_network_kwargs.dropout = opts.ref_dropout
    except AttributeError: c.ref_network_kwargs.dropout = opts.dropout
    c.loss_kwargs = dnnlib.EasyDict(class_name='karras.training.training_loop.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std)
    c.lr_kwargs = dnnlib.EasyDict(func_name='karras.training.training_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay, rampup_Mimg=opts.rampup)

    # Data selection.
    c.ref_path = opts.get('ref_path', 0) or None
    c.selection = opts.get('selection', 0) or None
    c.selection_early = opts.get('selection_early', 0) or None
    c.selection_late = opts.get('selection_late', 0) or None
    c.selection_kwargs = dnnlib.EasyDict(N=opts.N, filter_ratio=opts.filter_ratio, learnability=opts.learnability,
                                         inverted=opts.inverted, numeric_stability_trick=opts.numeric_stability_trick)
    if opts.acid: c.selection_kwargs.func_name = 'ours.selection.jointly_sample_batch'
    else: c.selection_kwargs.func_name = 'ours.selection.random_baseline'
    if c.selection: c.lr_kwargs.update(dict(super_batch_size=opts.batch))
    
    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', True)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_period = opts.get('status', 0) or None
    c.snapshot_period = config_presets[preset].get("snapshot_period", opts.get('snapshot', 0)) or None
    c.checkpoint_period = config_presets[preset].get("checkpoint_period", opts.get('checkpoint', 0)) or None
    c.seed = opts.get('seed', 0)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    break_training = False
    if dist.get_rank() == 0:
        if not os.path.isdir(run_dir):
            dist.print0('Creating output directory...')
            os.makedirs(run_dir)
        elif os.path.isfile(os.path.join(run_dir, 'training_options.json')):
            with open(os.path.join(run_dir, 'training_options.json'), 'r') as f:
                old_c = json.load(f)
            for k in old_c.keys():
                if old_c[k] != c[k]: break_training = True
        if not break_training:
            with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
                json.dump(c, f, indent=2)
    torch.distributed.barrier()
    sync_tensor = torch.tensor([break_training], dtype=torch.bool, 
                               device=torch.device(f"cuda:{get_device_number()}"))
    torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.MAX)
    break_training = bool(sync_tensor.item())
    if break_training: raise ValueError("Training configuration does not match existing configuration")
    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    trn.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--dataset',          help='Dataset to be used', metavar='STR',                   type=click.Choice(list(DATASET_OPTIONS.keys())), default="imagenet", show_default=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, required=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='INT',                     type=int, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# Data selection.
@click.option('--guide-path', 'ref_path',
                help='Use auto-guidance?', metavar='PATH', type=str, default=None, show_default=True)
@click.option('--selection/--no-selection', 'selection',
                help='Use data selection?', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--acid/--no-acid', 'acid',
                help='Use ACID batch selection?', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--filt',  'filter_ratio',
                help='ACID filter ratio', metavar='FLOAT', type=float, default=0.8, show_default=True)
@click.option('--n', 'N', 
                help='ACID number of data selection iterations', metavar='INT', type=int, default=8, show_default=True)
@click.option('--diff/--no-diff', 'learnability',
                help='Use ACID learnability score?', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--invert/--no-invert', 'inverted',
                help='Use inverted ACID scores?', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--late/--no-late', 'selection_late',
                help='Delay ACID start?', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--early/--no-early', 'selection_early',
                help='Run ACID just at the beginning?', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--trick/--no-trick', 'numeric_stability_trick',  
                help='Use the softmax stability trick?', metavar='BOOL', type=bool, default=False, show_default=True)

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='INT',             type=int, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Epoch period of status prints', metavar='INT',        type=int, default=int((128<<10)/2048), show_default=True)
@click.option('--snapshot',         help='Epoch period of network snapshots', metavar='INT',    type=int, default=int((8<<20)/2048), show_default=True)
@click.option('--checkpoint',       help='Epoch period of training checkpoints', metavar='INT', type=int, default=int((128<<20)/2048), show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

def cmdline(outdir, dry_run, **opts):
    """Train diffusion models according to the EDM2 recipe from the paper
    "Analyzing and Improving the Training Dynamics of Diffusion Models".

    Examples:

    \b
    # Train XS-sized model for ImageNet-512 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train_edm2.py \\
        --outdir=training-runs/00000-edm2-img512-xs \\
        --data=datasets/img512-sd.zip \\
        --preset=edm2-img512-xs \\
        --batch-gpu=32

    \b
    # To resume training, run the same command again.
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')

    outdir = os.path.join(dirs.MODELS_HOME, "Images", outdir)
    opts = dnnlib.EasyDict(opts)
    if opts.dataset=="imagenet":
        if "img512" in opts.preset:
            opts.update(dict(data=os.path.join(dirs.DATA_HOME, "img512.zip")))
        elif "img64" in opts.preset:
            opts.update(dict(data=os.path.join(dirs.DATA_HOME, "img512-sd.zip")))
        else: raise ValueError("Unknown ImageNet dataset")
    try: opts.ref_path = os.path.join(dirs.MODELS_HOME, "Images", opts.ref_path)
    except TypeError: opts.ref_path = None

    opts.selection = opts.selection or opts.acid
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=outdir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------