import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))

import shutil
import pickle
import json
import tqdm
import click
import time
from datetime import datetime
import math
import numpy as np
import torch
import torchvision as torchv
import torchvision.transforms as transforms
import wandb
import timm
from pyvtools.text import filter_by_string_must, find_numbers
from pyvtorch.aux import load_weights_and_check

import karras.torch_utils.distributed as dist
from karras.dnnlib.util import EasyDict, construct_class_by_name
from karras.training.encoders import PRETRAINED_HOME, From8bitTo01, From8bitToMinus11, FromNumpyToTorch
from torchvision.transforms.functional import InterpolationMode
from karras.torch_utils.misc import InfiniteSampler
from jeevan.wavemix.classification import WaveMix
from generate_images import DEFAULT_SAMPLER, generate_images, parse_int_list
import calculate_metrics as calc
import get_validation_metrics as valm
import reconstruct_phema as recp

from ours.dataset import DATASET_OPTIONS
from ours.utils import get_wandb_id


# guidance_weights = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5]
# emas = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 , 0.11, 0.12, 0.13, 0.14, 0.15]
# CUDA_VISIBLE_DEVICES=0 --> emas in [0.04, 0.06, 0.08, 0.1 , 0.12, 0.14]
# CUDA_VISIBLE_DEVICES=1 --> emas in [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]

def calculate_metrics_for_grid_search(
        emas, guidance_weights,
        super_dir, guide_path, out_epoch=None,
        fd_metrics=True, class_metrics=True, 
        metrics_filepath="grid_search.json", 
        dataset_name = "tiny",          # Dataset used for training.
        ref_path = None,                # Filepath to dataset reference metrics.
        class_idx = None,               # Class label. None = use automatic selection.
        random_class = False,           # Automatic selection can be uniformly random or forced exact distribution.
        seeds = range(0, 2000),         # List of random seeds.
        save_nimg = 0,                  # How many images to keep, the rest will be deleted.
        device = torch.device("cuda"),  # Which compute device to use.
        verbose=True, **sampler_kwargs):

    assert fd_metrics or class_metrics, ValueError("Nothing to calculate")
    assert dataset_name in DATASET_OPTIONS.keys(), "Unrecognized dataset"
    if dataset_name=="folder": raise NotImplementedError

    dist.init()
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)

    # Get reference stats
    if fd_metrics:
        # if dataset_name=="folder" and ref_path is None: 
        #     raise ValueError("Reference path needed for unrecognized dataset")
        # elif dataset_name=="folder":
        #     ref_path = os.path.join(dirs.DATA_HOME, "dataset_refs", ref_path)
        if ref_path is None:
            ref_path = os.path.join(dirs.DATA_HOME, "dataset_refs", dataset_name+".pkl")
        if dist.get_rank() == 0:
            ref_exists = os.path.isfile(ref_path)
        if not ref_exists: raise NotImplementedError
        if dist.get_rank() == 0:
            ref = calc.load_stats(path=ref_path) # do this first, just in case it fails

    # Reconstruct EMAs, if needed
    if dist.get_rank() == 0:
        for ema in emas:
            in_pkls = recp.list_input_pickles(in_dir=super_dir, in_prefix="network-snapshot")
            rec_iter = recp.reconstruct_phema(in_pkls=in_pkls, out_std=ema, out_epoch=out_epoch, 
                                              out_dir=super_dir, skip_existing=True)
            if rec_iter is not None:
                for _r in tqdm.tqdm(rec_iter, unit='step'): pass
    torch.distributed.barrier()

    # Get available checkpoints
    checkpoint_filenames = filter_by_string_must(os.listdir(super_dir), ".pkl")    
    checkpoint_filenames = filter_by_string_must(checkpoint_filenames, "0000000", must=False) # Skip randomly initialized networks
    if out_epoch is not None:
        checkpoint_epochs = [abs(find_numbers(f)[-2]) for f in checkpoint_filenames]
        checkpoint_filenames = [f for f, epoch in zip(checkpoint_filenames, checkpoint_epochs) if epoch==out_epoch]
    checkpoint_emas = [abs(find_numbers(f)[-1]) for f in checkpoint_filenames]
    checkpoint_filenames = [f for f, ema in zip(checkpoint_filenames, checkpoint_emas) if ema in emas]
    checkpoint_emas = [abs(find_numbers(f)[-1]) for f in checkpoint_filenames]
    
    # Configure sampler
    final_sampler_kwargs = EasyDict(DEFAULT_SAMPLER)
    for k in sampler_kwargs.keys(): final_sampler_kwargs[k] = sampler_kwargs[k]
    # final_sampler_kwargs.sampler_fn = "Images.generate_images.edm_full_sampler"

    # Gather all validation parameters
    if ref_path is not None: masked_ref = os.path.split(ref_path)[-1]
    else: masked_ref = None
    validation_kwargs={datetime.today().strftime('%Y-%m-%d %H:%M:%S'):dict(
                dataset_name=dataset_name, ref_path=masked_ref, guide_path=guide_path, 
                class_idx=class_idx, random_class=random_class, 
                seeds=seeds, **sampler_kwargs)}

    # Run
    results = {"Epoch":out_epoch}
    if fd_metrics:
        metrics = calc.parse_metric_list("fid,fd_dinov2")
        detectors = [calc.get_detector(metric, verbose=verbose) for metric in metrics]
    if class_metrics:
        classifier = valm.load_classifier_model("Swin", deterministic=False)
    for ema, filepath in zip(checkpoint_emas, checkpoint_filenames):
        ema_str = f"ema={ema:.3f}"
        results[ema_str] = {}

        for guidance_weight in guidance_weights:
            these_results = {}
            guidance_weight_str = f"guidance={guidance_weight:.3f}"
            checkpoint_filepath = os.path.join(super_dir, filepath)
            if verbose: dist.print0(f">>>>> Working on {ema_str} and {guidance_weight_str}")

            # Generate images
            if class_metrics: torch.use_deterministic_algorithms(False)
            if guidance_weight==1:
                temp_dir = os.path.join(super_dir, "gen_images", filepath.split(".pkl")[0])
            else:
                temp_dir = os.path.join(super_dir, "gen_images", filepath.split(".pkl")[0]+f"_{guidance_weight:.2f}")
            if not os.path.isdir(temp_dir) or len(os.listdir(temp_dir)) != len(seeds):
                generate_images(checkpoint_filepath, gnet=guide_path, outdir=temp_dir,
                                guidance=guidance_weight, class_idx=class_idx, random_class=random_class, 
                                seeds=seeds, verbose=verbose, device=device, **final_sampler_kwargs)
            
            # Load dataset
            dataset = valm.load_dataset(dataset_name="generated", image_path=temp_dir)
            
            # Calculate FID and FD-DINOv2 metrics for generated images
            if fd_metrics:
                stats_iter = calc.calculate_stats_for_dataset(dataset, metrics=metrics, detectors=detectors, device=device)
                r = calc.use_stats_iterator(stats_iter)
                if dist.get_rank() == 0:
                    initial_time = time.time()
                    fd_results = calc.calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics, verbose=verbose)
                    these_results.update(fd_results)
                    cumulative_time = time.time() - initial_time
                    dist.print0(f"Time to get FID/FD metrics = {cumulative_time:.2f} sec")
                torch.distributed.barrier()

            # Calculate classification metrics on the generated images using a pre-trained model 
            if class_metrics:
                torch.use_deterministic_algorithms(True)

                # Reconfigure the dataset to have the appropriate preprocessing
                transform_kwargs = valm.get_dataset_transform_kwargs("Swin", "generated")
                dataset = valm.set_up_dataset_transform(dataset, **transform_kwargs)

                # Get classification scores
                class_save_dir = valm.get_classification_metrics_dir("Swin", "generated", temp_dir)
                class_results = valm.get_classification_metrics(classifier, dataset, 
                                                                n_samples=None, batch_size=128, 
                                                                save_period=1, save_dir=class_save_dir, 
                                                                verbose=verbose)
                top_1_accuracy, top_5_accuracy, confusion_matrix, top_5_correct = class_results
                these_results.update(dict(top_1_accuracy=top_1_accuracy, top_5_accuracy=top_5_accuracy))

            # Delete temporary images
            if dist.get_rank()==0:
                if save_nimg==0:
                    shutil.rmtree(temp_dir)
                else:
                    contents = os.listdir(temp_dir)
                    contents = filter_by_string_must(contents, ["conf", "topf"], must=False)
                    contents.sort()
                    for i in range(save_nimg,len(contents)): os.remove(os.path.join(temp_dir, contents[i]))
            torch.distributed.barrier()
        
            # Save these results
            results[ema_str][guidance_weight_str] = dict(**these_results)
            if dist.get_rank()==0:
                with open(os.path.join(super_dir, metrics_filepath), 'wt') as f:
                    json.dump(results, f, indent=2)
            
            return results

    return results

@click.command()

@click.option('--emas', help='Chosen EMA length/s', metavar='LIST', type=str, required=True)
@click.option('--guidance-weights', help='Chosen EMA length/s', metavar='LIST', type=str, required=True)
@click.option('--super-dir', help='Path to raw checkpoint snapshots', metavar='DIR', type=str)
@click.option('--guide-path', 'guide_path', help='Guide model filepath', type=str, metavar='PATH')
@click.option('--out-epoch', 'out_epoch', help='Epoch of the snapshot to reconstruct', type=int, default=None)
@click.option('--fd-metrics/--no-fd-metrics', 'fd_metrics',  help='Calculate FID and FD-DINOv2 metrics?', type=bool, default=True, show_default=True)
@click.option('--class-metrics/--no-class-metrics', 'class_metrics',  help='Calculate classification metric?', type=bool, default=True, show_default=True)
@click.option('--out-filepath', 'metrics_filepath', help='Guide model filepath', type=str, metavar='PATH')
@click.option('--verbose/--no-verbose', 'verbose',  help='Show prints?', metavar='BOOL', type=bool, default=True, show_default=True)

def cmdline(emas, guidance_weights, super_dir, guide_path, out_epoch, 
            fd_metrics, class_metrics, metrics_filepath, verbose):

    if emas is None or guidance_weights is None: raise ValueError("Missing grid search parameters")
    emas = [float(ema) for ema in emas.split(",")]
    guidance_weights = [float(gw) for gw in guidance_weights.split(",")]
    super_dir = os.path.join(dirs.MODELS_HOME, "Images", super_dir)
    guide_path = os.path.join(dirs.MODELS_HOME, "Images", guide_path)
    if metrics_filepath is not None:
        metrics_filepath = os.path.join(dirs.MODELS_HOME, "Images", metrics_filepath)

    calculate_metrics_for_grid_search(emas, guidance_weights,
            super_dir, guide_path, out_epoch=out_epoch,
            fd_metrics=fd_metrics, class_metrics=class_metrics, save_nimg=200,
            metrics_filepath=metrics_filepath, verbose=verbose)

if __name__ == "__main__":
    cmdline()