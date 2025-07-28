import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))

import shutil
import tqdm
import click
import numpy as np
import torch
import wandb
from pyvtools.text import filter_by_string_must, find_numbers
import pyvtorch.aux as taux

import karras.torch_utils.distributed as dist
from generate_images import DEFAULT_SAMPLER, generate_images, parse_int_list
import calculate_metrics as calc
from ours.dataset import DATASET_OPTIONS

#----------------------------------------------------------------------------
# Utilities

def get_wandb_id(checkpoints_dir):
    """Identify the W&B run ID from the checkpoints folder"""
    contents = os.listdir(os.path.join(checkpoints_dir, "wandb"))
    wandb_folders = filter_by_string_must(contents, "run-")    
    if len(wandb_folders) > 1:
        wandb_logs_filepath = [os.path.join(checkpoints_dir, "wandb", f, "files", "output.log") for f in wandb_folders]
        wandb_logs_sizes = [os.path.getsize(f) for f in wandb_logs_filepath] # Rank 0 will log the most
        wandb_id = wandb_folders[np.argmax(wandb_logs_sizes)][-8:]
    else:
        wandb_id = wandb_folders[0][-8:]
    return wandb_id

#----------------------------------------------------------------------------
# Calculate metrics for all stored models as a post-hoc validation curve

def calculate_metrics_for_checkpoints(
        checkpoints_dir,
        dataset_name = "tiny",          # Dataset used for training.
        ref_path = None,                # Filepath to dataset reference metrics.
        guide_path = None,              # Filepath to guide model.
        guidance_weight = 1,            # Guidance weight. Default = 1 (no guidance).
        class_idx = None,               # Class label. None = select randomly.
        seeds = range(0, int(50e3)),    # List of random seeds.
        chosen_emas = None,             # List of chosen EMAs. Default: use all.
        verbose = True,                 # Enable status prints?
        device = torch.device("cuda"),  # Which compute device to use.
):
    
    # Identify the W&B run ID from the checkpoints folder
    wandb_id = get_wandb_id(checkpoints_dir)

    # Get available checkpoints
    checkpoint_filenames = filter_by_string_must(os.listdir(checkpoints_dir), ".pkl")    
    checkpoint_filenames = filter_by_string_must(checkpoint_filenames, "0000000", must=False) # Skip randomly initialized networks
    checkpoint_emas = [abs(find_numbers(f)[-1]) for f in checkpoint_filenames]
    
    # Separate by EMA and filter EMAs, if specified
    available_emas = list(set(checkpoint_emas))
    if chosen_emas is None:
        chosen_emas = available_emas
    else:
        if not isinstance(chosen_emas, list):
            chosen_emas = [chosen_emas]
        chosen_emas = [ema for ema in chosen_emas if ema in available_emas]
        if len(chosen_emas)==0: 
            raise ValueError("Specified EMA/s not available")
    checkpoint_filenames_by_ema = []
    for ema in chosen_emas:
        these_filenames = filter_by_string_must(checkpoint_filenames, f"{ema:.3f}")
        these_filenames.sort()
        checkpoint_filenames_by_ema.append(these_filenames)

    # Configure distributed execution and resume W&B logging
    dist.init()
    run = wandb.init(entity="ajest", project="Images", id=wandb_id, resume="allow",
        config=dict(validation_kwargs=dict(dataset_name=dataset_name, ref_path=ref_path, 
                                           guide_path=guide_path, guidance_weight=guidance_weight,
                                           class_idx=class_idx, seeds=seeds, chosen_emas=chosen_emas)),
        settings=wandb.Settings(x_disable_stats=True))

    # For each EMA in chosen EMAs, and for each checkpoint inside the directory
    metrics = calc.parse_metric_list("fid,fd_dinov2")
    for i, checkpoint_filenames in enumerate(checkpoint_filenames_by_ema):
        if guidance_weight!=1 and guide_path is not None:
            tag = f" (EMA={ema:.3f}, Guidance={guidance_weight:.2f})"
        else:
            tag = f" (EMA={ema:.3f})"
        for checkpoint_filename in checkpoint_filenames:

            # Generate 50k images
            checkpoint_epochs = abs(find_numbers(checkpoint_filename)[-2])
            checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_filename)
            temp_dir = os.path.join(checkpoints_dir, "temp_images")            
            image_iter = generate_images(checkpoint_filepath, gnet=guide_path, outdir=temp_dir,
                                         guidance=guidance_weight, class_idx=class_idx, seeds=seeds,
                                         verbose=verbose, device=device, **DEFAULT_SAMPLER)
            for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)): pass

            # Get reference stats
            assert dataset_name in DATASET_OPTIONS.keys(), "Unrecognized dataset"
            if dataset_name=="folder": raise NotImplementedError
            # if dataset_name=="folder" and ref_path is None: 
            #     raise ValueError("Reference path needed for unrecognized dataset")
            # elif dataset_name=="folder":
            #     ref_path = os.path.join(dirs.DATA_HOME, "dataset_refs", ref_path)
            else:
                ref_path = os.path.join(dirs.DATA_HOME, "dataset_refs", dataset_name+".pkl")
            if dist.get_rank() == 0:
                ref_exists = os.path.isfile(ref_path)
            if not ref_exists: raise NotImplementedError
            if dist.get_rank() == 0:
                ref = calc.load_stats(path=ref_path) # do this first, just in case it fails
            
            # Calculate metrics for generated images
            temp_file = os.path.join(checkpoints_dir, "temp_stats.json")
            stats_iter = calc.calculate_stats_for_files(dataset_name="folder", image_path=temp_dir,
                                                        metrics=metrics, dest_path=temp_file, device=device)
            for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)): pass
            if dist.get_rank() == 0:
                results = calc.calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics, verbose=verbose)
                run.log(dict({"Epoch": checkpoint_epochs, 
                              f"Validation FID"+tag: results["fid"],
                              f"Validation FD-DINOv2"+tag: results["fd_dinov2"]}))
            torch.distributed.barrier()

            # Delete temporary images
            shutil.rmtree(temp_dir)

@click.command()
@click.option("--models-dir", "models_dir", help="Relative path to directory containing the model checkpoints", metavar='PATH', required=True)
@click.option('--dataset', 'dataset_name', help='Dataset to be used', type=click.Choice(list(DATASET_OPTIONS.keys())), default="tiny", show_default=True)
@click.option('--ref', 'ref_path', help='Dataset reference statistics ', type=str, required=False, default=None, show_default=True)
@click.option('--guide-path', 'guide_path', help='Guide model filepath', metavar='PATH', type=str, default=None, show_default=True)
@click.option('--guidance-weight', 'guidance_weight', help='Guidance strength: default is 1 (no guidance)', metavar='PATH', type=float, default=1.0, show_default=True)
@click.option('--emas', help='Chosen EMA length/s', required=False, multiple=True, default=None, show_default=True)
@click.option('--seeds', help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST', type=parse_int_list, default='0-49999', show_default=True)
def get_validation_metrics(models_dir, dataset_name, ref_path, guide_path, guidance_weight, emas, seeds):
    models_dir = os.path.join(dirs.MODELS_HOME, "Images", models_dir)
    if len(emas)==0: emas=None
    calculate_metrics_for_checkpoints(models_dir,
        dataset_name=dataset_name, ref_path=ref_path,
        guide_path=guide_path, guidance_weight=guidance_weight,
        chosen_emas=emas, seeds=seeds)

if __name__ == "__main__":
    get_validation_metrics()
    # calculate_metrics_for_checkpoints("/mnt/hdd/vale/models/SCID/Images/04_Tiny_LR/AJEST/02",
    #                                   seeds=range(16,32),
    #                                   chosen_emas=0.1)
    # calculate_metrics_for_checkpoints("/mnt/hdd/vale/models/SCID/Images/04_Tiny_LR/AJEST/03",
    #                                   seeds=range(16,32))