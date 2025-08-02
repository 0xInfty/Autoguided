import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))

import shutil
import tqdm
import click
import time
import math
import numpy as np
import torch
import wandb
import timm
from pyvtools.text import filter_by_string_must, find_numbers

import karras.torch_utils.distributed as dist
from karras.dnnlib.util import EasyDict, construct_class_by_name
from generate_images import DEFAULT_SAMPLER, generate_images, parse_int_list
import calculate_metrics as calc
from ours.dataset import DATASET_OPTIONS
from ours.utils import get_wandb_id, upsample

#----------------------------------------------------------------------------
# Calculate metrics for all stored models as a post-hoc validation curve

def get_classification_metrics(
        dataset_name="tiny",
        n_samples=None,
        batch_size=128,
        shuffle=False,
        verbose=False,
):
    
    # Load dataset
    if dataset_name != "tiny": raise NotImplementedError("No classification model available")
    dataset_kwargs = calc.get_dataset_kwargs(dataset_name)
    dataset_obj = construct_class_by_name(**dataset_kwargs, random_seed=0)
    n_classes = dataset_obj.n_classes
    n_examples = len(dataset_obj)
    if n_samples is None: n_samples = n_examples
    n_samples = min(n_samples, n_examples)

    # Set data loader
    batch_size = min(batch_size, n_samples)
    n_batches = int(math.ceil( n_samples / batch_size ))
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size, shuffle=shuffle,
                                              num_workers=2, pin_memory=True, prefetch_factor=2)

    # Create Swin-L model
    model = timm.create_model('swin_large_patch4_window12_384', pretrained=False, drop_path_rate=0.1).cpu()
    for param in model.parameters():
        param.requires_grad = False
    model.reset_classifier(num_classes=200)

    # Load pre-trained weights (fine-tuned on Tiny ImageNet)
    checkpoint = torch.load("/mnt/hdd/vale/models/SCID/Images/00_PreTrained/swin_large_384.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Top-5 accuracy per class
    top_5_correct = np.zeros(n_classes, dtype=np.uint32)

    # Confusion matrix --> Can be used to get Top-1 accuracy per class
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.uint32)
    # i-th row and j-th column indicate the number of samples with
    # true label being i-th class and
    # predicted label being j-th class

    # Get stats
    for idx, (_, images, onehots) in tqdm.tqdm(enumerate(data_loader), total=n_batches):
        if idx>=n_batches: break
        labels = onehots.argmax(axis=1)
        upsampled = torch.stack([upsample(384, image) for image in images])
        predictions = model(upsampled)
        predicted_labels = predictions.argmax(axis=1)
        confusion_matrix[labels, predicted_labels] += 1
        top_5_labels = predictions.argsort(axis=1)[:,-5:]
        for gt, top_5 in zip(labels, top_5_labels):
            top_5_correct[gt] += gt in top_5
    if verbose:
        print("Top-1 Accuracy", float( confusion_matrix.diagonal().sum() / n_samples ))
        print("Top-5 Accuracy", float( top_5_correct.sum() / n_samples ))
    
    return confusion_matrix, top_5_correct

#----------------------------------------------------------------------------
# Calculate metrics for all stored models as a post-hoc validation curve

def calculate_metrics_for_checkpoints(
        checkpoints_dir,
        dataset_name = "tiny",          # Dataset used for training.
        ref_path = None,                # Filepath to dataset reference metrics.
        guide_path = None,              # Filepath to guide model.
        guidance_weight = 1,            # Guidance weight. Default = 1 (no guidance).
        class_idx = None,               # Class label. None = use automatic selection.
        random_class = False,           # Automatic selection can be uniformly random or forced exact distribution.
        seeds = range(0, int(2e3)),     # List of random seeds.
        chosen_emas = None,             # List of chosen EMAs. Default: use all.
        min_epoch = None,               # Number of epochs to start processing from.
        max_epoch = None,               # Number of epochs to stop processing at.
        save_nimg = 0,                  # How many images to keep, the rest will be deleted.
        verbose = True,                 # Enable status prints?
        log_to_wandb = True,            # Log to W&B?
        device = torch.device("cuda"),  # Which compute device to use.
        **sampler_kwargs
):

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

    # Get available checkpoints
    checkpoint_filenames = filter_by_string_must(os.listdir(checkpoints_dir), ".pkl")    
    checkpoint_filenames = filter_by_string_must(checkpoint_filenames, "0000000", must=False) # Skip randomly initialized networks
    checkpoint_emas = [abs(find_numbers(f)[-1]) for f in checkpoint_filenames]
    if min_epoch is not None or max_epoch is not None:
        checkpoint_filenames = np.array(checkpoint_filenames)
        checkpoint_epochs = np.array([abs(find_numbers(f)[0]) for f in checkpoint_filenames], dtype=np.int32)
        if min_epoch is not None:
            checkpoint_filenames = checkpoint_filenames[checkpoint_epochs >= min_epoch]
            checkpoint_epochs = np.array([abs(find_numbers(f)[0]) for f in checkpoint_filenames], dtype=np.int32)
        if max_epoch is not None:
            checkpoint_filenames = checkpoint_filenames[checkpoint_epochs <= max_epoch]
        checkpoint_filenames = list(checkpoint_filenames)
    
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

    # Configure sampler
    final_sampler_kwargs = EasyDict(DEFAULT_SAMPLER)
    for k in sampler_kwargs.keys(): final_sampler_kwargs[k] = sampler_kwargs[k]
    # final_sampler_kwargs.sampler_fn = "Images.generate_images.edm_full_sampler"

    # Configure distributed execution and resume W&B logging
    dist.init()
    if log_to_wandb:
        wandb_id = get_wandb_id(checkpoints_dir)
        run = wandb.init(entity="ajest", project="Images", id=wandb_id, resume="allow",
            config=dict(validation_kwargs=dict(dataset_name=dataset_name, ref_path=os.path.split(ref_path)[-1], 
                                            guide_path=guide_path, guidance_weight=guidance_weight,
                                            class_idx=class_idx, random_class=random_class, 
                                            seeds=seeds, chosen_emas=chosen_emas)),
            settings=wandb.Settings(x_disable_stats=True))

    # For each EMA in chosen EMAs, and for each checkpoint inside the directory
    metrics = calc.parse_metric_list("fid,fd_dinov2")
    detectors = [calc.get_detector(metric, verbose=verbose) for metric in metrics]
    for i, checkpoint_filenames in enumerate(checkpoint_filenames_by_ema):
        if guidance_weight!=1 and guide_path is not None:
            tag = f" [EMA={ema:.3f}, Guidance={guidance_weight:.2f}]"
        else:
            tag = f" [EMA={ema:.3f}]"
        for checkpoint_filename in checkpoint_filenames:
            checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_filename)
            checkpoint_epochs = abs(find_numbers(checkpoint_filename)[-2])
            if verbose: dist.print0(f">>>>> Working on EMA {ema:.3f} and Epoch {checkpoint_epochs}")

            # Generate images
            if guidance_weight==1:
                temp_dir = os.path.join(checkpoints_dir, "gen_images", checkpoint_filename.split(".pkl")[0])
            else:
                temp_dir = os.path.join(checkpoints_dir, "gen_images", checkpoint_filename.split(".pkl")[0]+f"_{guidance_weight:.2f}")
            image_iter = generate_images(checkpoint_filepath, gnet=guide_path, outdir=temp_dir,
                                         guidance=guidance_weight, class_idx=class_idx, random_class=random_class, 
                                         seeds=seeds, verbose=verbose, device=device, **final_sampler_kwargs)
            for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)): pass
            
            # Calculate metrics for generated images
            stats_iter = calc.calculate_stats_for_files(dataset_name="folder", image_path=temp_dir,
                                                        metrics=metrics, detectors=detectors, device=device)
            for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)): pass
            if dist.get_rank() == 0:
                initial_time = time.time()
                results = calc.calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics, verbose=verbose)
                if log_to_wandb:
                    run.log(dict({"Validation Epoch": checkpoint_epochs, 
                                f"Validation FID"+tag: results["fid"],
                                f"Validation FD-DINOv2"+tag: results["fd_dinov2"]}))
                cumulative_time = time.time() - initial_time
                dist.print0(f"Time to get metrics = {cumulative_time:.2f} sec")
            torch.distributed.barrier()

            # Delete temporary images
            if dist.get_rank()==0:
                if save_nimg==0:
                    shutil.rmtree(temp_dir)
                else:
                    contents = os.listdir(temp_dir)
                    contents.sort()
                    for i in range(save_nimg,len(contents)): os.remove(os.path.join(temp_dir, contents[i]))
            torch.distributed.barrier()

    if log_to_wandb:
        torch.distributed.barrier()
        try: run.finish()
        except AttributeError: pass
        torch.distributed.barrier()

@click.command()
@click.option("--models-dir", "models_dir", help="Relative path to directory containing the model checkpoints", type=str, metavar='PATH', required=True)
@click.option('--dataset', 'dataset_name', help='Dataset to be used', type=click.Choice(list(DATASET_OPTIONS.keys())), default="tiny", show_default=True)
@click.option('--ref', 'ref_path', help='Dataset reference statistics ', type=str, metavar='PATH', required=False, default=None, show_default=True)
@click.option('--guide-path', 'guide_path', help='Guide model filepath', type=str, metavar='PATH', default=None, show_default=True)
@click.option('--guidance-weight', 'guidance_weight', help='Guidance strength: default is 1 (no guidance)', type=float, default=1.0, show_default=True)
@click.option('--random/--no-random', 'random_class',  help='Use random classes?', type=bool, default=False, show_default=True)
@click.option('--emas', help='Chosen EMA length/s', required=False, multiple=True, default=None, show_default=True)
@click.option('--min-epoch', help='Number of batches at which to start', type=int, required=False, default=None, show_default=True)
@click.option('--max-epoch', help='Number of batches at which to stop', type=int, required=False, default=None, show_default=True)
@click.option('--save-nimg', help='Number of generated images to keep', type=int, required=False, default=0, show_default=True)
@click.option('--seeds', help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST', type=parse_int_list, default='0-1999', show_default=True)
@click.option('--wandb/--no-wandb', 'log_to_wandb',  help='Log to W&B?', type=bool, default=True, show_default=True)
def get_validation_metrics(models_dir, dataset_name, ref_path, guide_path, guidance_weight, random_class, emas, min_epoch, max_epoch, seeds, save_nimg, log_to_wandb):
    models_dir = os.path.join(dirs.MODELS_HOME, "Images", models_dir)
    if ref_path is not None: ref_path = os.path.join(dirs.DATA_HOME, "dataset_refs", ref_path)
    if guide_path is not None: guide_path = os.path.join(dirs.MODELS_HOME, "Images", guide_path)
    if len(emas)==0: emas=None
    else: emas = [float(ema) for ema in emas]
    calculate_metrics_for_checkpoints(models_dir,
        dataset_name=dataset_name, ref_path=ref_path,
        guide_path=guide_path, guidance_weight=guidance_weight,
        random_class=random_class, chosen_emas=emas, 
        min_epoch=min_epoch, max_epoch=max_epoch, seeds=seeds, 
        save_nimg=save_nimg, log_to_wandb=log_to_wandb)

if __name__ == "__main__":
    get_validation_metrics()