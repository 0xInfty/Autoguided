import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ToyExample"))

import json
import shutil
import numpy as np
import torch
import PIL
from re import finditer

from pyvtools.text import filter_by_string_must

### Global parameters ##############################################################################

DEVICE = torch.device("cuda")

GT_ORIGIN = (0.0030, 0.0325)

FIG1_KWARGS = dict(view_x=0.30, view_y=0.30, view_size=1.2, num_samples=1<<14, device=DEVICE)
FIG2_KWARGS = dict(view_x=0.45, view_y=1.22, view_size=0.3, num_samples=1<<12, device=DEVICE, sample_distance=0.045, sigma_max=0.03)

GT_LOGP_LEVEL = -2.12

### Grid creation ##############################################################################

def create_grid_samples(grid_resolution, 
                        x_centre=FIG1_KWARGS["view_x"], y_centre=FIG1_KWARGS["view_y"], 
                        x_side=2*FIG1_KWARGS["view_size"], y_side=2*FIG1_KWARGS["view_size"],
                        device=DEVICE):
    grid_x = torch.linspace(x_centre - x_side/2, x_centre + x_side/2, 
                            grid_resolution, device=device)
    grid_y = torch.linspace(y_centre - y_side/2, y_centre + y_side/2, 
                            grid_resolution, device=device)
    samples_x, samples_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
    return torch.stack([samples_x, samples_y]).swapaxes(0,2) # (X_Index, Y_Index, X_Y)

def get_grid_params(grid_resolution, 
                    x_centre=FIG1_KWARGS["view_x"], y_centre=FIG1_KWARGS["view_y"], 
                    x_side=2*FIG1_KWARGS["view_size"], y_side=2*FIG1_KWARGS["view_size"]):

    x_cell_size = x_side / (grid_resolution-1)
    y_cell_size = y_side / (grid_resolution-1)

    x_bounds = (x_centre - x_side/2 - x_cell_size/2, x_centre + x_side/2 + x_cell_size/2)
    y_bounds = (y_centre - y_side/2 - y_cell_size/2, y_centre + y_side/2 + y_cell_size/2)

    return (x_cell_size, y_cell_size), (x_bounds, y_bounds)

### Numeric integration ########################################################################

def get_simpson_params(grid_resolution, 
                       x_side=2*FIG1_KWARGS["view_size"], y_side=2*FIG1_KWARGS["view_size"],
                       device=DEVICE):
    
    simpson_vector = torch.ones(grid_resolution).to(device)
    simpson_vector[1::2] = 4
    simpson_vector[2:-2:2] = 2
    simpson_matrix = torch.kron(simpson_vector, simpson_vector.reshape(1,grid_resolution)).reshape((grid_resolution, grid_resolution))

    delta_x = x_side / (grid_resolution-1)
    delta_y = y_side / (grid_resolution-1)
    simpson_scale = delta_x * delta_y / 9

    return simpson_matrix, simpson_scale

def integrate_simpson(f, grid_resolution, 
                      x_centre=FIG1_KWARGS["view_x"], y_centre=FIG1_KWARGS["view_y"], 
                      x_side=2*FIG1_KWARGS["view_size"], y_side=2*FIG1_KWARGS["view_size"]):
    
    # Create the grid
    samples = create_grid_samples(grid_resolution, x_centre, y_centre, x_side, y_side)

    # Evaluate the function on the grid
    f_samples = f(samples) # Assume you can pass a Torch tensor and get a Torch tensor

    # Get 2D composite Simpson parameters
    simpson_matrix, simpson_scale = get_simpson_params(grid_resolution, x_side, y_side)

    # Integrate
    return float(torch.sum(simpson_matrix * f_samples)) * simpson_scale

### Classification metric utils ##################################################################

def is_sample_in_fractal(samples, ground_truth_distribution, sigma=0):
    
    logp = ground_truth_distribution.logp(samples, sigma=sigma)

    return logp >= GT_LOGP_LEVEL

### Directories ##################################################################################

def get_training_params(checkpoints_dir):

    filepath = os.path.join(checkpoints_dir, "training_options.json")
    with open(filepath, "rb") as f:
        params = json.load(f)
    return params

def get_final_state(checkpoints_dir):

    contents = os.listdir(checkpoints_dir)
    contents = filter_by_string_must(contents, "training-state")
    contents.sort()
    filepath = os.path.join(checkpoints_dir, contents[-1])

    data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)

    return data["state"]

def get_nimg(n_epochs, batch_size, mini_batch_size=None, selection=False,
             early=False, late=False, change_epoch=None, change_nimg=None):
    
    if selection and mini_batch_size is None: 
        raise ValueError("Mini batch size required if data selection was used")

    if selection:
        if early:
            return int((n_epochs - change_epoch) * batch_size + change_nimg)
        elif late:
            return int((n_epochs - change_epoch) * mini_batch_size + change_nimg)
        else:
            return n_epochs * mini_batch_size
    else:
        return n_epochs * batch_size

### Tools for Weights & Biases ###################################################################

def get_wandb_ids(checkpoints_dir):
    """Identify the W&B run ID from the checkpoints folder"""
    try:
        contents = os.listdir(os.path.join(checkpoints_dir, "wandb"))
        wandb_folders = filter_by_string_must(contents, "run-")
        wandb_folders.sort()
        date_string = wandb_folders[0].split("-")[1].split("_")[0]
        wandb_folders = filter_by_string_must(wandb_folders, date_string)
    except FileNotFoundError:
        wandb_folders = []
    if len(wandb_folders) >= 1:
        wandb_logs_filepath = [os.path.join(checkpoints_dir, "wandb", f, "files", "output.log") for f in wandb_folders]
        wandb_logs_sizes = [os.path.getsize(f) for f in wandb_logs_filepath] # Rank 0 will log the most
        wandb_ids = [wandb_folders[i][-8:] for i in np.argsort(wandb_logs_sizes)[::-1]]
    else:
        wandb_ids = None
    return wandb_ids

def get_wandb_id(checkpoints_dir):
    """Identify the W&B run ID from the checkpoints folder"""
    wandb_ids = get_wandb_ids(checkpoints_dir)
    if wandb_ids is None: return None
    else: return wandb_ids[0]

def get_wandb_tags(dataset_kwargs):
    try:
        if "cifar" in dataset_kwargs.path:
            return ["cifar"]
        elif "tiny" in dataset_kwargs.path:
            return ["tiny"]
        else:
            return ["imagenet"]
    except:
        return None

def get_wandb_name(wandb_dir):
    first, last = os.path.split(wandb_dir)
    accumulated = last
    iteration = 0
    while True:
        first, new_last = os.path.split(first)
        if new_last not in ["ToyExample", "Images"]:
            accumulated = "_".join([new_last, accumulated])
            last = new_last
            iteration += 1
        else: break
        if iteration >= 3: raise RecursionError("W&B name could not be determined")
    return accumulated

def move_wandb_files(origin, destination):
    if os.path.isdir(os.path.join(origin, "wandb")):
        origin = os.path.join(origin, "wandb")
    if os.path.isfile(os.path.join(origin, "latest-run")): 
        os.remove(os.path.join(origin, "latest-run"))
    if os.path.isfile(os.path.join(origin, "debug.log")): 
        os.remove(os.path.join(origin, "debug.log"))
    if os.path.isfile(os.path.join(origin, "debug-internal.log")): 
        os.remove(os.path.join(origin, "debug-internal.log"))
    folders = [c for c in os.listdir(origin) if c.startswith("run-")]
    for folder in folders:
        if os.path.isdir(os.path.join(origin, folder, "tmp")): 
            shutil.rmtree(os.path.join(origin, folder, "tmp"))
        contents = os.listdir(os.path.join(origin, folder))
        for c in contents:
            os.rename(os.path.join(origin, folder, c), 
                      os.path.join(destination, c+"-"+folder.split("run-")[-1]))
        shutil.rmtree(os.path.join(origin, folder))
    shutil.rmtree(origin)
    return

### Basic utilities ##############################################################################

def split_camel_case(identifier):
    """https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python"""
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def get_stats(array):
    if isinstance(array, np.ndarray):
        return (float(array.min()), float(array.sum())/array.size, float(array.max()), array.shape)
    else:
        return (float(array.min()), float(array.sum())/array.numel(), float(array.max()), tuple(array.shape))

def check_nested_dict(old_dict, new_dict, is_alright=True, exception_keys=None, verbose=False):
    if exception_keys is None: exception_keys = []
    for k in old_dict.keys():
        if k in new_dict.keys() and k not in exception_keys:
            if isinstance(old_dict[k], dict):
                is_alright = check_nested_dict(old_dict[k], new_dict[k], is_alright, 
                                               exception_keys=exception_keys, verbose=verbose)
            else:
                if old_dict[k] != new_dict[k]: 
                    if verbose:
                        print("Detected problem with key", k)
                        print("> Old:", old_dict[k])
                        print("> New:", new_dict[k])
                    is_alright = False
    return is_alright

def find_all_indices(value, list_of_values):
    is_here = [v == value for v in list_of_values]
    return [int(i) for i in np.where(is_here)[0]]

### Image processing ##############################################################################

def upsample(image_size, image):

    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        device = image.device
        torch_dtype = image.dtype
        image = image.swapaxes(0,1).swapaxes(1,2).detach().cpu().numpy().astype(np.uint8)
    numpy_dtype = image.dtype
    pil_image = PIL.Image.fromarray(image)
    
    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    assert len(new_size) == 2
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)
    
    new_image = np.asarray(pil_image).astype(numpy_dtype).swapaxes(1,2).swapaxes(0,1)
    if is_torch:
        new_image = torch.Tensor(new_image, device=device).to(torch_dtype)

    return new_image
