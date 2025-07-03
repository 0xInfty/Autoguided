import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ToyExample"))

import shutil
import numpy as np
import torch
from re import finditer

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

### Tools for Weights & Biases ###################################################################

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