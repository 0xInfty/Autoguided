import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ToyExample"))

import torch
DEVICE = torch.device("cuda")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import ToyExample.toy_example as toy
import mandala_exploration.fractal_step_by_step as mand

### Grid creation ##############################################################################

def create_grid_samples(grid_resolution, 
                        x_centre=toy.FIG1_KWARGS["view_x"], y_centre=toy.FIG1_KWARGS["view_y"], 
                        x_side=2*toy.FIG1_KWARGS["view_size"], y_side=2*toy.FIG1_KWARGS["view_size"],
                        device=DEVICE):
    grid_x = torch.linspace(x_centre - x_side/2, x_centre + x_side/2, 
                            grid_resolution, device=device)
    grid_y = torch.linspace(y_centre - y_side/2, y_centre + y_side/2, 
                            grid_resolution, device=device)
    samples_x, samples_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
    return torch.stack([samples_x, samples_y]).swapaxes(0,2) # (X_Index, Y_Index, X_Y)

def get_grid_params(grid_resolution, 
                    x_centre=toy.FIG1_KWARGS["view_x"], y_centre=toy.FIG1_KWARGS["view_y"], 
                    x_side=2*toy.FIG1_KWARGS["view_size"], y_side=2*toy.FIG1_KWARGS["view_size"]):

    x_cell_size = x_side / (grid_resolution-1)
    y_cell_size = y_side / (grid_resolution-1)

    x_bounds = (x_centre - x_side/2 - x_cell_size/2, x_centre + x_side/2 + x_cell_size/2)
    y_bounds = (y_centre - y_side/2 - y_cell_size/2, y_centre + y_side/2 + y_cell_size/2)

    return (x_cell_size, y_cell_size), (x_bounds, y_bounds)

### Numeric integration ########################################################################

def get_simpson_params(grid_resolution, 
                       x_side=2*toy.FIG1_KWARGS["view_size"], y_side=2*toy.FIG1_KWARGS["view_size"],
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
                      x_centre=toy.FIG1_KWARGS["view_x"], y_centre=toy.FIG1_KWARGS["view_y"], 
                      x_side=2*toy.FIG1_KWARGS["view_size"], y_side=2*toy.FIG1_KWARGS["view_size"]):
    
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

    return logp >= toy.GT_LOGP_LEVEL

### Mandala metric ###############################################################################

def mandala_score(model, ground_truth_dist, guide=None, guidance_weight=1,
                  samples=None, n_samples=2**14, sigma_max=5,
                  grid_resolution=101, 
                  x_centre=toy.GT_ORIGIN[0], y_centre=toy.GT_ORIGIN[1], 
                  x_side=2*1.5, y_side=2*1.5,
                  logging=False, plotting=True, 
                  full_scale=True, log_scale=False, device=DEVICE):

    # If no samples provided, generate samples
    if samples is None:
        if isinstance(model, toy.GaussianMixture):
            samples = ground_truth_dist.sample(n_samples, sigma=0)
        else:
            gt_samples = ground_truth_dist.sample(n_samples, sigma=sigma_max)
            samples = toy.do_sample(net=model, gnet=guide, guidance=guidance_weight,
                                        x_init=gt_samples, sigma_max=sigma_max)[-1]
    else:
        n_samples = len(samples)

    # Create grid
    grid_coords = create_grid_samples(grid_resolution, x_centre, y_centre, x_side, y_side, device=device)
    cell_size, bounds = get_grid_params(grid_resolution, x_centre, y_centre, x_side, y_side)
    cell_size = cell_size[0] # Squared symmetric grid ==> Retain just one
    x_edges = np.linspace(*bounds[0], grid_resolution+1)
    y_edges = np.linspace(*bounds[1], grid_resolution+1)

    # Discard samples that fell off the grid
    samples = samples[samples[:,0]>=bounds[0][0]]
    samples = samples[samples[:,0]<=bounds[0][1]]
    samples = samples[samples[:,1]>=bounds[1][0]]
    samples = samples[samples[:,1]<=bounds[1][1]]
    n_in_grid = len(samples)

    # Count hits and misses
    hit_samples_mask = is_sample_in_fractal(samples, ground_truth_dist)
    n_hits = int(torch.sum(hit_samples_mask))
    n_miss = n_in_grid - n_hits
    if logging: print("Total in grid", n_in_grid, "\n> Hits", n_hits, "\n> Misses", n_miss)
    hit_samples = samples[hit_samples_mask]
    miss_samples = samples[torch.logical_not(hit_samples_mask)]

    # Calculate non-unique metric
    non_unique_score = n_hits / n_samples
    if logging: print("Initial score", non_unique_score)

    # Get grid coordinates for hits and misses
    hit_cells = np.array( mand.point_to_cell(hit_samples, 
                                             bounds[0][0], bounds[1][0], 
                                             cell_size, xy_order=True ), dtype=np.int32 ).T
    miss_cells = np.array( mand.point_to_cell(miss_samples, 
                                              bounds[0][0], bounds[1][0], 
                                              cell_size, xy_order=True ), dtype=np.int32 ).T

    # Register hits and missess in a plottable grid
    grid = np.zeros((grid_resolution, grid_resolution))
    for cell in hit_cells: grid[*cell] += 1
    for cell in miss_cells: grid[*cell] += 1

    # Get unique hits and misses
    hit_unique = set([(x,y) for x,y in hit_cells])
    miss_unique = set([(x,y) for x,y in miss_cells if not (x,y) in hit_unique])
    n_hits_unique = len(hit_unique)
    n_miss_unique = len(miss_unique)
    n_unique = n_hits_unique + n_miss_unique
    if logging: print("Unique total in grid", n_unique, "\n> Unique hits", n_hits_unique, 
                      "\n> Unique misses", n_miss_unique)

    # Calculate metric    
    unique_score = n_hits_unique / n_unique
    if logging: print("Mandala score", unique_score)

    # Optional plotting step
    if plotting:
        plot_mandala_score(samples, grid, 
            grid_coords, x_edges, y_edges, bounds,
            x_centre, y_centre, x_side, y_side, 
            n_hits_unique, n_miss_unique, 
            ground_truth_dist,
            full_scale=full_scale, log_scale=log_scale)

    return unique_score, non_unique_score

### Visualization utils ########################################################################

def plot_mandala_score(samples, grid, 
                       grid_coords, x_edges, y_edges, bounds,
                       x_centre, y_centre, x_side, y_side, 
                       n_hits_unique, n_miss_unique, 
                       ground_truth_dist,
                       full_scale=True, log_scale=False):
       
    # Create figure with square aspect ratio
    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(10.2, 5), dpi=300, 
                                        gridspec_kw={'width_ratios': [5, 5, .2]})

    # 1. Draw fractal with scatter
    ax1.set_title("Fractal Tree with Scatter")
    ax1.set_aspect("equal")

    # Draw fractal
    toy.do_plot(ground_truth_dist, elems={'gt_uncond', 'gt_outline'},
                view_x=x_centre, view_y=y_centre, view_size=x_side/2, ax=ax1)

    # Plot scatter points
    ax1.scatter(*samples.swapaxes(0,1).detach().cpu().numpy(), color='k', alpha=0.1, s=10, zorder=10)

    # 2. Hit/Miss Visualization
    ax2.set_title("Hit/Miss Analysis")
    ax2.set_aspect("equal")

    if not full_scale:
        grid[grid>1]=2 # Set every n>=2 to 2
    if log_scale:
        kwargs = dict(norm=LogNorm(vmin=1))
    else: 
        kwargs = dict(vmin=0)
    cmap = LinearSegmentedColormap.from_list('', ['white', *plt.cm.Reds(np.arange(255))])
    map = ax2.pcolormesh(*grid_coords.swapaxes(0,2).detach().cpu().numpy(), 
                          grid.T, cmap=cmap, **kwargs)
    toy.do_plot(ground_truth_dist, elems={'gt_uncond_thin', 'gt_outline_thin'},
                view_x=x_centre, view_y=y_centre, view_size=x_side/2, ax=ax2)
    plt.colorbar(map, ax=ax2, cax=ax3)

    # Add thin grid lines
    for x in x_edges:
        ax2.axvline(x=x, color='gray', alpha=0.2, linewidth=0.1)
    for y in y_edges:
        ax2.axhline(y=y, color='gray', alpha=0.2, linewidth=0.1)

    # Set bounds and remove axis numbers for all plots
    for ax in [ax1, ax2]:
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()

    # Add stats only to the last plot
    stats_text = f"Unique Total: {n_hits_unique+n_miss_unique}\nHits: {n_hits_unique}\nMisses: {n_miss_unique}"\
        f"\nMandala Score: {n_hits_unique/(n_hits_unique+n_miss_unique):.3f}"
    ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))