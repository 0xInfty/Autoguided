import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ours"))

import csv
import numpy as np
import torch
import wandb
import click
import tqdm
import matplotlib.pyplot as plt

import karras.dnnlib as dnnlib
from ours.dataset import DATASET_OPTIONS

api = wandb.Api()

def download_metrics(run_ids, output_filepath, max_epochs=None, page_size=100):
    wait_for_trigger = max_epochs is not None

    # Download data
    history = []
    for run_id in run_ids:
        run = api.run(f"ajest/Images/{run_id}")
        hist = run.scan_history(keys=["Epoch", "Indices", "Selected indices"], page_size=100)
        history.append(hist)

    # Save data to CSV file
    with open(output_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Rank", "Epoch", "Round", "Image ID", "Selected"])
        
        # Write one image index on each row
        for epoch_i, rank_rows in enumerate(tqdm.tqdm(zip(*history))):
            if wait_for_trigger and epoch_i >= max_epochs: break
            for round_i in range(len(rank_rows[0]["Indices"])):
                for image_i in range(len(rank_rows[0]["Indices"][round_i])):
                    for rank_i in range(2):
                        img_id = rank_rows[rank_i]["Indices"][round_i][image_i]
                        is_img_id_selected = image_i in rank_rows[rank_i]["Selected indices"][round_i]
                        writer.writerow([rank_i, rank_rows[rank_i]["Epoch"], round_i, img_id, int(is_img_id_selected)])

A4_DIMS = [11.7, 8.3] # H,W in inches; 2480x3508 pixels at 300 dpi

def visualize_images(dataset, img_ids, are_ids_selected=None):

    # Get parameters for figure
    n_cols = 32
    n_rows = int(len(img_ids)/n_cols)
    plot_all = are_ids_selected is None
    if plot_all: are_ids_selected = [True for img_id in img_ids]

    # Create landscape figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=A4_DIMS, facecolor="black")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot images
    for idx, (img_id, is_selected) in enumerate(zip(img_ids, are_ids_selected)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        ax.set_facecolor("black")
        if plot_all or is_selected:
            _, img, _ = dataset[img_id]
            img = img.to(torch.float32) / 255
            ax.imshow(img.numpy().swapaxes(0,1).swapaxes(1,2))
        ax.axis('off')

    # Final layout and export
    plt.tight_layout(pad=0)
    plt.show()

    return fig, axes

def visualize_images_per_iteration(dataset, img_ids, are_ids_selected, N_iterations=16):

    # Get parameters for figure
    n_rows = N_iterations
    n_cols = int(sum(are_ids_selected)/n_rows)
    plot_all = are_ids_selected is None
    if plot_all: are_ids_selected = [True for img_id in img_ids]

    fig_size = A4_DIMS[::-1] # Portrait
    padding = 0.2 # 0.5 cm in inches
    cell_size = min((np.array(fig_size)-2*padding)/np.array([n_rows, n_cols]))
    fig_size = 2*padding+cell_size*np.array((n_rows, n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, facecolor="black", figsize=fig_size[::-1])

    # Plot images
    selected_img_ids = img_ids[are_ids_selected]
    for idx, img_id in enumerate(selected_img_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        ax.set_facecolor("black")
        _, img, _ = dataset[img_id]
        img = img.to(torch.float32) / 255
        ax.imshow(img.numpy().swapaxes(0,1).swapaxes(1,2))
        ax.axis('off')

    # Final layout and export
    plt.show()

    return fig, axes

def visualize_all_selected_images(dataset_name, indices_filepath, ajest_N=None,
                                  comparison=True, per_iteration=True, max_epochs=None):

    # General configuration
    folder = os.path.dirname(indices_filepath)
    if per_iteration and ajest_N is None:
        raise ValueError("Unknown number of data selection iterations")

    # Load indices data
    indices_data = np.loadtxt(indices_filepath, skiprows=1,delimiter=",",dtype=np.int32)
    n_epochs = int(max(indices_data[:,1])+1)
    n_rounds = int(max(indices_data[:,2])+1)

    # Load dataset
    if dataset_name == "imagenet":
        # dataset_kwargs = dnnlib.EasyDict(class_name=DATASET_OPTIONS[dataset_name]["class_name"], path=opts.data)
        raise ValueError("Need data path")
    else:
        dataset_kwargs = dnnlib.EasyDict(DATASET_OPTIONS[dataset_name])
    dataset_kwargs.use_labels = True
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    # Generate visualizations and store them
    if max_epochs is not None: n_epochs = max_epochs
    for epoch_i in tqdm.tqdm(range(n_epochs)):
        for round_i in range(n_rounds):

            # Get round data
            round_indices = np.where(np.logical_and(indices_data[:,1]==epoch_i, 
                                        indices_data[:,2]==round_i))[0]
            img_ids = indices_data[round_indices,-2]
            are_img_ids_selected = indices_data[round_indices,-1].astype(bool)

            # Visualize all images in the round, and then show only those selected
            if comparison:
                fig, axes = visualize_images(dataset, img_ids)
                fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_all.jpeg")
                fig.savefig(fig_filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                fig, axes = visualize_images(dataset, img_ids, are_img_ids_selected)
                fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_compared.jpeg")
                fig.savefig(fig_filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # Visualize images in the round in the order in which they were selected
            if per_iteration:
                fig_filepath = os.path.join(folder, f"epoch_{epoch_i}__round_{round_i}_selected.jpeg")
                fig, axes = visualize_images_per_iteration(dataset, img_ids, are_img_ids_selected, ajest_N)
                fig.savefig(fig_filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''

@cmdline.command()
@click.option('--runs',       help='W&B run IDs, as many as GPUs used', metavar='STR',  type=str, multiple=True, required=True)
@click.option('--dest',       help='Relative path to output CSV file', metavar='PATH',   type=str, required=True)
@click.option('--maxn',       help='Maximum number of epochs to store', metavar='INT', type=int, required=False, default=None, show_default=True)

def download(runs, dest, maxn):
    dest = os.path.join(dirs.MODELS_HOME, dest)
    download_metrics(runs, dest, max_epochs=maxn)

@cmdline.command()
@click.option('--source',     help='Relative path to input CSV file', metavar='STR', type=str, required=True)
@click.option('--dataset',    help='Dataset to be used', metavar='STR', type=click.Choice(list(DATASET_OPTIONS.keys())), default="imagenet", show_default=True)
@click.option('--seln',       help='Number of data selection iterations per round', metavar='INT', type=int, required=False, default=None, show_default=True)
@click.option('--maxn',       help='Maximum number of epochs to store', metavar='INT', type=int, required=False, default=None, show_default=True)
@click.option('--compare/--no-compare',    help='Show comparison between selected and not selected', metavar='BOOL', type=bool, default=True)
@click.option('--per-iter/--no-per-iter',  help='Show selected images per iteration', metavar='BOOL', type=bool, default=True)

def visualize(source, dataset, seln, maxn, compare, per_iter):
    source = os.path.join(dirs.MODELS_HOME, source)
    visualize_all_selected_images(dataset, source, ajest_N=seln, 
                                  comparison=compare, per_iteration=per_iter, max_epochs=maxn)

if __name__ == "__main__":
    cmdline()

    # run_ids = ["ionsbhmm", "4esqrkm8"] # Each group has two runs, one per GPU
    # filepath = os.path.join(dirs.MODELS_HOME, "Images/03_TinyImageNet/EarlyAJEST/00", "indices/indices.csv")
    # download_metrics(run_ids, filepath)#, max_epochs=2, page_size=50)