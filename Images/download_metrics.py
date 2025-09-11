import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ours"))

import csv
import wandb
import click
import tqdm
from itertools import chain
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import karras.dnnlib as dnnlib
from ours.dataset import DATASET_OPTIONS, get_dataset_kwargs
import ours.visualize as vis

import warnings
warnings.filterwarnings('ignore', 'No artists with labels found to put in legend')

api = wandb.Api()

sns.set_theme(style="darkgrid")

def download_training_metrics(run_ids, output_filepath, max_epochs=None, page_size=100, n_ranks=2, 
                              selection=False):
    wait_for_trigger = max_epochs is not None

    # Configuration
    metrics = ["Epoch", "Loss", "Seen images [kimg]", "Training time [sec]"]
    if selection:
        metrics += ["Epoch Super-Batch Learner Loss",
                    "Epoch Super-Batch Reference Loss",
                    "Selection time [sec]"]

    # Download data
    history = []
    for run_id in run_ids:
        run = api.run(f"ajest/Images/{run_id}")
        hist = run.scan_history(keys=metrics, page_size=page_size)
        history.append(hist)

    # Save data to CSV file
    with open(output_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Rank"] + metrics)
        
        # Write one image index on each row
        for epoch_i, rank_rows in enumerate(tqdm.tqdm(zip(*history))):
            if wait_for_trigger and epoch_i >= max_epochs: break
            print(rank_rows[0])
            for rank_i in range(n_ranks):
                writer.writerow([rank_i]+[rank_rows[rank_i][k] for k in metrics])

def download_selection_indices(run_ids, output_filepath, max_epochs=None, page_size=100, n_ranks=2):
    wait_for_trigger = max_epochs is not None

    # Download data
    history = []
    for run_id in run_ids:
        run = api.run(f"ajest/Images/{run_id}")
        hist = run.scan_history(keys=["Epoch", "Indices", "Selected indices"], page_size=page_size)
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
                    for rank_i in range(n_ranks):
                        img_id = rank_rows[rank_i]["Indices"][round_i][image_i]
                        is_img_id_selected = image_i in rank_rows[rank_i]["Selected indices"][round_i]
                        writer.writerow([rank_i, rank_rows[rank_i]["Epoch"], round_i, img_id, int(is_img_id_selected)])

def visualize_selection(dataset_name, indices_filepath, ajest_N=None,
                        per_round=False, per_iteration=False, 
                        show_images=True, show_labels=False, show_histograms=True, 
                        max_epochs=None, period=None, including=None):

    # General configuration
    folder = os.path.dirname(indices_filepath)
    if per_iteration and ajest_N is None:
        raise ValueError("Unknown number of data selection iterations")

    # Get basic parameters from the indices file
    n_ranks = 0
    n_rounds = 0
    with open(indices_filepath, "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i==0: pass # Header
            else:
                if int(line[0]) >= n_ranks:
                    n_ranks = int(line[0])
                if int(line[2]) >= n_rounds:
                    n_rounds = int(line[2])
                else: break
    n_ranks += 1
    n_rounds += 1
    n_lines_per_epoch = i-1
    n_lines_per_round = int(n_lines_per_epoch / n_rounds)
    limited_epochs = max_epochs is not None
    period = period or 1
    try: including = [int(i) for i in including[0].split(",")]
    except: including = None
    
    # Load dataset
    dataset_kwargs = get_dataset_kwargs(dataset_name)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    # Collect and visualize data
    epoch_i = 0
    with open(indices_filepath, "r") as f:
        reader = csv.reader(f)
        img_ids, are_img_ids_selected = [], []
        for i, line in tqdm.tqdm(enumerate(reader)):
            if i>0:
                epoch_i = int(line[1])
                round_i = int(line[2])
                if epoch_i % period == 0 or epoch_i in including:
                    try: 
                        img_ids[round_i].append(int(line[-2]))
                        are_img_ids_selected[round_i].append(bool(int(line[-1])))
                    except:
                        img_ids.append([int(line[-2])])
                        are_img_ids_selected.append([bool(int(line[-1]))])

                    if i % n_lines_per_round == 0:
                        
                        if i>0 and i % n_lines_per_round == 0:

                            # Visualize all images in the round, and then show only those selected
                            if per_round:
                                these_img_ids = img_ids[round_i]
                                are_these_img_ids_selected = are_img_ids_selected[round_i]
                                if show_images:
                                    fig, axes = vis.visualize_images(dataset, these_img_ids)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_images_all.jpeg")
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                    fig, axes = vis.visualize_images(dataset, these_img_ids, are_these_img_ids_selected)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_images.jpeg")
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                if show_labels:
                                    fig, axes = vis.visualize_classes(dataset, these_img_ids)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_labels_all.jpeg")
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                    fig, axes = vis.visualize_classes(dataset, these_img_ids, are_these_img_ids_selected)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_labels.jpeg")
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)

                            # Visualize images in the round in the order in which they were selected
                            if per_iteration:
                                these_img_ids = np.array(img_ids[round_i], dtype=np.int32)
                                are_these_img_ids_selected = np.array(are_img_ids_selected[round_i], dtype=bool)
                                if show_images:
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_rank_0_images.jpeg")
                                    fig, axes = vis.visualize_images_per_iteration(dataset, these_img_ids[0::2], 
                                                                                   are_these_img_ids_selected[0::2], ajest_N)
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_rank_1_images.jpeg")
                                    fig, axes = vis.visualize_images_per_iteration(dataset, these_img_ids[1::2], 
                                                                                   are_these_img_ids_selected[1::2], ajest_N)
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                if show_labels:
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_rank_0_labels.jpeg")
                                    fig, axes = vis.visualize_classes_per_iteration(dataset, these_img_ids[0::2], 
                                                                                    are_these_img_ids_selected[0::2], ajest_N)
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_rank_1_labels.jpeg")
                                    fig, axes = vis.visualize_classes_per_iteration(dataset, these_img_ids[1::2], 
                                                                                    are_these_img_ids_selected[1::2], ajest_N)
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                if show_histograms:
                                    these_img_ids = these_img_ids.reshape((ajest_N,-1))
                                    are_these_img_ids_selected = are_these_img_ids_selected.reshape((ajest_N,-1))
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_rank_0_histogram.jpeg")
                                    fig, axes = vis.plot_classes_histograms(dataset, these_img_ids[:,0::2], 
                                                                            are_these_img_ids_selected[:,0::2])
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_round_{round_i}_rank_1_histogram.jpeg")
                                    fig, axes = vis.plot_classes_histograms(dataset, these_img_ids[:,1::2], 
                                                                            vis.are_these_img_ids_selected[:,1::2])
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                        
                        if i % n_lines_per_epoch == 0:
                            these_img_ids = np.array(list(chain(*img_ids)), dtype=np.int32)
                            are_these_img_ids_selected = np.array(list(chain(*are_img_ids_selected)), dtype=bool)
                            if show_images:
                                fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_images.jpeg")
                                fig, axes = vis.visualize_images(dataset, these_img_ids[are_these_img_ids_selected], n_cols=24)
                                fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                            if show_labels:
                                fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_labels.jpeg")
                                fig, axes = vis.visualize_classes(dataset, these_img_ids[are_these_img_ids_selected], n_cols=24)
                                fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                            if show_histograms:
                                fig, axes = vis.plot_classes_histogram(dataset, these_img_ids, are_these_img_ids_selected)
                                fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_histogram.jpeg")
                                fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                                if per_round:
                                    fig, axes = vis.plot_classes_histograms(dataset, img_ids, are_img_ids_selected)
                                    fig_filepath = os.path.join(folder, f"epoch_{epoch_i}_histogram_per_round.jpeg")
                                    fig.savefig(fig_filepath, bbox_inches='tight', dpi=fig.dpi); plt.close(fig)
                            img_ids, are_img_ids_selected = [], []

            if limited_epochs and epoch_i >= max_epochs: break

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''

@cmdline.command()
@click.option('--runs',       help='W&B run IDs, as many as GPUs used', metavar='STR',  type=str, multiple=True, required=True)
@click.option('--dest',       help='Relative path to output CSV file', metavar='PATH',   type=str, required=True)
@click.option('--maxn',       help='Maximum number of epochs to store', metavar='INT', type=int, required=False, default=None, show_default=True)
@click.option('--selection/--no-selection',   help='Search for data selection loss?', metavar='BOOL', type=bool, default=False, show_default=True)

def download_loss(runs, dest, maxn, selection):
    dest = os.path.join(dirs.MODELS_HOME, dest)
    download_training_metrics(runs, dest, max_epochs=maxn, selection=selection)

@cmdline.command()
@click.option('--runs',       help='W&B run IDs, as many as GPUs used', metavar='STR',  type=str, multiple=True, required=True)
@click.option('--dest',       help='Relative path to output CSV file', metavar='PATH',   type=str, required=True)
@click.option('--maxn',       help='Maximum number of epochs to store', metavar='INT', type=int, required=False, default=None, show_default=True)

def download_indices(runs, dest, maxn):
    dest = os.path.join(dirs.MODELS_HOME, dest)
    download_selection_indices(runs, dest, max_epochs=maxn)

@cmdline.command()
@click.option('--source',     help='Relative path to input CSV file', metavar='STR', type=str, required=True)
@click.option('--dataset',    help='Dataset to be used', metavar='STR', type=click.Choice(list(DATASET_OPTIONS.keys())), default="imagenet", show_default=True)
@click.option('--seln',       help='Number of data selection iterations per round', metavar='INT', type=int, required=False, default=None, show_default=True)

@click.option('--maxn',       help='Maximum number of epochs to store', metavar='INT', type=int, required=False, default=None, show_default=True)
@click.option('--period',     help='Execute every period epochs', metavar='INT', type=int, required=False, default=None, show_default=True)
@click.option('--inc',        help='Epoch indices to be included regardless of period', metavar='INT', required=False, multiple=True, default=None, show_default=True)

@click.option('--images/--no-images',         help='Show images', metavar='BOOL', type=bool, default=True)
@click.option('--labels/--no-labels',         help='Show labels in the image grid', metavar='BOOL', type=bool, default=False)
@click.option('--histograms/--no-histograms', help='Show histograms of labels', metavar='BOOL', type=bool, default=True)

@click.option('--per-round/--no-per-round',   help='Show selected images per round', metavar='BOOL', type=bool, default=False)
@click.option('--per-iter/--no-per-iter',     help='Show selected images per iteration', metavar='BOOL', type=bool, default=False)

def visualize_indices(source, dataset, seln, maxn, period, inc, images, labels, histograms, per_round, per_iter):
    source = os.path.join(dirs.MODELS_HOME, source)
    visualize_selection(dataset, source, ajest_N=seln, 
                        show_images=images, show_labels=labels, show_histograms=histograms,
                        per_round=per_round, per_iteration=per_iter, 
                        max_epochs=maxn, period=period, including=inc)

if __name__ == "__main__":
    cmdline()

    # run_ids = ["ionsbhmm", "4esqrkm8"] # Each group has two runs, one per GPU
    # filepath = os.path.join(dirs.MODELS_HOME, "Images/03_TinyImageNet/EarlyAJEST/00", "indices/indices.csv")
    # download_metrics(run_ids, filepath)#, max_epochs=2, page_size=50)