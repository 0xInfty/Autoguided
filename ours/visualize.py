import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

A4_DIMS = [11.7, 8.3] # H,W in inches; 2480x3508 pixels at 300 dpi

def set_up_figure(n_rows=1, n_cols=1, portrait=True, facecolor="w", 
                  space=0, padding=0.2, dpi=100, aspect="auto"):

    assert aspect in ["equal", "auto"], "Aspect should be either 'equal' or 'auto'"

    if n_cols==1 and n_rows > 1:
        kwargs = dict( gridspec_kw=dict(hspace=space), sharey=True )
    elif n_rows == 1 and n_cols > 1:
        kwargs = dict( gridspec_kw=dict(wspace=space), sharex=True )
    else: kwargs = dict()

    if n_rows>4 or n_cols>4:
        if portrait: fig_size = A4_DIMS[::-1] # Portrait
        else: fig_size = A4_DIMS # Landscape

        cell_size = (np.array(fig_size)-2*padding)/np.array([n_rows, n_cols])
        if aspect == "equal":
            fig_size = 2*padding+min(cell_size)*np.array((n_rows, n_cols))
        else:
            fig_size = 2*padding+cell_size*np.array((n_rows, n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, facecolor=facecolor, figsize=fig_size[::-1], 
                                 squeeze=False, dpi=dpi, **kwargs)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, facecolor=facecolor, 
                                 squeeze=False, dpi=dpi, **kwargs)

    return fig, axes

def visualize_images(dataset, img_ids, are_ids_selected=None, n_cols=32, **kwargs):

    # Get parameters for figure
    n_rows = int(len(img_ids)/n_cols)
    plot_all = are_ids_selected is None
    if plot_all: are_ids_selected = [True]*len(img_ids)

    # Create landscape figure
    fig, axes = set_up_figure(n_rows, n_cols, portrait=False, facecolor="k", 
                              padding=0.5, aspect="equal", **kwargs)

    # Check image type
    _, img, _ = dataset[img_ids[0]]
    if isinstance(img, torch.Tensor): # Assume 0-255 (C,H,W) Torch tensor
        preprocess = lambda img : img.numpy().swapaxes(0,1).swapaxes(1,2) / 255
    elif img.shape[0] in [1,3]: # Assume 0-255 (C,H,W) Numpy array
        preprocess = lambda img : img.swapaxes(0,1).swapaxes(1,2) / 255
    else: # Assume 0-255 (H,W,C) Numpy array
        preprocess = lambda img : img / 255

    # Plot images
    for idx, (img_id, is_selected) in enumerate(zip(img_ids, are_ids_selected)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        ax.set_facecolor("black")
        if plot_all or is_selected:
            _, img, _ = dataset[img_id]
            img = preprocess(img)
            ax.imshow(img)
        ax.axis('off')

    # Final layout and export
    plt.show()

    return fig, axes

# Per AJEST iteration, but can be generalized to anything
def visualize_images_per_iteration(dataset, img_ids, are_ids_selected, N_iterations=16):

    # Get parameters for figure
    n_rows = N_iterations
    n_cols = int(sum(are_ids_selected)/n_rows)
    plot_all = are_ids_selected is None
    if plot_all: are_ids_selected = [True]*len(img_ids)

    # Create figure
    fig, axes = set_up_figure(n_rows, n_cols, portrait=True, facecolor="black", aspect="equal")

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

def visualize_classes(dataset, img_ids, are_ids_selected=None, n_cols=32, **kwargs):

    # Get parameters for figure
    n_rows = int(len(img_ids)/n_cols)
    plot_all = are_ids_selected is None
    if plot_all: are_ids_selected = [True]*len(img_ids)

    # Create landscape figure
    fig, axes = set_up_figure(n_rows, n_cols, portrait=False, facecolor="k", 
                              padding=0.5, aspect="equal", **kwargs)

    # Plot images
    for idx, (img_id, is_selected) in enumerate(zip(img_ids, are_ids_selected)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        ax.set_facecolor("black")
        if plot_all or is_selected:
            _, _, label = dataset[img_id]
            plt.text(0.5, 0.5, str(list(label).index(1)), fontdict=dict(fontsize="xx-small"),
                    horizontalalignment='center', verticalalignment='center',
                    color="w", transform = ax.transAxes)
        ax.axis('off')

    # Final layout and export
    plt.show()

    return fig, axes

def visualize_classes_per_iteration(dataset, img_ids, are_ids_selected, N_iterations=16):

    # Get parameters for figure
    n_rows = N_iterations
    n_cols = int(sum(are_ids_selected)/n_rows)
    plot_all = are_ids_selected is None
    if plot_all: are_ids_selected = [True]*len(img_ids)

    # Create figure
    fig, axes = set_up_figure(n_rows, n_cols, portrait=True, facecolor="black", aspect="equal")

    # Plot images
    selected_img_ids = img_ids[are_ids_selected]
    for idx, img_id in enumerate(selected_img_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        ax.set_facecolor("black")
        _, _, label = dataset[img_id]
        plt.text(0.5, 0.5, str(list(label).index(1)),
                horizontalalignment='center', verticalalignment='center',
                color="w", transform = ax.transAxes)
        ax.axis('off')

    # Final layout and export
    plt.show()

    return fig, axes

def plot_classes_histogram(dataset, img_ids, are_ids_selected, ax=None):

    # Create landscape figure
    creating_fig = ax is None
    if creating_fig: fig, ax = plt.subplots()
    else: fig = ax.figure

    # Plot images
    labels = np.array([dataset[img_id][-1] for img_id in img_ids])
    n_classes = dataset.label_shape[0]-1
    labels = np.array([list(lab).index(1) for lab in labels])
    
    sns.histplot(data=pd.DataFrame(dict(labels=labels)), bins=n_classes, binrange=(0,n_classes), 
                 x="labels", color="blue", label="Observed", alpha=0.5, ax=ax)
    sns.histplot(data=pd.DataFrame(dict(selected=labels))[are_ids_selected], 
                 bins=n_classes, binrange=(0,n_classes), 
                 x="selected", color="red", label="Selected", alpha=.7, ax=ax)
    plt.xlabel("Classes")
    plt.legend()
    if creating_fig: plt.show()

    return fig, ax

def plot_classes_histograms(dataset, list_img_ids, list_are_ids_selected):

    # Create figure
    n_rows = len(list_img_ids)
    fig, axes = set_up_figure(n_rows=n_rows)

    for round_i, ax in enumerate(axes):
        fig, ax = plot_classes_histogram(dataset, list_img_ids[round_i], list_are_ids_selected[round_i], ax=ax)
    for i, ax in enumerate(axes):
        if i!=0: ax.set_ylabel("")
        if i<n_rows-1: ax.set_xlabel("")
    fig.show()

    return fig, ax