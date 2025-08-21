import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))

import shutil
import pickle
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
from ours.dataset import DATASET_OPTIONS
from ours.utils import get_wandb_id

#----------------------------------------------------------------------------
# Load classifier model

def load_swin_l_model(verbose=False, deterministic=True):
    """Tiny ImageNet pre-trained Swin-L model from Hyun et al, 2022.
    
    See more
    --------
    https://arxiv.org/abs/2205.10660
    https://github.com/ehuynh1106/TinyImageNet-Transformers/tree/main
    https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/swin_large_384.pth
    """

    if verbose: print("Loading Swin-L for TinyImageNet")

    # Ensure deterministic algortithms
    if deterministic: torch.use_deterministic_algorithms(True)
    # Without it, this Swin-L model can somehow return different values for the same input

    # Create Swin-L model
    model = timm.create_model('swin_large_patch4_window12_384', pretrained=False, drop_path_rate=0.1).cpu()
    for param in model.parameters():
        param.requires_grad = False
    model.reset_classifier(num_classes=200)

    # Load pre-trained weights (fine-tuned on Tiny ImageNet)
    model_filepath = os.path.join(PRETRAINED_HOME, "swin_large_384.pth")
    checkpoint = torch.load(model_filepath)
    model = load_weights_and_check(model, checkpoint["model_state_dict"], verbose=verbose)
    model.eval()

    return model

def load_resnet_101_model(verbose=False):
    """Tiny ImageNet pre-trained ResNet-101 model from Luo et al, 2019.
    
    See more
    --------
    https://arxiv.org/abs/1912.08136
    https://github.com/luoyan407/congruency/tree/master
    https://drive.google.com/open?id=1RLyQIcJ8qNqds9US-Oo2a0uQEL0t6kSZ 
    """

    if verbose: print("Loading ResNet-101 for TinyImageNet")

    # Create a ResNet-101 model to classify TinyImageNet (200 classes) instead of ImageNet (1000 classes)
    model = torchv.models.ResNet(torchv.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=200)

    # Load pre-trained weights (fine-tuned on Tiny ImageNet)
    model_filepath = os.path.join(PRETRAINED_HOME, "resnet101_dcl_60_1_TinyImageNet.pth.tar")
    checkpoint = torch.load(model_filepath)

    # Get the trainable parameters from the checkpoint and from the model
    checkpoint_params = {k:p for k,p in checkpoint["state_dict"].items() if "running" not in k and "num_batches" not in k}
    checkpoint_params.update( {k:p for k,p in checkpoint["classifier_state_dict"].items() if "running" not in k and "num_batches" not in k} )
    model_params = dict({k: p for k, p in model.named_parameters()})

    # Make a list of checkpoint keys that will need to be replaced
    # (the model is from an old library, so the names need to be changed in order to make it compatible with torchvision.models)
    replace_keys = {}
    all_matches = True
    for ((check_k, check_p), (model_k, model_p)) in zip(checkpoint_params.items(), model_params.items()):
        all_matches = all_matches and check_p.shape==model_p.shape
        check_k_start = ".".join( check_k.split(".")[:-1] )
        model_k_start = ".".join( model_k.split(".")[:-1] )
        if check_k_start not in replace_keys.keys():
            replace_keys[check_k_start] = model_k_start
        else:
            if replace_keys[check_k_start] != model_k_start:
                print("Beware! Unconsistency for...", check_k_start, model_k_start, replace_keys[check_k_start])
        if check_p.shape != model_p.shape:
            print("Beware! Dimensions are unmatched for...", (check_k, check_p.shape), (model_k, model_p.shape))
    
    # Build a checkpoint that is compatible with the model
    all_checkpoint_params = dict(**checkpoint["state_dict"], **checkpoint["classifier_state_dict"])
    new_checkpoint = {}
    for k, p in all_checkpoint_params.items():
        decomposed_k = k.split(".")
        k_start = ".".join( decomposed_k[:-1] )
        new_k = replace_keys[k_start] + "." + decomposed_k[-1]
        # print(k, new_check_k)
        new_checkpoint[new_k] = p
        
    # Load checkpoint into the model
    model = load_weights_and_check(model, new_checkpoint, verbose=verbose)
    model.eval()

    return model

def load_wavemix_model(verbose=False):

    model = WaveMix(num_classes = 200,
                    depth = 16,
                    mult = 2,
                    ff_channel = 192,
                    final_dim = 192,
                    dropout = 0.5,
                    level = 3,
                    initial_conv = 'pachify',
                    patch_size = 4)

    model_filepath = os.path.join(PRETRAINED_HOME, "tiny_78.76.pth")
    checkpoint = torch.load(model_filepath)

    model = load_weights_and_check(model, checkpoint, verbose=verbose)
    model.eval()

    return model

def load_classifier_model(model_name, verbose=True, **kwargs):
    if model_name == "ResNet":
        return load_resnet_101_model(verbose=verbose)
    elif model_name == "Swin":
        return load_swin_l_model(verbose=verbose, **kwargs)
    elif model_name == "WaveMix":
        return load_wavemix_model(verbose=verbose)
    else:
        raise ValueError("Unrecognized or unsupported model")

#--------------------------------------------------------------------------------
# Utilities to calculate classification metrics

def get_classification_metrics_dir(model, dataset_name="tiny", image_path=None):
    if model not in ["Swin", "ResNet", "WaveMix"]: raise NotImplementedError("Unknown model")
    assert dataset_name in ["tiny", "generated"], NotImplementedError("Dataset is not supported")
    if dataset_name=="generated":
        assert image_path is not None, ValueError("Generated dataset requires image path")
        folder_path, folder_name = os.path.split(image_path)
        if "gen_images" in folder_path: folder_path = os.path.split(folder_path)[0]
        save_dir = os.path.join(folder_path, "class_metrics", folder_name)
    else:
        if model == "ResNet":
            save_dir = os.path.join(dirs.DATA_HOME, "class_metrics", "tiny", "resnet")
        elif model == "Swin":
            save_dir = os.path.join(dirs.DATA_HOME, "class_metrics", "tiny", "swin")
        elif model == "WaveMix":
            save_dir = os.path.join(dirs.DATA_HOME, "class_metrics", "tiny", "wavemix")
    return save_dir

def load_last_classification_metrics(model, dataset_name="tiny", image_path=None):
    save_dir = get_classification_metrics_dir(model, dataset_name=dataset_name, image_path=image_path)

    contents = os.listdir(save_dir)

    conf_files = filter_by_string_must(contents, "conf")
    conf_files.sort()
    conf_filepath = os.path.join(save_dir, conf_files[-1])

    topf_files = filter_by_string_must(contents, "topf")
    topf_files.sort()
    topf_filepath = os.path.join(save_dir, topf_files[-1])

    confusion_matrix = np.load(conf_filepath)
    top_5_correct = np.load(topf_filepath)
    return confusion_matrix, top_5_correct

class LabelsTranslator:

    def __init__(self, from_aux_to_main_indices):
        if not isinstance(from_aux_to_main_indices, list):
            from_aux_to_main_indices = [int(i) for i in from_aux_to_main_indices]
        self._indices_from_aux = from_aux_to_main_indices
        self._indices_to_aux = [from_aux_to_main_indices.index(i) for i in range(len(from_aux_to_main_indices))]

    @property
    def indices_from_aux(self):
        return self._indices_from_aux
    
    @property
    def indices_to_aux(self):
        return self._indices_to_aux

    def from_aux_to_main(self, aux_index):
        return self.indices_from_aux[ aux_index ]
    
    def to_aux_from_main(self, main_index):
        return self.indices_to_aux[ main_index ]

def load_labels_translator(from_aux_to_main_filepath=os.path.join(dirs.DATA_HOME, 
                                                                  "tiny_kaggle_to_huggingface.txt")):

    from_aux_to_main_indices = np.loadtxt(from_aux_to_main_filepath).astype(int)
    labels_translator = LabelsTranslator(from_aux_to_main_indices)

    return labels_translator

def load_dataset(dataset_name="tiny", image_path=None, random_seed=0, **transform_kwargs):
    
    dataset_kwargs = calc.get_dataset_kwargs(dataset_name, image_path=image_path)
    dataset = construct_class_by_name(**dataset_kwargs, random_seed=random_seed)
    dataset = set_up_dataset_transform(dataset, **transform_kwargs)

    return dataset

def set_up_dataset_transform(dataset,
                             do_minus11_rescale=False, do_01_rescale=False,
                             do_upsample=False, upsample_dim=128,
                             do_normalize=True, convert_to_torch=False):
    
    transformations = []
    if convert_to_torch:
        transformations.append( FromNumpyToTorch() )
    if do_upsample:
        transformations.append( transforms.Resize(upsample_dim, interpolation=InterpolationMode.BICUBIC) )
    if do_01_rescale:
        transformations.append( From8bitTo01() )
    elif do_minus11_rescale:
        transformations.append( From8bitToMinus11() )
    if do_normalize:
        transformations.append( transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]) )
    if len(transformations)>0:
        transformations = transforms.Compose( transformations )
    else: transformations = None

    dataset.set_up_transform( transformations )
    return dataset

def get_dataset_transform_kwargs(model_name, dataset_name):

    # According to model...
    if model_name == "ResNet":
        transform_kwargs = dict(do_upsample=False, do_normalize=True,
                                do_01_rescale=True, do_minus11_rescale=False)
    elif model_name == "Swin":
        transform_kwargs = dict(do_upsample=True, upsample_dim=384, do_normalize=True,
                                do_01_rescale=True, do_minus11_rescale=False)
    elif model_name == "WaveMix":
        transform_kwargs = dict(do_upsample=True, upsample_dim=128, do_normalize=True,
                                do_01_rescale=True, do_minus11_rescale=False)
    else:
        transform_kwargs = dict(do_upsample=False, upsample_dim=128, do_normalize=False,
                                do_01_rescale=False, do_minus11_rescale=False)
    
    # According to dataset name too...
    if dataset_name=="generated" and model_name in ["ResNet", "Swin", "WaveMix"]:
        transform_kwargs.update(dict(convert_to_torch=True))
    else:
        transform_kwargs.update(dict(convert_to_torch=True))
    
    return transform_kwargs

#--------------------------------------------------------------------------------
# Calculate classification metrics for a given classifier for the whole dataset

REORDER_FILEPATH = os.path.join(dirs.DATA_HOME, "tiny_swin_to_huggingface.txt") # Swin-L Classifier
# REORDER_FILEPATH = os.path.join(dirs.DATA_HOME, "tiny_kaggle_to_huggingface.txt") # Kaggle

def get_classification_metrics(
        model, dataset,
        n_samples=None,
        batch_size=128,
        start_idx=0,
        shuffle=False,
        save_period=None,
        save_dir=os.path.join(dirs.DATA_HOME, "class_metrics", "tiny"),
        verbose=False,
):

    # Extract dataset information
    n_classes = dataset.n_classes
    n_examples = len(dataset)
    if n_samples is None: n_samples = n_examples
    n_samples = min(n_samples, n_examples)

    # Create a data loader
    batch_size = min(batch_size, n_samples)
    n_batches = int(math.ceil( n_samples / batch_size ))
    last_batch_size = n_samples % batch_size
    if last_batch_size == 0: last_batch_size = batch_size
    data_sampler = InfiniteSampler(dataset, shuffle=False, start_idx=start_idx)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle,
                                              num_workers=2, pin_memory=True, prefetch_factor=2,
                                              sampler=data_sampler)
    
    # Define a labels translator to compensate order discrepancy
    labels_translator = load_labels_translator(REORDER_FILEPATH)

    # Top-5 accuracy per class
    top_5_correct = np.zeros(n_classes, dtype=np.uint32)

    # Confusion matrix --> Can be used to get Top-1 accuracy per class
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.uint32)
    # i-th row and j-th column indicate the number of samples with
    # true label being i-th class and
    # predicted label being j-th class

    # Configure saving
    saving = save_period is not None
    get_numbers = lambda idx, n_seen, acc : f"{idx+1:05d}-{n_seen:07d}-{acc:.4f}"
    get_fpath = lambda lab, idx, n_seen, acc : os.path.join(save_dir, f"{lab}-{get_numbers(idx, n_seen, acc)}.npy")

    # Get stats
    n_seen = 0
    last_conf_mat, last_top_5 = None, None
    for idx, (_, images, onehots) in tqdm.tqdm(enumerate(data_loader), total=n_batches):
        if idx>=n_batches: break
        elif idx==n_batches-1: images, onehots = images[:last_batch_size], onehots[:last_batch_size]
        labels = onehots.argmax(axis=1) # HuggingFace
        predictions = model(images) # Classifier
        predictions = predictions[:, labels_translator.indices_from_aux] # HuggingFace
        predicted_labels = predictions.argmax(axis=1)
        top_5_labels = predictions.argsort(axis=1)[:,-5:]
        for gt, top_1, top_5 in zip(labels, predicted_labels, top_5_labels):
            confusion_matrix[gt, top_1] += 1
            top_5_correct[gt] += gt in top_5
        n_seen += len(images)
        if (saving and (idx-1)%save_period==0) or idx==n_batches-1:
            if last_conf_mat is not None:
                os.remove(last_conf_mat)
                os.remove(last_top_5)
            elif n_batches>save_period: 
                os.makedirs(save_dir, exist_ok=True)
            top_1_accuracy = float( confusion_matrix.diagonal().sum() / n_seen )
            top_5_accuracy = float( top_5_correct.sum() / n_seen )
            last_conf_mat = get_fpath("conf", idx, n_seen, top_1_accuracy)
            last_top_5 = get_fpath("topf", idx, n_seen, top_5_accuracy)
            np.save(last_conf_mat, confusion_matrix)
            np.save(last_top_5, top_5_correct)
    if n_seen != n_samples: print(f"Inconsistency found: n_seen is {n_seen}, but n_samples is {n_samples}")
    if verbose:
        print("Top-1 Accuracy", top_1_accuracy)
        print("Top-5 Accuracy", top_5_accuracy)
    
    return top_1_accuracy, top_5_accuracy, confusion_matrix, top_5_correct

def calculate_classification_metrics(model="Swin", dataset_name="tiny", 
                                     image_path=None, batch_size=128, n_samples=None, 
                                     save_period=1, verbose=False):
    # Choose directory
    save_dir = get_classification_metrics_dir(model, dataset_name, image_path)

    # Load dataset
    if dataset_name not in ["tiny", "generated"]: raise NotImplementedError("Dataset is not supported yet")
    transform_kwargs = get_dataset_transform_kwargs(model, dataset_name)
    dataset = load_dataset(dataset_name=dataset_name, image_path=image_path, **transform_kwargs)

    # Load model
    model = load_classifier_model(model)
    
    # Get classification metrics
    return get_classification_metrics(model, dataset, n_samples=n_samples, batch_size=batch_size, 
                                      save_period=save_period, save_dir=save_dir, verbose=verbose)

#----------------------------------------------------------------------------
# Calculate metrics for generated image datasets stored in the same directory

def calculate_metrics_for_generated_images(super_dir, fd_metrics=True, 
                                           class_metrics=True, verbose=True,
                                           metrics_filepath=None):

    dist.init()
    saving = metrics_filepath is not None
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)

    # Get series of generated images: 
    # each generated images dataset is assumed to be inside a folder inside the main directory
    series = os.listdir(super_dir)
    series = [s for s in series if os.path.isdir(os.path.join(super_dir, s))]
    series = filter_by_string_must(series, ["visualize", "class_metrics", "grid_search"], must=False)

    if fd_metrics:

        # Get reference stats
        ref_path = os.path.join(dirs.DATA_HOME, "dataset_refs", "tiny.pkl")
        if dist.get_rank() == 0:
            ref_exists = os.path.isfile(ref_path)
        if not ref_exists: raise NotImplementedError
        if dist.get_rank() == 0:
            ref = calc.load_stats(path=ref_path) # do this first, just in case it fails
        torch.distributed.barrier()

        # Get FID and FD_DINOv2 models
        metrics = calc.parse_metric_list("fid,fd_dinov2")
        detectors = [calc.get_detector(metric, verbose=verbose) for metric in metrics]

    if class_metrics:

        # Get classifier model
        classifier = load_classifier_model("Swin", deterministic=not fd_metrics)

    results = {}
    for s in series:
        if verbose: dist.print0("Series", s)
        image_path = os.path.join(super_dir, s)
        results[s] = {}

        # Load dataset
        dataset = load_dataset(dataset_name="generated", image_path=image_path)

        # Calculate FID and FD-DINOv2 metrics for generated images
        if fd_metrics:
            if class_metrics: torch.use_deterministic_algorithms(False)
            stats_iter = calc.calculate_stats_for_dataset(dataset, metrics=metrics, detectors=detectors)
            r = calc.use_stats_iterator(stats_iter)
            if dist.get_rank() == 0:
                initial_time = time.time()
                fd_results = calc.calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics, verbose=verbose)
                results[s].update({"Validation FID": fd_results["fid"],
                                   "Validation FD-DINOv2": fd_results["fd_dinov2"]})
                cumulative_time = time.time() - initial_time
                dist.print0(f"Time to get FD metrics = {cumulative_time:.2f} sec")
            torch.distributed.barrier()

        # Calculate classification metrics on the generated images using a pre-trained model 
        if class_metrics:
            if fd_metrics: torch.use_deterministic_algorithms(True)
            initial_time = time.time()

            # Reconfigure the dataset to have the appropriate preprocessing
            transform_kwargs = get_dataset_transform_kwargs("Swin", "generated")
            dataset = set_up_dataset_transform(dataset, **transform_kwargs)

            # Get classification scores
            class_save_dir = get_classification_metrics_dir("Swin", "generated", image_path)
            class_results = get_classification_metrics(classifier, dataset, 
                                                            n_samples=None, batch_size=128, 
                                                            save_period=1, save_dir=class_save_dir, 
                                                            verbose=verbose)
            cumulative_time = time.time() - initial_time
            dist.print0(f"Time to get classification metrics = {cumulative_time:.2f} sec")
            top_1_accuracy, top_5_accuracy, confusion_matrix, top_5_correct = class_results
            results[s].update({"Swin-L Top-1 Accuracy": top_1_accuracy,
                               "Swin-L Top-5 Accuracy": top_5_accuracy,
                               "Confusion matrix": confusion_matrix,
                               "Top 5 correct": top_5_correct})
        
        if verbose: dist.print0(results[s])
        if saving and dist.get_rank()==0:
            with open(metrics_filepath, 'wb') as f:
                pickle.dump(results, f)
        torch.distributed.barrier()

    return results

#----------------------------------------------------------------------------
# Calculate metrics for a series of models

def calculate_metrics_for_checkpoints(
        checkpoint_filepaths,
        dataset_name = "tiny",          # Dataset used for training.
        ref_path = None,                # Filepath to dataset reference metrics.
        guide_path = None,              # Filepath to guide model.
        guidance_weight = 1,            # Guidance weight. Default = 1 (no guidance).
        class_idx = None,               # Class label. None = use automatic selection.
        random_class = False,           # Automatic selection can be uniformly random or forced exact distribution.
        seeds = range(0, int(2e3)),     # List of random seeds.
        fd_metrics = True,              # Whether to calculate FID and FD-DINOv2 metrics or not
        class_metrics = True,           # Whether to calculate classification metrics or not
        chosen_emas = None,             # List of chosen EMAs. Default: use all.
        save_nimg = 0,                  # How many images to keep, the rest will be deleted.
        verbose = True,                 # Enable status prints?
        log_to_wandb = True,            # Log to W&B?
        device = torch.device("cuda"),  # Which compute device to use.
        results_filepath = None,
        **sampler_kwargs
):
    
    assert fd_metrics or class_metrics, ValueError("Nothing to calculate")
    assert dataset_name in DATASET_OPTIONS.keys(), "Unrecognized dataset"
    if dataset_name=="folder": raise NotImplementedError

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

    # Separate by EMA and filter EMAs, if specified
    checkpoint_emas = [abs(find_numbers(f)[-1]) for f in checkpoint_filepaths]
    available_emas = list(set(checkpoint_emas))
    if chosen_emas is None:
        chosen_emas = available_emas
    else:
        if not isinstance(chosen_emas, list):
            chosen_emas = [chosen_emas]
        chosen_emas = [ema for ema in chosen_emas if ema in available_emas]
        if len(chosen_emas)==0: 
            raise ValueError("Specified EMA/s not available")
    checkpoint_filepaths_by_ema = []
    for ema in chosen_emas:
        these_filepaths = filter_by_string_must(checkpoint_filepaths, f"{ema:.3f}")
        these_filepaths.sort()
        checkpoint_filepaths_by_ema.append(these_filepaths)

    # Configure sampler
    final_sampler_kwargs = EasyDict(DEFAULT_SAMPLER)
    for k in sampler_kwargs.keys(): final_sampler_kwargs[k] = sampler_kwargs[k]
    # final_sampler_kwargs.sampler_fn = "Images.generate_images.edm_full_sampler"

    # Configure distributed execution and resume W&B logging
    dist.init()
    if log_to_wandb:
        def open_wandb_validation_run(checkpoint_dir):
            wandb_id = get_wandb_id(checkpoint_dir)
            if ref_path is not None: 
                masked_ref = os.path.split(ref_path)[-1]
            else: masked_ref = None
            run = wandb.init(entity="ajest", project="Images", id=wandb_id, resume="allow",
                config=dict(validation_kwargs={datetime.today().strftime('%Y-%m-%d %H:%M:%S'):dict(
                    dataset_name=dataset_name, ref_path=masked_ref, 
                    guide_path=guide_path, guidance_weight=guidance_weight,
                    class_idx=class_idx, random_class=random_class, 
                    seeds=seeds, chosen_emas=chosen_emas, **sampler_kwargs)}),
                settings=wandb.Settings(x_disable_stats=True))
            return run
        def close_wandb_validation_run(run):
            torch.distributed.barrier()
            try: run.finish()
            except AttributeError: pass
            torch.distributed.barrier()

    # For each EMA in chosen EMAs, and for each checkpoint inside the directory
    if fd_metrics:
        metrics = calc.parse_metric_list("fid,fd_dinov2")
        detectors = [calc.get_detector(metric, verbose=verbose) for metric in metrics]
    if class_metrics:
        classifier = load_classifier_model("Swin", deterministic=False)
    if log_to_wandb: 
        open_directory = os.path.dirname(checkpoint_filepaths_by_ema[0][0])
        if dist.get_rank()==0: 
            run = open_wandb_validation_run(open_directory)
    for i, (filepaths, ema) in enumerate(zip(checkpoint_filepaths_by_ema, chosen_emas)):
        if guidance_weight!=1 and guide_path is not None:
            tag = f" [EMA={ema:.3f}, Guidance={guidance_weight:.2f}]"
        else:
            tag = f" [EMA={ema:.3f}]"
        if log_to_wandb: 
            wandb_logs = {"Validation Epoch": 0}
            if fd_metrics:
                wandb_logs.update({f"Validation FID"+tag: 0., f"Validation FD-DINOv2"+tag: 0.})
            if class_metrics:
                wandb_logs.update({f"Swin-L Top-1 Accuracy"+tag: 0., f"Swin-L Top-5 Accuracy"+tag: 0.})
        for checkpoint_filepath in filepaths:
            checkpoint_dir, checkpoint_filename = os.path.split(checkpoint_filepath)
            if log_to_wandb and checkpoint_dir != open_directory:
                close_wandb_validation_run(run)
                open_directory = os.path.dirname(checkpoint_filepaths_by_ema[0][0])
                if dist.get_rank()==0: 
                    run = open_wandb_validation_run(open_directory)
            checkpoint_epochs = abs(find_numbers(checkpoint_filename)[-2])
            if log_to_wandb: wandb_logs.update({"Validation Epoch": checkpoint_epochs})
            if verbose: dist.print0(f">>>>> Working on EMA {ema:.3f} and Epoch {checkpoint_epochs}")

            # Generate images
            if class_metrics: torch.use_deterministic_algorithms(False)
            if guidance_weight==1:
                temp_dir = os.path.join(checkpoint_dir, "gen_images", checkpoint_filename.split(".pkl")[0])
            else:
                temp_dir = os.path.join(checkpoint_dir, "gen_images", checkpoint_filename.split(".pkl")[0]+f"_{guidance_weight:.2f}")
            generate_images(checkpoint_filepath, gnet=guide_path, outdir=temp_dir,
                            guidance=guidance_weight, class_idx=class_idx, random_class=random_class, 
                            seeds=seeds, verbose=verbose, device=device, **final_sampler_kwargs)
            
            # Load dataset
            dataset = load_dataset(dataset_name="generated", image_path=temp_dir)
            
            # Calculate FID and FD-DINOv2 metrics for generated images
            if fd_metrics:
                stats_iter = calc.calculate_stats_for_dataset(dataset, metrics=metrics, detectors=detectors, device=device)
                r = calc.use_stats_iterator(stats_iter)
                if dist.get_rank() == 0:
                    initial_time = time.time()
                    results = calc.calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics, verbose=verbose)
                    if log_to_wandb:
                        wandb_logs.update({"Validation FID"+tag: results["fid"],
                                           "Validation FD-DINOv2"+tag: results["fd_dinov2"]})
                        # run.log(wandb_logs)
                    cumulative_time = time.time() - initial_time
                    dist.print0(f"Time to get metrics = {cumulative_time:.2f} sec")
                torch.distributed.barrier()

            # Calculate classification metrics on the generated images using a pre-trained model 
            if class_metrics:
                torch.use_deterministic_algorithms(True)

                # Reconfigure the dataset to have the appropriate preprocessing
                transform_kwargs = get_dataset_transform_kwargs("Swin", "generated")
                dataset = set_up_dataset_transform(dataset, **transform_kwargs)

                # Get classification scores
                class_save_dir = get_classification_metrics_dir("Swin", "generated", temp_dir)
                class_results = get_classification_metrics(classifier, dataset, 
                                                           n_samples=None, batch_size=128, 
                                                           save_period=1, save_dir=class_save_dir, 
                                                           verbose=verbose)
                top_1_accuracy, top_5_accuracy, confusion_matrix, top_5_correct = class_results
                if log_to_wandb:
                    wandb_logs.update({"Validation Swin-L Top-1 Accuracy"+tag: top_1_accuracy,
                                       "Validation Swin-L Top-5 Accuracy"+tag: top_5_accuracy})
            
            # Log to W&B
            if log_to_wandb:
                if class_metrics and checkpoint_filename==filepaths[-1]:
                    run.log({f"Swin-L Confusion Matrix"+tag: confusion_matrix,
                             f"Swin-L Top-5 Correct"+tag: top_5_correct,
                             **wandb_logs})
                else: run.log(wandb_logs)

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

    if log_to_wandb: close_wandb_validation_run(run)

#----------------------------------------------------------------------------
# Calculate metrics for all stored models as a post-hoc validation curve

def calculate_metrics_for_all_checkpoints(
        checkpoints_dir,
        min_epoch = None,               # Number of epochs to start processing from.
        max_epoch = None,               # Number of epochs to stop processing at.
        period_epoch = None,            # Sample every period_epoch epochs
        **calc_metrics_kwargs
):

    # Get available checkpoints
    checkpoint_filenames = filter_by_string_must(os.listdir(checkpoints_dir), ".pkl")    
    checkpoint_filenames = filter_by_string_must(checkpoint_filenames, "0000000", must=False) # Skip randomly initialized networks
    checkpoint_emas = [abs(find_numbers(f)[-1]) for f in checkpoint_filenames]
    if min_epoch is not None or max_epoch is not None:
        checkpoint_filenames = np.array(checkpoint_filenames)
        checkpoint_epochs = np.array([abs(find_numbers(f)[0]) for f in checkpoint_filenames], dtype=np.int32)
        if period_epoch is not None:
            keep_epochs = checkpoint_epochs%period_epoch==0
            checkpoint_epochs = checkpoint_epochs[keep_epochs]
            checkpoint_filenames = checkpoint_filenames[keep_epochs]
        if min_epoch is not None:
            checkpoint_filenames = checkpoint_filenames[checkpoint_epochs >= min_epoch]
            checkpoint_epochs = np.array([abs(find_numbers(f)[0]) for f in checkpoint_filenames], dtype=np.int32)
        if max_epoch is not None:
            checkpoint_filenames = checkpoint_filenames[checkpoint_epochs <= max_epoch]
        checkpoint_filenames = list(checkpoint_filenames)
    checkpoint_filepaths = [os.path.join(checkpoints_dir, f) for f in checkpoint_filenames]

    calculate_metrics_for_checkpoints(checkpoint_filepaths, **calc_metrics_kwargs)

@click.group()
def cmdline(): pass

@cmdline.command()
@click.option('--model', help='Model to be used', type=click.Choice(["Swin", "ResNet", "WaveMix"]), required=False, default="ResNet", show_default=True)
@click.option('--dataset', 'dataset_name',  help='Dataset to be used', metavar='STR', type=click.Choice(list(DATASET_OPTIONS.keys())), default="tiny", show_default=True)
@click.option('--data', 'image_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, default=None)
@click.option('--batch', 'batch_size', help='Batch size', type=int, required=False, default=128, show_default=True)
@click.option('--n', 'n_samples', help='Number of samples to consider', type=int, required=False, default=None, show_default=True)
@click.option('--period', 'save_period', help='Saving period, expressed in batches', type=int, required=False, default=None, show_default=True)
@click.option('--verbose/--no-verbose', 'verbose',  help='Show prints?', metavar='BOOL', type=bool, default=True, show_default=True)
def classification_dataset(model, dataset_name, image_path, batch_size, n_samples, save_period, verbose):
    calculate_classification_metrics(model=model, dataset_name=dataset_name, image_path=image_path, 
                                     batch_size=batch_size, n_samples=n_samples, 
                                     save_period=save_period, verbose=verbose);

@cmdline.command()
@click.option('--path', 'super_dir', help='Path to directory containing generated images datasets', metavar='PATH|ZIP', type=str)
@click.option('--out', 'metrics_filepath', help='Filepath to .pkl file in which results will be stored', metavar='PATH|ZIP', type=str)
@click.option('--fd-metrics/--no-fd-metrics', 'fd_metrics',  help='Calculate FID and FD-DINOv2 metrics?', type=bool, default=True, show_default=True)
@click.option('--class-metrics/--no-class-metrics', 'class_metrics',  help='Calculate classification metric?', type=bool, default=True, show_default=True)
@click.option('--verbose/--no-verbose', 'verbose',  help='Show prints?', metavar='BOOL', type=bool, default=True, show_default=True)
def metrics_generated(super_dir, fd_metrics, class_metrics, verbose, metrics_filepath):
    super_dir = os.path.join(dirs.RESULTS_HOME, "Images", super_dir)
    metrics_filepath = os.path.join(dirs.RESULTS_HOME, "Images", metrics_filepath)
    calculate_metrics_for_generated_images(super_dir, fd_metrics=fd_metrics, 
                                            class_metrics=class_metrics, verbose=verbose,
                                            metrics_filepath=metrics_filepath);    

@cmdline.command()
@click.option("--models-dir", "models_dir", help="Relative path to directory containing the model checkpoints", type=str, metavar='PATH', required=True)
@click.option('--dataset', 'dataset_name', help='Dataset to be used', type=click.Choice(list(DATASET_OPTIONS.keys())), default="tiny", show_default=True)
@click.option('--ref', 'ref_path', help='Dataset reference statistics ', type=str, metavar='PATH', required=False, default=None, show_default=True)
@click.option('--guide-path', 'guide_path', help='Guide model filepath', type=str, metavar='PATH', default=None, show_default=True)
@click.option('--guidance-weight', 'guidance_weight', help='Guidance strength: default is 1 (no guidance)', type=float, default=1.0, show_default=True)
@click.option('--fd-metrics/--no-fd-metrics', 'fd_metrics',  help='Calculate FID and FD-DINOv2 metrics?', type=bool, default=True, show_default=True)
@click.option('--class-metrics/--no-class-metrics', 'class_metrics',  help='Calculate classification metric?', type=bool, default=True, show_default=True)
@click.option('--random/--no-random', 'random_class',  help='Use random classes?', type=bool, default=False, show_default=True)
@click.option('--emas', help='Chosen EMA length/s', metavar='LIST', type=str, required=False, default=None, show_default=True)
@click.option('--min-epoch', help='Number of batches at which to start', type=int, required=False, default=None, show_default=True)
@click.option('--max-epoch', help='Number of batches at which to stop', type=int, required=False, default=None, show_default=True)
@click.option('--period', help='Period of the number of batches to use for sampling', type=int, required=False, default=None, show_default=True)
@click.option('--save-nimg', help='Number of generated images to keep', type=int, required=False, default=0, show_default=True)
@click.option('--seeds', help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST', type=parse_int_list, default='0-1999', show_default=True)
@click.option('--wandb/--no-wandb', 'log_to_wandb',  help='Log to W&B?', type=bool, default=True, show_default=True)
def validation(models_dir, dataset_name, ref_path, guide_path, guidance_weight, fd_metrics, class_metrics,
               random_class, emas, min_epoch, max_epoch, period, seeds, save_nimg, log_to_wandb):
    models_dir = os.path.join(dirs.MODELS_HOME, "Images", models_dir)
    if ref_path is not None: ref_path = os.path.join(dirs.DATA_HOME, "dataset_refs", ref_path)
    if guide_path is not None: guide_path = os.path.join(dirs.MODELS_HOME, "Images", guide_path)
    if emas is not None: emas = [float(ema) for ema in emas.split(",")]
    calculate_metrics_for_all_checkpoints(models_dir,
        dataset_name=dataset_name, ref_path=ref_path,
        guide_path=guide_path, guidance_weight=guidance_weight,
        fd_metrics=fd_metrics, class_metrics=class_metrics,
        random_class=random_class, chosen_emas=emas, 
        min_epoch=min_epoch, max_epoch=max_epoch, period_epoch=period, seeds=seeds, 
        save_nimg=save_nimg, log_to_wandb=log_to_wandb)

if __name__ == "__main__":
    cmdline()