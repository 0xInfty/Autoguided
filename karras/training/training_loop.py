# This is an adaptation from code found at "EDM2 and Autoguidance" by Tero Karras et al
# https://github.com/NVlabs/edm2/blob/main/training/training_loop.py licensed under CC BY-NC-SA 4.0
#
# Original copyright disclaimer:
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Main training loop."""

import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)

import os
import time
import copy
import pickle
import psutil
import builtins
import math
import numpy as np
import torch
import wandb
import pyvtorch.aux as taux
import json

import karras.dnnlib as dnnlib
import karras.torch_utils.distributed as dist
import karras.torch_utils.training_stats as training_stats
import karras.torch_utils.persistence as persistence
import karras.torch_utils.misc as misc
from ours.utils import move_wandb_files, get_wandb_name, get_wandb_tags
from ours.selection import REQUIRES_REF_LOSS, get_selection_size

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule

def learning_rate_schedule(cur_nimg, cur_epoch, run_selection=False, early=False, late=False,
                           mini_batch_size=None, super_batch_size=None, ref_batch_size=2048,
                           ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10, 
                           change_nimg=0, change_epoch=0,
                           verbose=False):
    
    super_batch_size = super_batch_size or ref_batch_size
    mini_batch_size = mini_batch_size or super_batch_size
    assert super_batch_size >= mini_batch_size, "Super batch size cannot be smaller than mini batch size"
    assert not run_selection or (super_batch_size > mini_batch_size), "Selection requires uper_batch_size > mini_batch_size"

    # Infer batch size
    if run_selection: batch_size = mini_batch_size
    else: batch_size = super_batch_size

    if verbose:
        print("Original")
        print("> N img", cur_nimg)
        print("> Rampup img", rampup_Mimg*1e6)
        print("> Ref batches", ref_batches)
        print("> Ref batches * batch size > Rampup Nimg", ref_batches * mini_batch_size > rampup_Mimg*1e6)

    # Data selection factor
    factor = super_batch_size / mini_batch_size
    if verbose: print("Data selection factor", factor)

    # Make the function continuous if AJEST is triggered to run early or late
    if change_epoch != 0:
        if not run_selection and early:
            change_factor = factor - 1
            cur_nimg = cur_nimg + change_factor * change_nimg
            cur_epoch = cur_epoch + change_factor * change_epoch
            # Maybe change_epoch should be fractional if round>0
        elif run_selection and late:
            if verbose: print("Late data selection clause")
            change_factor = (factor - 1)/factor
            cur_nimg = cur_nimg - change_factor * change_nimg
            cur_epoch = cur_epoch - change_factor * change_epoch
    
    # Make it decay faster while selecting data
    if verbose: print("Actual data selection factor", super_batch_size / batch_size)
    cur_epoch = cur_epoch * super_batch_size / batch_size
    cur_nimg = cur_nimg * super_batch_size / batch_size

    # Ramp up according to number of images seen with the super batch size and no data selection
    rampup_nimg = rampup_Mimg * 1e6 * batch_size / super_batch_size
    
    # Decay according to number of batches, regardless of batch size
    # ref_batches = ref_batches

    if verbose:
        print("Final")
        print("> N batches", cur_nimg)
        print("> Rampup img", rampup_nimg)
        print("> Ref batches (t0)", ref_batches)
        print("> Ref batches * batch size > Rampup Nimg", ref_batches * batch_size > rampup_nimg)

    # Apply original Karras et al's schedule from the paper "Analyzing and Improving
    # the Training Dynamics of Diffusion Models".
    lr = ref_lr
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / rampup_nimg, 1)
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_epoch / ref_batches, 1))
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='karras.training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='karras.training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='karras.training.networks_edm2.Precond'),
    ref_network_kwargs  = dict(class_name='karras.training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='karras.training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='karras.training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='karras.training.phema.PowerFunctionEMA'),
    selection_kwargs    = dict(func_name='ours.selection.jointly_sample_batch', N=8, filter_ratio=0.8),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.

    ref_path            = None,     # Reference model path - will be used for data selection if needed

    selection           = False,    # Run data selection
    selection_late      = False,    # Run data selection once the learner becomes better than the ref
    selection_early     = False,    # Run data selection only while the learner is worse than the ref

    status_period         = (128<<10)/2048,  # Report status every N epochs. None = disable.
    snapshot_period       = (8<<20)/2048,    # Save network snapshot every N epochs. None = disable.
    checkpoint_period     = (128<<20)/2048,  # Save state checkpoint every N epochs. None = disable.
    detail_max_epoch      = 1000,            # Save five times as frequently until this point

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    prev_status_time = time.time()
    rank = dist.get_rank()
    misc.set_random_seed(seed, rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    world_size = dist.get_world_size()
    batch_gpu_total = batch_size // world_size
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    loss_factor = loss_scaling / batch_gpu_total
    dist.print0("\nBatch size calculation")
    dist.print0(">>> Batch size total", batch_size)
    dist.print0(">>> World size", world_size)
    dist.print0(">>> Batch size total per GPU", batch_gpu_total)
    dist.print0(">>> Maximum batch size per GPU", batch_gpu)
    dist.print0(">>> Num accumulation rounds", num_accumulation_rounds, "\n")
    assert batch_size == batch_gpu * num_accumulation_rounds * world_size
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_period is None or status_period % 4 == 0
    assert snapshot_period is None or snapshot_period % 4 == 0
    assert checkpoint_period is None or checkpoint_period % 4 == 0
    detail_status_period = int(status_period / 4) if status_period is not None else None
    detail_snapshot_period = int(snapshot_period / 4) if snapshot_period is not None else None
    detail_checkpoint_period = int(checkpoint_period / 4) if checkpoint_period is not None else None

    # Set up data selection
    dist.print0("Data selection =", selection)
    is_ref_available = ref_path is not None
    if selection:
        mini_batch_gpu = get_selection_size(batch_gpu, **selection_kwargs)
        mini_batch_size = mini_batch_gpu * num_accumulation_rounds *  world_size
        requires_ref_loss = selection_kwargs.func_name.split.split("ours.selection.")[-1] in REQUIRES_REF_LOSS
        if not is_ref_available and selection and requires_ref_loss:
            raise ValueError("Missing reference model")
        if selection_late:
            run_selection = False
            is_selection_waiting = True
            broadcast_operation = torch.distributed.ReduceOp.MAX # If any process sets run to True, set all of them to True
            dist.print0("Data selection with delayed execution strategy")
        elif selection_early:
            run_selection = True
            is_selection_waiting = True
            broadcast_operation = torch.distributed.ReduceOp.MIN # If any process sets run to False, set all of them to False
            dist.print0("Data selection programmed stop strategy")
        else:
            run_selection = True
            is_selection_waiting = False
            dist.print0("Data selection with early-start execution strategy")
        selection_kwargs.requires_ref_loss = requires_ref_loss
        selection_kwargs.selection_size = mini_batch_gpu
        selection_kwargs.mini_batch_size = mini_batch_size
        dist.print0("Data selection configuration =", json.dumps(selection_kwargs, indent=2))
    else: 
        mini_batch_gpu = batch_gpu
        mini_batch_size = batch_size
        run_selection = False
        is_selection_waiting = False
        requires_ref_loss = False
    change_just_happened = False
    change_epoch, change_nimg = 0, 0
    net_beats_ref = False
    
    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    _, ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    if is_ref_available:
        dist.print0('Constructing reference network...')
        ref = dnnlib.util.construct_class_by_name(**ref_network_kwargs, **interface_kwargs)
        ref.eval().requires_grad_(False).to(device)

    # Print network summary.
    if rank == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Print reference network summary.
    if is_ref_available:
        if rank == 0:
            misc.print_module_summary(ref, [
                torch.zeros([batch_gpu, ref.img_channels, ref.img_resolution, ref.img_resolution], device=device),
                torch.ones([batch_gpu], device=device),
                torch.zeros([batch_gpu, ref.label_dim], device=device),
            ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0, cur_epoch=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    if is_ref_available: 
        with builtins.open(ref_path, "rb") as f:
            data = dnnlib.EasyDict(pickle.load(f))
        ref = data.ema.to(device)
        ref.eval().requires_grad_(False)
    
    # Decide how long to train.
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_period if checkpoint_period is not None else snapshot_period if snapshot_period is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Set up a W&B experiment
    os.environ["WANDB_DIR"] = run_dir
    group_name = get_wandb_name(run_dir)
    tags = get_wandb_tags(dataset_kwargs)
    if is_ref_available: wandb_ref_path = ref_path.split(dirs.MODELS_HOME)[-1]
    else: wandb_ref_path = None
    if world_size>1:
        run_name = f"{group_name}_R{rank}"
    else:
        run_name = group_name
    run = wandb.init(
        entity="ajest", project="Images", name=run_name, group=group_name, tags=tags,
        config=dict(dataset_kwargs=dataset_kwargs, encoder_kwargs=encoder_kwargs,
                    data_loader_kwargs=data_loader_kwargs, network_kwargs=network_kwargs,
                    loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs,
                    lr_kwargs=lr_kwargs, ema_kwargs=ema_kwargs,
                    selection_kwargs=selection_kwargs,
                    selection=selection, selection_early=selection_early, selection_late=selection_late,
                    ref_path=wandb_ref_path,
                    seed=seed, batch_size=batch_size, batch_gpu=batch_gpu, 
                    total_nimg=total_nimg, loss_scaling=loss_scaling, device=device),
        settings=wandb.Settings(x_stats_gpu_device_ids=[taux.get_device_number(device)]))

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=rank, num_replicas=world_size, 
                                           seed=seed, start_idx=state.cur_epoch*batch_size)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    prev_cumulative_training_time = 0
    cumulative_selection_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, cur_epoch=state.cur_epoch, 
                                       run_selection=run_selection, early=selection_early, late=selection_late,
                                       mini_batch_size=mini_batch_size, **lr_kwargs)
    kimg_logs = {"Stats' Epoch":state.cur_epoch, 'Speed [sec/tick]':None, 'Speed [sec/kimg]':None}
    epoch_logs = {"Epoch":state.cur_epoch, "Loss":None, "Seen images [kimg]": 0,
                  "Learning rate": lr, "Training Time [sec]":cumulative_training_time*1000}
    round_logs = {"Round's Epoch":state.cur_epoch, "Round":0, "Total Rounds":state.cur_epoch*num_accumulation_rounds, 
                  "Round Loss":None}
    if selection:
        epoch_logs.update({"Epoch Super-Batch Learner Loss": None, "Epoch Super-Batch Reference Loss": None,
                           "Selection time [sec]":cumulative_selection_time*1000})
        round_logs.update({"Round Super-Batch Learner Loss": None, "Round Super-Batch Reference Loss": None})
    while True:
        dist.print0(f"Training round {state.cur_epoch}")
        kimg_logs.update({"Stats' Epoch": state.cur_epoch})
        epoch_logs.update({"Epoch": state.cur_epoch})
        round_logs.update({"Round's Epoch": state.cur_epoch})
        batch_start_time = time.time()

        # Evaluate loss and accumulate gradients.
        misc.set_random_seed(seed, rank, state.cur_epoch*batch_size)
        optimizer.zero_grad(set_to_none=True)
        epoch_nimg = 0
        epoch_loss = []
        epoch_ref_loss, epoch_learner_loss = [], []
        epoch_indices, epoch_selected_indices = [], []
        for round_idx in range(num_accumulation_rounds):
            dist.print0(f"Accumulating {round_idx+1}")
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):

                # Evaluate loss
                indices, images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
                loss = loss.sum(dim=(1,2,3)).mul(loss_factor)
                training_stats.report('Loss/loss', loss)
                epoch_indices.append(list(indices))
                
                # Calculate reference loss
                if (run_selection and requires_ref_loss) or is_selection_waiting:
                    selection_time_start = time.time()
                    ref_loss = loss_fn(net=ref, images=images, labels=labels.to(device))
                    ref_loss = ref_loss.sum(dim=(1,2,3)).mul(loss_factor)
                    training_stats.report('RefLoss/ref_loss', ref_loss)
                    epoch_learner_loss.append(float(loss.mean()))
                    epoch_ref_loss.append(float(ref_loss.mean()))
                    if not net_beats_ref and loss.mean() < ref_loss.mean():
                        net_beats_ref = True
                        print("Network has beaten the reference")
                        if is_selection_waiting:
                            run_selection = not(run_selection)
                            if run_selection: print("Selection will now be run")
                            else: print("Selection will now be stopped")
                            is_selection_waiting = False
                            change_just_happened = True
                    # Inform all other GPUs of changes in run_selection
                    if is_selection_waiting or change_just_happened:
                        sync_tensor = torch.tensor([run_selection], dtype=torch.bool, device=device)
                        torch.distributed.all_reduce(sync_tensor, op=broadcast_operation)
                        new_run_selection = bool(sync_tensor.item())
                        if is_selection_waiting and new_run_selection != run_selection:
                            net_beats_ref = True
                            print("Network has beaten the reference")
                            if new_run_selection: print("Selection will now be run")
                            else: print("Selection will now be stopped")
                            is_selection_waiting = False
                            change_just_happened = True
                        run_selection = new_run_selection
                    cumulative_selection_time += time.time() - selection_time_start
                elif run_selection: ref_loss = None

                # Calculate overall loss
                if run_selection:
                    selection_time_start = time.time()
                    try:
                        dist.print0("Using selection")
                        selected_indices = dnnlib.util.call_func_by_name(mini_batch_gpu, loss, ref_loss, **selection_kwargs)
                        loss = loss[selected_indices] # Use indices of the selection mini-batch
                        epoch_selected_indices.append(list(selected_indices))
                    except ValueError:
                        raise ValueError("Selection has crashed on at least 1 of the GPUs")
                    cumulative_selection_time += time.time() - selection_time_start
                gpu_nimg = len(loss)
                round_nimg = world_size * gpu_nimg
                epoch_nimg += round_nimg
                state.cur_nimg += round_nimg
                epoch_loss.append(float(loss.mean()))
                if change_just_happened:
                    change_epoch = state.cur_epoch + round_idx / (num_accumulation_rounds-1)
                    change_nimg = state.cur_nimg
                    change_just_happened = False

                # Accumulate loss and calculate gradients
                if run_selection:
                    loss.sum().mul(batch_gpu/gpu_nimg).backward()
                    # Instead of B, I'm only adding up b<B terms, so I multiply by B/b
                else: loss.sum().backward()

            # Log on each round and each epoch
            round_logs.update({"Round Loss": epoch_loss[-1], "Round":round_idx, 
                               "Total Rounds":state.cur_epoch*num_accumulation_rounds+round_idx})
            if len(epoch_ref_loss)!=0:
                round_logs.update({"Round Super-Batch Learner Loss": epoch_learner_loss[-1],
                                   "Round Super-Batch Reference Loss": epoch_ref_loss[-1]})
            else:
                round_logs.update({"Round Super-Batch Learner Loss": None,
                                   "Round Super-Batch Reference Loss": None})
            run.log(round_logs)
        epoch_logs.update({"Loss": float(np.mean(epoch_loss)), 
                           "Seen images [kimg]": state.cur_nimg/1000,
                           "Indices": epoch_indices})
        if len(epoch_ref_loss)!=0:
            epoch_logs.update({"Super-Batch Learner Loss": float(np.mean(epoch_learner_loss)),
                               "Super-Batch Reference Loss": float(np.mean(epoch_ref_loss)),
                               "Selection time [sec]": cumulative_selection_time*1000,
                               "Selected indices": epoch_selected_indices})
        else:
            epoch_logs.update({"Super-Batch Learner Loss": None,
                               "Super-Batch Reference Loss": None,
                               "Selected indices": None})

        # Evaluate learning rate.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, cur_epoch=state.cur_epoch, 
                                           run_selection=run_selection, early=selection_early, late=selection_late,
                                           change_epoch=change_epoch, change_nimg=change_nimg,
                                           mini_batch_size=mini_batch_size, **lr_kwargs)
        # Used to say mini_batch_size=epoch_nimg
        training_stats.report('Loss/learning_rate', lr)
        epoch_logs.update({"Learning rate": float(lr)})

        # Run optimizer and update weights.
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

        # Report status.
        done = (state.cur_nimg >= stop_at_nimg)
        epoch_logs.update({"Training Time [sec]": cumulative_training_time*1000})
        run.log(epoch_logs)
        if status_period is not None and (done or state.cur_epoch % status_period == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            if state.cur_epoch < detail_max_epoch: 
                status_condition = state.cur_epoch % detail_status_period == 0
            else:
                status_condition = state.cur_epoch % status_period == 0
            if status_condition:
                cur_time = time.time()
                state.total_elapsed_time += cur_time - prev_status_time
                cur_process = psutil.Process(os.getpid())
                cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
                dist.print0(' '.join(['Status:',
                    'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                    'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                    'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                    'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                    'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                    'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                    'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                    'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
                ]))
                kimg_logs.update({
                    'Speed [sec/tick]' : cur_time - prev_status_time,
                    'Speed [sec/kimg]' : (cumulative_training_time - prev_cumulative_training_time) / max(state.cur_nimg - prev_status_nimg, 1) * 1e3,
                })
                run.log(kimg_logs)
                prev_cumulative_training_time = cumulative_training_time
                prev_status_nimg = state.cur_nimg
                prev_status_time = cur_time
                torch.cuda.reset_peak_memory_stats()

                # Flush training stats.
                training_stats.default_collector.update()
                if rank == 0:
                    if stats_jsonl is None:
                        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                    fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                    items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                    items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                    stats_jsonl.write('{' + ', '.join(items) + '}\n')
                    stats_jsonl.flush()

                # Update progress and check for abort.
                dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
                if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                    dist.request_suspend()
                if dist.should_stop() or dist.should_suspend():
                    done = True

        # Save network snapshot.
        if snapshot_period is not None and (state.cur_nimg != start_nimg or start_nimg == 0) and rank == 0:
            if state.cur_epoch < detail_max_epoch: 
                snapshot_condition = state.cur_epoch % detail_snapshot_period == 0
            else:
                snapshot_condition = state.cur_epoch % snapshot_period == 0
            if done or snapshot_condition:
                ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
                for ema_net, ema_suffix in ema_list:
                    data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                    data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                    fname = f'network-snapshot-{state.cur_epoch:07d}{ema_suffix}.pkl'
                    dist.print0(f'Saving {fname} ... ', end='', flush=True)
                    with open(os.path.join(run_dir, fname), 'wb') as f:
                        pickle.dump(data, f)
                    dist.print0('done')
                    del data # conserve memory

        # Save state checkpoint.
        if checkpoint_period is not None and state.cur_nimg != start_nimg:
            if state.cur_epoch < detail_max_epoch: 
                checkpoint_condition = state.cur_epoch % detail_checkpoint_period == 0
            else:
                checkpoint_condition = state.cur_epoch % checkpoint_period == 0
            if done or checkpoint_condition:
                checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_epoch:07d}.pt'))
                misc.check_ddp_consistency(net)

        # Done?
        if done:
            break
        else:
            state.cur_epoch += 1

    try: run.finish()
    except AttributeError: pass
    torch.distributed.barrier()
    if rank==0: move_wandb_files(run_dir, run_dir)

#----------------------------------------------------------------------------