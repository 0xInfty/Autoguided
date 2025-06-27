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
import numpy as np
import torch
import builtins

import karras.dnnlib as dnnlib
import karras.torch_utils.distributed as dist
import karras.torch_utils.training_stats as training_stats
import karras.torch_utils.persistence as persistence
import karras.torch_utils.misc as misc

import ours.selection as sel

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
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='karras.training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='karras.training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='karras.training.networks_edm2.Precond'),
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

    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    dist.print0("\nBatch size calculation")
    dist.print0(">>> World size", dist.get_world_size())
    dist.print0(">>> Batch GPU", batch_gpu)
    dist.print0(">>> Batch size total", batch_gpu_total)
    dist.print0(">>> Num accumulation rounds", num_accumulation_rounds, "\n")
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Set up data selection
    dist.print0("Data selection =", selection)
    is_ref_available = ref_path is not None
    if not is_ref_available and selection:
        raise ValueError("Missing reference model")
    if selection:
        if selection_late:
            run_selection = False
            is_selection_waiting = True
            dist.print0("Data selection with delayed execution strategy")
        elif selection_early:
            run_selection = True
            is_selection_waiting = True
            dist.print0("Data selection programmed stop strategy")
        else:
            run_selection = True
            is_selection_waiting = False
            dist.print0("Data selection with early-start execution strategy")
        dist.print0("Data selection configuration =", selection_kwargs)
    else: 
        run_selection = False
        is_selection_waiting = False
    net_beats_ref = False
    
    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    if is_ref_available:
        dist.print0('Constructing reference network...')
        interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
        ref = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
        ref.eval().requires_grad_(False).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Print reference network summary.
    if is_ref_available:
        if dist.get_rank() == 0:
            misc.print_module_summary(ref, [
                torch.zeros([batch_gpu, ref.img_channels, ref.img_resolution, ref.img_resolution], device=device),
                torch.ones([batch_gpu], device=device),
                torch.zeros([batch_gpu, ref.label_dim], device=device),
            ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
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
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
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
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
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
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):

                # Evaluate loss
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
                training_stats.report('Loss/loss', loss)

                # Calculate reference loss
                if run_selection or is_selection_waiting:
                    ref_loss = loss_fn(net=ref, images=images, labels=labels.to(device))
                    training_stats.report('RefLoss/ref_loss', ref_loss)
                    if not net_beats_ref and loss.mean() < ref_loss.mean():
                        net_beats_ref = True
                        dist.print0("Network has beaten the reference")
                        if is_selection_waiting:
                            run_selection = not(run_selection)
                            if run_selection: dist.print0("Selection will now be run")
                            else: dist.print0("Selection will now be stopped")
                            is_selection_waiting = False
            
                # Calculate overall loss
                if run_selection:
                    try:
                        dist.print0("Using selection")
                        dist.print0("Average Super-Batch Learner Loss =", float(loss.mean()))
                        dist.print0("Average Super-Batch Reference Loss =", float(ref_loss.mean()))
                        dist.print0("Loss shape =", loss.shape)
                        mean_loss = torch.mean(loss, dim=(1,2,3))
                        mean_ref_loss = torch.mean(ref_loss, dim=(1,2,3))
                        indices = dnnlib.util.call_func_by_name(mean_loss, mean_ref_loss, 
                                                                device=torch.device("cpu"), **selection_kwargs)
                        loss = loss[indices] # Use indices of the selection mini-batch
                    except ValueError:
                        dist.print0("Selection has crashed, so it has been deactivated")
                        run_selection = False
                        is_selection_waiting = False

                # Accumulate loss
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

#----------------------------------------------------------------------------