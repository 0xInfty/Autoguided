# This is an adaptation from code found at "EDM2 and Autoguidance" by Tero Karras et al
# https://github.com/NVlabs/edm2/blob/main/toy_example.py licensed under CC BY-NC-SA 4.0
#
# Original copyright disclaimer:
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Auto-guidance 2D toy example that can be run with ACID data selection"""

import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)

import os
import copy
import pickle
import warnings
import functools
import builtins
import numpy as np
import torch
import matplotlib.pyplot as plt
import click
import tqdm
import pyvtools.text as vtext

import dnnlib
from torch_utils import persistence
from utils import mandala_score
import training.phema
import logs

PRETRAINED_HOME = os.path.join(dirs.DATA_HOME, "ToyExample")
if not os.path.isdir(PRETRAINED_HOME): os.mkdir(PRETRAINED_HOME)

log = logs.create_logger("errors")

def get_stats(array):
    if isinstance(array, np.ndarray):
        return (float(array.min()), float(array.sum())/array.size, float(array.max()), array.shape)
    else:
        return (float(array.min()), float(array.sum())/array.numel(), float(array.max()), tuple(array.shape))
get_debug_log = lambda name, array : (f"{name} = [ %s | %s | %s ] %s", *get_stats(array))

#----------------------------------------------------------------------------
# Multivariate mixture of Gaussians. Allows efficient evaluation of the
# probability density function (PDF) and score vector, as well as sampling,
# using the GPU. The distribution can be optionally smoothed by applying heat
# diffusion (sigma >= 0) on a per-sample basis.

class GaussianMixture(torch.nn.Module):
    def __init__(self,
        phi,                        # Per-component weight: [comp]
        mu,                         # Per-component mean: [comp, dim]
        Sigma,                      # Per-component covariance matrix: [comp, dim, dim]
        sample_lut_size = 64<<10,   # Lookup table size for efficient sampling.
    ):
        super().__init__()
        self.register_buffer('phi', torch.tensor(np.asarray(phi) / np.sum(phi), dtype=torch.float32))
        self.register_buffer('mu', torch.tensor(np.asarray(mu), dtype=torch.float32))
        self.register_buffer('Sigma', torch.tensor(np.asarray(Sigma), dtype=torch.float32))

        # Precompute eigendecompositions of Sigma for efficient heat diffusion.
        L, Q = torch.linalg.eigh(self.Sigma) # Sigma = Q @ L @ Q
        self.register_buffer('_L', L) # L: [comp, dim, dim]
        self.register_buffer('_Q', Q) # Q: [comp, dim, dim]

        # Precompute lookup table for efficient sampling.
        self.register_buffer('_sample_lut', torch.zeros(sample_lut_size, dtype=torch.int64))
        phi_ranges = (torch.cat([torch.zeros_like(self.phi[:1]), self.phi.cumsum(0)]) * sample_lut_size + 0.5).to(torch.int32)
        for idx, (begin, end) in enumerate(zip(phi_ranges[:-1], phi_ranges[1:])):
            self._sample_lut[begin : end] = idx

    # Evaluate the terms needed for calculating PDF and score.
    def _eval(self, x, sigma=0):                                                    # x: [..., dim], sigma: [...]
        L = self._L + sigma[..., None, None] ** 2                                   # L' = L + sigma * I: [..., dim]
        d = L.prod(-1)                                                              # d = det(Sigma') = det(Q @ L' @ Q) = det(L'): [...]
        y = self.mu - x[..., None, :]                                               # y = mu - x: [..., comp, dim]
        z = torch.einsum('...ij,...j,...kj,...k->...i', self._Q, 1 / L, self._Q, y) # z = inv(Sigma') @ (mu - x): [..., comp, dim]
        c = self.phi / (((2 * np.pi) ** x.shape[-1]) * d).sqrt()                    # normalization factor of N(x; mu, Sigma')
        w = c * (-1/2 * torch.einsum('...i,...i->...', y, z)).exp()                 # w = N(x; mu, Sigma'): [..., comp]
        return z, w

    # Calculate p(x; sigma) for the given sample points, processing at most the given number of samples at a time.
    def pdf(self, x, sigma=0, max_batch_size=1<<14):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1])
        x_batches = x.flatten(0, -2).split(max_batch_size)
        sigma_batches = sigma.flatten().split(max_batch_size)
        pdf_batches = [self._eval(xx, ss)[1].sum(-1) for xx, ss in zip(x_batches, sigma_batches)]
        return torch.cat(pdf_batches).reshape(x.shape[:-1]) # x.shape[:-1]

    # Calculate log(p(x; sigma)) for the given sample points, processing at most the given number of samples at a time.
    def logp(self, x, sigma=0, max_batch_size=1<<14):
        return self.pdf(x, sigma, max_batch_size).log()

    # Calculate \nabla_x log(p(x; sigma)) for the given sample points.
    def score(self, x, sigma=0):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1])
        z, w = self._eval(x, sigma)
        w = w[..., None]
        return (w * z).sum(-2) / w.sum(-2) # x.shape

    # Draw the given number of random samples from p(x; sigma).
    def sample(self, shape, sigma=0, generator=None):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.mu.device).broadcast_to(shape)
        i = self._sample_lut[torch.randint(len(self._sample_lut), size=sigma.shape, device=sigma.device, generator=generator)]
        L = self._L[i] + sigma[..., None] ** 2                                                  # L' = L + sigma * I: [..., dim]
        x = torch.randn(L.shape, device=sigma.device, generator=generator)                      # x ~ N(0, I): [..., dim]
        y = torch.einsum('...ij,...j,...kj,...k->...i', self._Q[i], L.sqrt(), self._Q[i], x)    # y = sqrt(Sigma') @ x: [..., dim]
        return y + self.mu[i] # [..., dim]

#----------------------------------------------------------------------------
# Construct a ground truth 2D distribution for the given set of classes
# ('A', 'B', or 'AB').

GT_ORIGIN = (0.0030, 0.0325)

@functools.lru_cache(None)
def gt(classes='A', device=torch.device('cpu'), seed=2, origin=np.array(GT_ORIGIN), scale=np.array([1.3136, 1.3844])):
    rnd = np.random.RandomState(seed)
    comps = []

    # Recursive function to generate a given branch of the distribution.
    def recurse(cls, depth, pos, angle):
        if depth >= 7:
            return

        # Choose parameters for the current branch.
        dir = np.array([np.cos(angle), np.sin(angle)])
        dist = 0.292 * (0.8 ** depth) * (rnd.randn() * 0.2 + 1)
        thick = 0.2 * (0.8 ** depth) / dist
        size = scale * dist * 0.06

        # Represent the current branch as a sequence of Gaussian components.
        for t in np.linspace(0.07, 0.93, num=8):
            c = dnnlib.util.EasyDict()
            c.cls = cls
            c.phi = dist * (0.5 ** depth)
            c.mu = (pos + dir * dist * t) * scale
            c.Sigma = (np.outer(dir, dir) + (np.eye(2) - np.outer(dir, dir)) * (thick ** 2)) * np.outer(size, size)
            c.depth = depth
            comps.append(c)

        # Generate each child branch.
        for sign in [1, -1]:
            recurse(cls=cls, depth=(depth + 1), pos=(pos + dir * dist), angle=(angle + sign * (0.7 ** depth) * (rnd.randn() * 0.2 + 1)))

    # Generate each class.
    recurse(cls='A', depth=0, pos=origin, angle=(np.pi * 0.25))
    recurse(cls='B', depth=0, pos=origin, angle=(np.pi * 1.25))

    # Construct a GaussianMixture object for the selected classes.
    comps = [c for c in comps if c.cls in classes]
    distrib = GaussianMixture([c.phi for c in comps], [c.mu for c in comps], [c.Sigma for c in comps])

    return distrib.to(device), comps

#----------------------------------------------------------------------------
# Low-level primitives used by ToyModel.
# Adapted from training/networks/networks_edm2.py.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

@persistence.persistent_class
class MPSiLU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x) / 0.596

@persistence.persistent_class
class MPLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        w = normalize(self.weight) / np.sqrt(self.weight[0].numel())
        return x @ w.t()

#----------------------------------------------------------------------------
# Denoiser model for learning 2D toy distributions. Inputs a set of sample
# positions and a single scalar for each, representing the logarithm of the
# corresponding unnormalized probability density. The score vector can then
# be obtained through automatic differentiation.

@persistence.persistent_class
class ToyModel(torch.nn.Module):
    def __init__(self,
        in_dim      = 2,    # Input dimensionality.
        num_layers  = 4,    # Number of hidden layers.
        hidden_dim  = 64,   # Number of hidden features.
        sigma_data  = 0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.layers = torch.nn.Sequential()
        self.layers.append(MPLinear(in_dim + 2, hidden_dim))
        for _layer_idx in range(num_layers):
            self.layers.append(MPSiLU())
            self.layers.append(MPLinear(hidden_dim, hidden_dim))
        self.gain = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, sigma=0):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1]).unsqueeze(-1)
        x = x / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        y = self.layers(torch.cat([x, sigma.log() / 4, torch.ones_like(sigma)], dim=-1))
        z = (y ** 2).mean(-1) * self.gain / sigma.squeeze(-1) - 0.5 * (x ** 2).sum(-1) # preconditioning
        return z

    def logp(self, x, sigma=0):
        return self(x, sigma)

    def pdf(self, x, sigma=0):
        logp = self.logp(x, sigma=sigma)
        pdf = (logp - logp.max()).exp()
        return pdf

    def score(self, x, sigma=0, graph=False):
        x = x.detach().requires_grad_(True)
        logp = self.logp(x, sigma=sigma)
        score = torch.autograd.grad(outputs=[logp.sum()], inputs=[x], create_graph=graph)[0]
        return score

#----------------------------------------------------------------------------
# JEST & ACID's joint sampling batch selection method

def jointly_sample_batch(learner_loss, ref_loss, N=16, filter_ratio=0.8, 
                         learnability=True, inverted=False, numeric_stability_trick=False, 
                         plotting=False):
    """Joint sampling batch selection method used in JEST and ACID

    Adapted from Evans & Parthasarathy et al's publication titled 
    "Data curation via joint example selection further accelerates multimodal learning"
    (Google DeepMind, London, UK, 2024)
    Available at https://arxiv.org/abs/2406.17711 under CC BY licence
    """
	
    # Define size of super-batch
    B = int(learner_loss.numel()) # Size B of a super-batch

    # Each mini-batch of size b will be split in n chunks
    b_over_N = int(B * (1 - filter_ratio) / N) # Size b/N of each mini-batch chunk

    # Construct the scores matrix
    learner_loss = learner_loss.reshape((B,1))
    ref_loss = ref_loss.reshape((1,B))
    scores = learner_loss - ref_loss if learnability else - ref_loss  # Shape (B,B)
    log.debug(*get_debug_log("Scores", scores))
    if inverted: scores = - scores
    device = scores.get_device()
    # Rows use different learner loss ==> i is the learner/student's index
    # Columns use different reference loss ==> j is the reference/teacher's index
    
    log.debug("> JEST Round 0")

    # Extract basic score for each element of the super-batch
    logits_ii = torch.diag(scores) # Elements from the diagonal of the scores matrix
    #Q: Is this associated to the probability of picking learner i and ref j?
    log.debug(*get_debug_log("Logits ii", logits_ii))
    
    # Draw the first mini-batch chunk using a uniform probability distribution
    indices = np.random.choice(B, b_over_N, replace=False)
    log.debug("Indices (%s)", len(indices))
    log.debug("Number of unique new indices? %s", len(set(indices)))

    # Sample all the rest of the mini-batch chunks
    all_sum_of_probs = []
    for n in range(1, N):

        log.debug("> JEST Round %s", n)
        sampled_so_far = n*b_over_N

        # Get a binary mask that indicates which samples have been selected so far
        is_sampled = torch.eye(B).to(device)[indices].sum(axis=0) # (B,)
        
        # Mask scores to only keep learner rows k that have already been sampled
        logits_kj = (scores * is_sampled.view(B, 1)).sum(axis=0) # Sum over columns (B,)
        log.debug(*get_debug_log("Logits kj", logits_kj))
        #Q: Associated to prob of picking learner i and an ref k that was already selected?

        # Mask scores to only keep ref columns k that have already been sampled
        logits_ik = (scores * is_sampled.view(1, B)).sum(axis=1) # Sum over rows (B,)
        log.debug(*get_debug_log("Logits ik", logits_ik))
        #Q: Associated to prob of picking learner i and an ref k that was already selected?
        
        # Get conditional scores given past samples
        logits = logits_ii + logits_kj + logits_ik
        logits = logits / (2*sampled_so_far+1) # Normalize the logits by the number of terms added up
        logits = logits - is_sampled * 1e8 # Avoid sampling with replacement
        #Q: Why subtract that value, instead of setting all these to 0?

        # Sample new mini-batch chunk using the conditional probability distribution
        log.debug(*get_debug_log("Logits", logits))
        if numeric_stability_trick: logits = logits - max(logits)
        probabilities = np.exp(logits.detach().cpu().numpy())
        sum_of_probs = sum(probabilities)
        log.debug(*get_debug_log("Probabilities", probabilities))
        log.debug("Sum of Probabilities = %s", sum_of_probs)
        probabilities = probabilities / sum_of_probs
        new_indices = np.random.choice(np.arange(B), b_over_N, replace=False, p=probabilities)
        log.debug("Any repeated indices? %s", any([i in indices for i in new_indices]))
        log.debug("Number of unique new indices? %s", len(set(new_indices)))
        all_sum_of_probs.append(sum_of_probs)

        # Expand the array of sampled indices
        indices = np.concatenate((indices, new_indices))
        log.debug("Indices (%s)", len(indices))

    log.debug("All sums %s", all_sum_of_probs)
    if plotting:
        fig, ax = plt.subplots()
        ax.plot(range(n-1), all_sum_of_probs)
        ax.set_xlabel("Joint batch selection iteration")
        plt.show()
        plt.close(fig)

    return indices # Gather the n chunks of size b/n and return mini-batch of size b

#----------------------------------------------------------------------------
# Train a 2D toy model with the given parameters.

@logs.errors
def do_train(
    classes='A', num_layers=4, hidden_dim=64, batch_size=4<<10, total_iter=4<<10, seed=0,
    P_mean=-2.3, P_std=1.5, sigma_data=0.5, lr_ref=1e-2, lr_iter=512, ema_std=0.010,
    guidance=False, guidance_weight=3, guide_path=None, guide_interpolation=False,
    validation=False, val_batch_size=4<<7, sigma_max=5,
    testing=False, n_test_samples=4<<8, test_batch_size=4<<8, 
    test_outer=False, test_mandala=False,
    acid=False, acid_N=16, acid_f=0.8, acid_diff=True, acid_inverted=False, 
    acid_stability_trick=False, acid_late=False, acid_early=False,
    device=torch.device('cuda'),
    pkl_pattern=None, pkl_iter=256, 
    viz_iter=32, viz_save=True, 
    verbosity=0, log_filename=None,
):

    # Set up the logger
    logging_to_file = log_filename is not None
    def set_up_logger(verbosity, log_filename=None):
        log_level = logs.get_log_level(verbosity)
        if logging_to_file:
            logs.set_log_file(log, log_filename)
            logs.set_log_format(log, color=False)
        logs.set_log_level(log_level, log)
    set_up_logger(verbosity, log_filename)

    # Set random seed, if specified
    if seed is not None:
        log.info("Seed = %s", seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device).manual_seed(seed)
        np.random.seed(seed)
    else: generator = torch.Generator(device)
    
    # Log basic parameters
    log.info("Device = %s", device)
    log.info("Number of training epochs = %s", total_iter)
    log.info("Training batch size = %s", batch_size)
    log.info("Number of training samples = %s", total_iter*batch_size)
    log.info("Number of validation samples = %s", val_batch_size)
    log.info("Number of test samples = %s", n_test_samples)
    log.info("Test batch size = %s", test_batch_size)

    # Log ACID parameters
    log.info("ACID = %s", acid)
    if acid:
        if acid_late:
            run_acid = False
            is_acid_waiting = True
            log.info("ACID with delayed execution strategy")
        elif acid_early:
            run_acid = True
            is_acid_waiting = True
            log.info("ACID programmed stop strategy")
        else:
            run_acid = True
            is_acid_waiting = False
            log.info("ACID with early-start execution strategy")
        log.info("ACID's Number of Chunks = %s", acid_N)
        log.info("ACID's Filter Ratio = %s", acid_f)
        acid_b_over_N = int(batch_size * (1 - acid_f) / acid_N) # Size of a mini-batch chunk
        acid_batch_size = acid_N * acid_b_over_N # Size of a mini-batch
        log.info("ACID's Mini-Batch Size = %s", acid_batch_size)
        log.info("ACID's Learnability = %s", acid_diff)
        log.info("ACID's Scores Inverted = %s", acid_inverted)
        log.info("ACID's Numeric Stability Trick = %s", acid_stability_trick)
    else: 
        run_acid = False
        is_acid_waiting = False

    # Log other parameters
    if guidance and guide_interpolation:
        log.warning("Guide interpolation of scores during training")
    else:
        guide_interpolation = False
        log.warning("No guide interpolation of scores during training")
    log.info("Guide interpolation = %s", guide_interpolation)

    # Basic configuration
    plotting_checkpoints = viz_iter is not None
    saving_checkpoints = pkl_pattern is not None
    saving_checkpoint_plots = plotting_checkpoints and saving_checkpoints and viz_save
    if saving_checkpoints:
        if not os.path.isdir(os.path.split(pkl_pattern)[0]):
            os.makedirs(os.path.split(pkl_pattern)[0])
        log.warning(f'Will save checkpoints to {os.path.split(pkl_pattern)[0]}')
        pkl_pattern_learner = pkl_pattern[:-4]+"learner.pkl"
        plt_pattern = pkl_pattern[:-4]+".jpg"
        if saving_checkpoint_plots:
            log.warning(f'Will save snapshots to {os.path.split(plt_pattern)[0]}')

    # Initialize models
    gtd = gt(classes, device)[0]
    net = ToyModel(num_layers=num_layers, hidden_dim=hidden_dim, sigma_data=sigma_data).to(device).train().requires_grad_(True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), betas=(0.9, 0.99))
    log.info("Guidance = %s", guidance)
    if guidance:
        log.info("Guidance weight = %s", guidance_weight)
        if guide_path is not None:
            with builtins.open(guide_path, "rb") as f:
                guide = pickle.load(f).to(device)
            set_up_logger(verbosity, log_filename) # No idea why, but builtins.open or pickle.load break the logger's setup
            log.warning("Guide model loaded from %s", guide_path)
        else:
            raise ValueError("No guide model checkpoint path specified")
    else: guide = None
    if guidance and acid:
        ref = guide
        log.warning("Guide model assigned as ACID reference")
    elif acid: 
        ref = ema
        log.warning("EMA assigned as ACID reference")
    else: ref = None
    if ref is not None: 
        net_beats_ref = False # Assume reference is always better than the model at first

    # Initialize plot.
    if viz_iter is not None:
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['figure.subplot.left'] = plt.rcParams['figure.subplot.bottom'] = 0
        plt.rcParams['figure.subplot.right'] = plt.rcParams['figure.subplot.top'] = 1
        fig, ax = plt.subplots(figsize=[12, 12], dpi=75)
        do_plot(ema, elems={'gt_uncond', 'gt_outline'}, ax=ax, device=device)
        plt.gcf().canvas.flush_events()
        if saving_checkpoint_plots:
            plt_path = plt_pattern % 0
            plt.savefig(plt_path)

    # Training loop.
    progress_bar = tqdm.tqdm(range(total_iter))
    for iter_idx in progress_bar:
        if logging_to_file: log.info("Iteration = %i", iter_idx)

        # Run forward-pass
        opt.param_groups[0]['lr'] = lr_ref / np.sqrt(max(iter_idx / lr_iter, 1))
        opt.zero_grad()
        sigma = (torch.randn(batch_size, device=device) * P_std + P_mean).exp()
        samples = gtd.sample(batch_size, sigma)
        gt_scores = gtd.score(samples, sigma)
        net_scores = net.score(samples, sigma, graph=True)
        if run_acid or is_acid_waiting: ref_scores = ref.score(samples, sigma)
        if guide_interpolation: interpol_scores = guide.score(samples, sigma).lerp(net_scores, guidance_weight)

        # Calculate teacher and student loss
        net_loss = (sigma ** 2) * ((gt_scores - net_scores) ** 2).mean(-1)
        if run_acid or is_acid_waiting:
            ref_loss = (sigma ** 2) * ((gt_scores - ref_scores) ** 2).mean(-1)
            if not net_beats_ref and net_loss.mean() < ref_loss.mean():
                net_beats_ref = True
                log.warning("Network has beaten the reference")
                run_acid = not(run_acid)
                if run_acid: log.warning("ACID will now be run")
                else: log.warning("ACID will now be stopped")
        if run_acid: 
            if guide_interpolation: 
                acid_loss = (sigma ** 2) * ((gt_scores - interpol_scores) ** 2).mean(-1)
            else: acid_loss = net_loss

        # Calculate overall loss
        if run_acid:
            try:
                log.warning("Using ACID")
                log.info("Average Super-Batch Learner Loss = %s", float(acid_loss.mean()))
                log.info("Average Super-Batch Reference Loss = %s", float(ref_loss.mean()))
                indices = jointly_sample_batch(acid_loss, ref_loss, 
                    N=acid_N, filter_ratio=acid_f,
                    learnability=acid_diff, inverted=acid_inverted,
                    numeric_stability_trick=acid_stability_trick)
                loss = acid_loss[indices] # Use indices of the ACID mini-batch
            except ValueError:
                log.warning("ACID has crashed, so it has been deactivated")
                loss = net_loss
                run_acid = False
                is_acid_waiting = False
        else:
            loss = net_loss

        # Backpropagate either on the full batch or only on ACID mini-batch
        log.info("Average Learner Loss = %s", float(loss.mean()))
        loss.mean().backward()

        # Update learner parameters
        params = dict({"Old_params."+k: p for k, p in net.named_parameters() if p.requires_grad})
        for k,v in params.items(): 
            try: log.debug(*get_debug_log(k, v))
            except: log.debug("%s = %s", k, float(v))
        opt.step()
        new_params = dict({"New_params."+k: p for k, p in net.named_parameters() if p.requires_grad})
        for k,v in new_params.items(): 
            try: log.debug(*get_debug_log(k, v))
            except: log.debug("%s = %s", k, float(v))

        # Update reference EMA parameters
        beta = training.phema.power_function_beta(std=ema_std, t_next=iter_idx+1, t_delta=1)
        for p_net, p_ema in zip(net.parameters(), ema.parameters()):
            p_ema.lerp_(p_net.detach(), 1 - beta)

        # Evaluate average loss and L2 metric on validation batch
        if validation:
            val_results = run_test(net, ema, guide, ref, acid=run_acid,
                classes=classes, P_mean=P_mean, P_std=P_std, sigma_max=sigma_max,
                n_samples=val_batch_size, batch_size=val_batch_size, 
                test_outer=test_outer, test_mandala=test_mandala,
                guidance_weight=guidance_weight,
                generator=generator, device=device)

            # Log validation loss
            log.info("Average Validation Learner Loss = %s", val_results["learner_loss"])
            log.info("Average Validation EMA Loss = %s", val_results["ema_loss"])
            if guidance: log.info("Average Validation Guide Loss = %s", val_results["guide_loss"])
            if run_acid: log.info("Average Validation ACID Reference Loss = %s", val_results["ref_loss"])

            # Log validation L2 metric
            log.info("Average Validation Learner L2 Distance = %s", val_results["learner_L2_metric"])
            log.info("Average Validation EMA L2 Distance = %s", val_results["ema_L2_metric"])
            if guidance:
                log.info("Average Validation Guided Learner L2 Distance = %s", val_results["learner_guided_L2_metric"])
                log.info("Average Validation Guided EMA L2 Distance = %s", val_results["ema_guided_L2_metric"])

            # Log validation metrics in outer branches of the distribution
            if test_outer:
                log.info("Average Outer Validation Learner Loss = %s", val_results["learner_out_loss"])
                log.info("Average Outer Validation EMA Loss = %s", val_results["ema_out_loss"])
                if guidance: log.info("Average Outer Validation Guide Loss = %s", val_results["guide_out_loss"])
                if run_acid: log.info("Average Outer Validation ACID Reference Loss = %s", val_results["ref_out_loss"])
                log.info("Average Outer Validation Learner L2 Distance = %s", val_results["learner_out_L2_metric"])
                log.info("Average Outer Validation EMA L2 Distance = %s", val_results["ema_out_L2_metric"])
                if guidance:
                    log.info("Average Outer Validation Guided Learner L2 Distance = %s", val_results["learner_guided_out_L2_metric"])
                    log.info("Average Outer Validation Guided EMA L2 Distance = %s", val_results["ema_guided_out_L2_metric"])
            
            # Log validation mandala score
            if test_mandala:
                log.info("Validation Learner Mandala Score = %s", val_results["learner_mandala_score"])
                log.info("Validation EMA Mandala Score = %s", val_results["ema_mandala_score"])
                log.info("Validation Learner Classification Score = %s", val_results["learner_classification_score"])
                log.info("Validation EMA Classification Score = %s", val_results["ema_classification_score"])
                if guidance:
                    log.info("Validation Guided Learner Mandala Score = %s", val_results["learner_guided_mandala_score"])
                    log.info("Validation Guided EMA Mandala Score = %s", val_results["ema_guided_mandala_score"])
                    log.info("Validation Guided Learner Classification Score = %s", val_results["learner_guided_classification_score"])
                    log.info("Validation Guided EMA Classification Score = %s", val_results["ema_guided_classification_score"])

        # Visualize resulting sample distribution.
        if plotting_checkpoints and iter_idx % viz_iter == 0:
            for x in plt.gca().lines: x.remove()
            do_plot(ema, elems={'samples'}, sigma_max=sigma_max, ax=ax, device=device)
            plt.gcf().canvas.flush_events()

        # Save model snapshot.
        if saving_checkpoints and (iter_idx + 1) % pkl_iter == 0:
            pkl_path = pkl_pattern % (iter_idx + 1)
            with open(pkl_path, 'wb') as f:
                pickle.dump(copy.deepcopy(ema).cpu(), f)
            if saving_checkpoint_plots:
                plt_path = plt_pattern % (iter_idx + 1)
                plt.savefig(plt_path)
    
    log.info("Finished training")
    
    # Evaluate average loss and L2 metric on test data
    if testing:
        test_results = run_test(net, ema, guide, ref, acid=acid,
            classes=classes, P_mean=P_mean, P_std=P_std, sigma_max=sigma_max,
            n_samples=n_test_samples, batch_size=test_batch_size, test_outer=test_outer,
            guidance_weight=guidance_weight,
            generator=generator, device=device)

        # Log test loss
        log.warning("Average Test Learner Loss = %s", test_results["learner_loss"])
        log.warning("Average Test EMA Loss = %s", test_results["ema_loss"])
        if guidance: log.warning("Average Test Guide Loss = %s", test_results["guide_loss"])
        if acid: log.warning("Average Test ACID Reference Loss = %s", test_results["ref_loss"])

        # Log test L2 metric
        log.warning("Average Test Learner L2 Distance = %s", test_results["learner_L2_metric"])
        log.warning("Average Test EMA L2 Distance = %s", test_results["ema_L2_metric"])
        if guidance:
            log.warning("Average Test Guided Learner L2 Distance = %s", test_results["learner_guided_L2_metric"])
            log.warning("Average Test Guided EMA L2 Distance = %s", test_results["ema_guided_L2_metric"])

        # Log metrics on outer branches of the distribution
        if test_outer:
            log.warning("Average Outer Test Learner Loss = %s", test_results["learner_out_loss"])
            log.warning("Average Outer Test EMA Loss = %s", test_results["ema_out_loss"])
            if guidance: log.warning("Average Outer Test Guide Loss = %s", test_results["guide_out_loss"])
            if acid: log.warning("Average Outer Test ACID Reference Loss = %s", test_results["ref_out_loss"])
            log.warning("Average Outer Test Learner L2 Distance = %s", test_results["learner_out_L2_metric"])
            log.warning("Average Outer Test EMA L2 Distance = %s", test_results["ema_out_L2_metric"])
            if guidance:
                log.warning("Average Outer Test Guided Learner L2 Distance = %s", test_results["learner_guided_out_L2_metric"])
                log.warning("Average Outer Test Guided EMA L2 Distance = %s", test_results["ema_guided_out_L2_metric"])

        # Log test mandala score
        if test_mandala:
            log.info("Test Learner Mandala Score = %s", test_results["learner_mandala_score"])
            log.info("Test EMA Mandala Score = %s", test_results["ema_mandala_score"])
            log.info("Test Learner Classification Score = %s", test_results["learner_classification_score"])
            log.info("Test EMA Classification Score = %s", test_results["ema_classification_score"])
            if guidance:
                log.info("Test Guided Learner Mandala Score = %s", test_results["learner_guided_mandala_score"])
                log.info("Test Guided EMA Mandala Score = %s", test_results["ema_guided_mandala_score"])
                log.info("Test Guided Learner Classification Score = %s", test_results["learner_guided_classification_score"])
                log.info("Test Guided EMA Classification Score = %s", test_results["ema_guided_classification_score"])

    # Save and visualize last iteration
    if saving_checkpoints:
        pkl_path = pkl_pattern % (iter_idx + 1)
        with open(pkl_path, 'wb') as f:
            pickle.dump(copy.deepcopy(ema).cpu(), f)
        pkl_path_learner = pkl_pattern_learner % (iter_idx + 1)
        with open(pkl_path_learner, 'wb') as f:
            pickle.dump(copy.deepcopy(net).cpu(), f)
        for x in plt.gca().lines: x.remove()
        do_plot(ema, elems={'samples'}, ax=ax, sigma_max=sigma_max, device=device)
        plt.gcf().canvas.flush_events()
        if saving_checkpoint_plots:
            plt_path = plt_pattern % (iter_idx + 1)
            plt.savefig(plt_path)

    if logging_to_file and verbosity>=1: 
        results = extract_results_from_log(log_filename)
        if saving_checkpoint_plots and logging_to_file: 
            loss_plt_path = log_filename.replace("log", "plot").replace(".txt", ".png")
            plot_loss(results, loss_plt_path);

#----------------------------------------------------------------------------
# Custom helper functions

def extract_results_from_log(log_path):

    results = {}
    super_learner_loss = []
    super_ref_loss = []
    learner_loss = []
    learner_val_loss = []
    ema_val_loss = []
    guide_val_loss = []
    ref_val_loss = []
    learner_out_val_loss = []
    ema_out_val_loss = []
    guide_out_val_loss = []
    ref_out_val_loss = []
    ema_L2_val_metric = []
    ema_guided_L2_val_metric = []
    L2_val_metric = []
    guided_L2_val_metric = []
    ema_out_L2_val_metric = []
    ema_guided_out_L2_val_metric = []
    out_L2_val_metric = []
    guided_out_L2_val_metric = []
    learner_mandala_score = []
    learner_classification_score = []
    learner_guided_mandala_score = []
    learner_guided_classification_score = []
    ema_mandala_score = []
    ema_classification_score = []
    ema_guided_mandala_score = []
    ema_guided_classification_score = []
    with builtins.open(log_path, "r") as f:
        for line in f:
            if "Average" in line:
                if "Average Learner Loss" in line: learner_loss.append(vtext.find_numbers(line)[-1]); continue
                elif "Super-Batch" in line:
                    if "Average Super-Batch Learner Loss" in line: super_learner_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Super-Batch Reference Loss" in line: super_ref_loss.append(vtext.find_numbers(line)[-1]); continue
                elif "Outer Validation" in line:
                    if "Average Outer Validation Learner Loss" in line: learner_out_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Outer Validation EMA Loss" in line: ema_out_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Outer Validation Guide Loss" in line: guide_out_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Outer Test ACID Reference Loss" in line: ref_out_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Outer Validation EMA L2 Distance" in line: ema_out_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Outer Validation Guided EMA L2 Distance" in line: ema_guided_out_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Outer Validation Learner L2 Distance" in line: out_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Outer Validation Guided Learner L2 Distance" in line: guided_out_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                elif "Outer Test" in line:
                    if "Average Outer Test Learner Loss" in line: results["learner_out_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Outer Test EMA Loss" in line: results["ema_out_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Outer Test Guide Loss" in line: results["guide_out_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Outer Test ACID Reference Loss" in line: results["ref_out_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Outer Test EMA L2 Distance With Guidance" in line: results["ema_guided_out_L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Outer Test EMA L2 Distance" in line: results["ema_out_L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Outer Test Learner L2 Distance With Guidance" in line: results["guided_out_L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Outer Test Learner L2 Distance" in line: results["out_L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
                elif "Validation" in line:
                    if "Average Validation Learner Loss" in line: learner_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation EMA Loss" in line: ema_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation Guide Loss" in line: guide_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation ACID Reference Loss" in line: ref_val_loss.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation EMA L2 Distance" in line: ema_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation Guided EMA L2 Distance" in line: ema_guided_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation Learner L2 Distance" in line: L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation Guided Learner L2 Distance" in line: guided_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue
                    elif "Average Validation L2 Distance" in line: ema_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue # Backwards compatibility
                    elif "Average Validation L2 Distance with Guidance" in line: ema_guided_L2_val_metric.append(vtext.find_numbers(line)[-1]); continue # Backwards compatibility
                elif "Test" in line:
                    if "Average Test Learner Loss" in line: results["learner_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Test EMA Loss" in line: results["ema_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Test Guide Loss" in line: results["guide_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Test ACID Reference Loss" in line: results["ref_test_loss"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Test EMA L2 Distance With Guidance" in line: results["ema_guided_L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Test EMA L2 Distance" in line: results["ema_L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Test Learner L2 Distance With Guidance" in line: results["guided_L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
                    elif "Average Test Learner L2 Distance" in line: results["L2_test_metric"] = vtext.find_numbers(line)[-1]; continue
            elif "Score" in line:
                if "Validation" in line:
                    if "Validation Learner Mandala Score" in line: learner_mandala_score.append(vtext.find_numbers(line)[-1]); continue
                    elif "Validation EMA Mandala Score" in line: ema_mandala_score.append(vtext.find_numbers(line)[-1]); continue
                    elif "Validation Learner Classification Score" in line: learner_classification_score.append(vtext.find_numbers(line)[-1]); continue
                    elif "Validation EMA Classification Score" in line: ema_classification_score.append(vtext.find_numbers(line)[-1]); continue
                    elif "Validation Guided Learner Mandala Score" in line: learner_guided_mandala_score.append(vtext.find_numbers(line)[-1]); continue
                    elif "Validation Guided EMA Mandala Score" in line: ema_guided_mandala_score.append(vtext.find_numbers(line)[-1]); continue
                    elif "Validation Guided Learner Classification Score" in line: learner_guided_classification_score.append(vtext.find_numbers(line)[-1]); continue
                    elif "Validation Guided EMA Classification Score" in line: ema_guided_classification_score.append(vtext.find_numbers(line)[-1]); continue
                elif "Test" in line:
                    if "Test Learner Mandala Score" in line: results["learner_test_mandala_score"] = vtext.find_numbers(line)[-1]; continue
                    elif "Test EMA Mandala Score" in line: results["ema_test_mandala_score"] = vtext.find_numbers(line)[-1]; continue
                    elif "Test Learner Classification Score" in line: results["learner_test_classification_score"] = vtext.find_numbers(line)[-1]; continue
                    elif "Test EMA Classification Score" in line: results["ema_test_classification_score"] = vtext.find_numbers(line)[-1]; continue
                    elif "Test Guided Learner Mandala Score" in line: results["learner_guided_test_mandala_score"] = vtext.find_numbers(line)[-1]; continue
                    elif "Test Guided EMA Mandala Score" in line: results["ema_guided_test_mandala_score"] = vtext.find_numbers(line)[-1]; continue
                    elif "Test Guided Learner Classification Score" in line: results["learner_guided_test_classification_score"] = vtext.find_numbers(line)[-1]; continue
                    elif "Test Guided EMA Classification Score" in line: results["ema_guided_test_classification_score"] = vtext.find_numbers(line)[-1]; continue
            else: pass
    
    results["super_learner_loss"] = super_learner_loss
    results["super_ref_loss"] = super_ref_loss
    results["learner_loss"] = learner_loss
    results["learner_val_loss"] = learner_val_loss
    results["ema_val_loss"] = ema_val_loss
    results["guide_val_loss"] = guide_val_loss
    results["ref_val_loss"] = ref_val_loss
    results["ema_L2_val_metric"] = ema_L2_val_metric
    results["ema_guided_L2_val_metric"] = ema_guided_L2_val_metric
    results["L2_val_metric"] = L2_val_metric
    results["guided_L2_val_metric"] = guided_L2_val_metric
    
    results["learner_out_val_loss"] = learner_out_val_loss
    results["ema_out_val_loss"] = ema_out_val_loss
    results["guide_out_val_loss"] = guide_out_val_loss
    results["ref_out_val_loss"] = ref_out_val_loss
    results["ema_out_L2_val_metric"] = ema_out_L2_val_metric
    results["ema_guided_out_L2_val_metric"] = ema_guided_out_L2_val_metric
    results["out_L2_val_metric"] = out_L2_val_metric
    results["guided_out_L2_val_metric"] = guided_out_L2_val_metric

    results["learner_mandala_score"] = learner_mandala_score
    results["ema_mandala_score"] = ema_mandala_score
    results["learner_classification_score"] = learner_classification_score
    results["ema_classification_score"] = ema_classification_score
    results["learner_guided_mandala_score"] = learner_guided_mandala_score
    results["ema_guided_mandala_score"] = ema_guided_mandala_score
    results["learner_guided_classification_score"] = learner_guided_classification_score
    results["ema_guided_classification_score"] = ema_guided_classification_score

    return results

def plot_loss(loss_dict, fig_path=None):

    figs = []
    if fig_path is not None:
        fig_path_base, fig_extension = os.path.splitext(fig_path)

    # Basic plot
    def plot_training_loss():
        fig, axes = plt.subplots(nrows=2, gridspec_kw=dict(hspace=0))
        axes[0].plot(loss_dict["learner_loss"], "C0", label="Training Loss", alpha=0.8, linewidth=2)
        if len(loss_dict["super_ref_loss"])>0: 
            axes[0].plot(loss_dict["super_ref_loss"], "C3", label="ACID Ref Loss", alpha=1, linewidth=1)
            axes[0].plot(loss_dict["super_learner_loss"], "k", label="ACID Learner Loss", alpha=1, linewidth=0.5)
        axes[1].set_xlabel("Epoch")
        axes[0].set_ylabel("Average Training Loss")
        axes[0].legend()
        for ax in axes: ax.grid()
        return fig, axes
    
    # First, plot validation loss values
    fig_1, axes_1 = plot_training_loss()
    if len(loss_dict["learner_val_loss"])>0:
        if len(loss_dict["ref_val_loss"])>0:
            axes_1[1].plot(loss_dict["ref_val_loss"], "C3", label="Ref Val Loss", alpha=1, linewidth=0.5)
        axes_1[1].plot(loss_dict["learner_val_loss"], "k", label="Learner Val Loss", alpha=1.0, linewidth=0.5)
        axes_1[1].plot(loss_dict["ema_val_loss"], color="m", label="EMA Val Loss", alpha=0.35, linewidth=3)
        if len(loss_dict["guide_val_loss"])>0:
            axes_1[1].plot(loss_dict["guide_val_loss"], color="r", label="Guide Val Loss", alpha=0.25, linewidth=2)
    axes_1[1].set_ylabel("Average Validation Loss")
    axes_1[1].legend()
    plt.tight_layout()
    plt.savefig(fig_path_base+"_1"+fig_extension)
    figs.append(fig_1)
        
    # Then, plot validation L2 distance values
    fig_2, axes_2 = plot_training_loss()
    if len(loss_dict["learner_val_loss"])>0:
        axes_2[1].plot(loss_dict["ema_L2_val_metric"], "-.", color="navy", label="EMA L2 Val Metric", alpha=1, linewidth=1)
        if len(loss_dict["L2_val_metric"])>0:
            axes_2[1].plot(loss_dict["L2_val_metric"], "-", color="blue", label="Learner L2 Val Metric", alpha=0.35, linewidth=3)
        if len(loss_dict["ema_guided_L2_val_metric"])>0:
            axes_2[1].plot(loss_dict["ema_guided_L2_val_metric"], "--", color="deeppink", label="Guided EMA L2 Val Metric", alpha=1, linewidth=1)
        if len(loss_dict["guided_L2_val_metric"])>0:
            axes_2[1].plot(loss_dict["guided_L2_val_metric"], "-", color="mediumvioletred", label="Guided Learner L2 Val Metric", 
                           alpha=0.35, linewidth=3)
    axes_2[1].set_ylabel("Average Validation L2 Distance")
    axes_2[1].legend()
    plt.tight_layout()
    plt.savefig(fig_path_base+"_2"+fig_extension)
    figs.append(fig_2)

    # Also compare validation loss on the outer branches vs the whole distribution
    if len(loss_dict["learner_out_val_loss"])>0:
        fig_3, axes_3 = plot_training_loss()
        axes_3[1].plot(loss_dict["learner_val_loss"], "k", label="Learner Val Loss", alpha=1.0, linewidth=0.5)
        axes_3[1].plot(loss_dict["ema_val_loss"], color="m", label="EMA Val Loss", alpha=0.35, linewidth=3)
        axes_3[1].plot(loss_dict["learner_out_val_loss"], "k", label="Learner Out Val Loss", alpha=1.0, linewidth=0.5, linestyle="dashed")
        axes_3[1].plot(loss_dict["ema_out_val_loss"], color="orange", label="EMA Out Val Loss", alpha=0.35, linewidth=3)
        axes_3[1].set_ylabel("Average Validation Loss")
        axes_3[1].legend(loc="upper right", ncols=2)
        plt.tight_layout()
        plt.savefig(fig_path_base+"_3"+fig_extension)
        figs.append(fig_3)

    # Finally, compare validation L2 metrics on the outer branches vs the whole distribution
    if len(loss_dict["ema_out_L2_val_metric"])>0:
        fig_4, axes_4 = plot_training_loss()
        axes_4[1].plot(loss_dict["ema_L2_val_metric"], "-", color="navy", label="EMA L2 Val Metric", alpha=1, linewidth=1)
        axes_4[1].plot(loss_dict["L2_val_metric"], "-", color="blue", label="Learner L2 Val Metric", alpha=0.35, linewidth=3)
        axes_4[1].plot(loss_dict["ema_out_L2_val_metric"], "-.", color="firebrick", label="EMA Out L2 Val Metric", alpha=1, linewidth=1)
        axes_4[1].plot(loss_dict["out_L2_val_metric"], "-", color="salmon", label="Learner Out L2 Val Metric", alpha=0.45, linewidth=3)
        axes_4[1].set_ylabel("Average Validation L2 Distance")
        axes_4[1].legend(loc="upper right", ncols=2)
        plt.tight_layout()
        plt.savefig(fig_path_base+"_4"+fig_extension)
        figs.append(fig_4)

        fig_5, axes_5 = plot_training_loss()
        axes_5[1].plot(loss_dict["ema_guided_L2_val_metric"], "--", color="deeppink", label="Guided EMA L2 Val Metric", alpha=1, linewidth=1)
        axes_5[1].plot(loss_dict["guided_L2_val_metric"], "-", color="mediumvioletred", label="Guided Learner L2 Val Metric", 
                        alpha=0.35, linewidth=3)
        axes_5[1].plot(loss_dict["ema_guided_out_L2_val_metric"], "--", color="teal", label="Guided EMA Out L2 Val Metric", alpha=1, linewidth=1)
        axes_5[1].plot(loss_dict["guided_out_L2_val_metric"], "-", color="lightseagreen", label="Guided Learner Out L2 Val Metric", 
                        alpha=0.45, linewidth=3)
        axes_5[1].set_ylabel("Average Validation L2 Distance")
        axes_5[1].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(fig_path_base+"_5"+fig_extension)
        figs.append(fig_5)
    
    return figs

#----------------------------------------------------------------------------
# Run test

def run_test(net, ema=None, guide=None, ref=None, acid=False,
             classes='A', P_mean=-2.3, P_std=1.5, sigma_max=5, depth_sep=5,
             n_samples=4<<8, batch_size=4<<8, test_outer=False, test_mandala=False,
             guidance_weight=3,
             generator=None,
             device=torch.device('cuda'),
             logging=False):

    # Basic configuration
    test_ema = ema is not None
    test_guide = guide is not None
    if acid: test_ref = ref is not None
    else: test_ref = False

    # Ground truth distribution
    gtd, gtcomps = gt(classes, device)
    outcomps = [c for c in gtcomps if c.depth>=depth_sep]
    outd = GaussianMixture([c.phi for c in outcomps], 
                           [c.mu for c in outcomps], 
                           [c.Sigma for c in outcomps]).to(device)
    
    # Basic loop configuration
    n_epochs = n_samples//batch_size
    n_remainder = n_samples - n_epochs*batch_size
    if n_remainder > 0: n_epochs += 1    
    results = {"learner_loss":0, "learner_L2_metric":0}
    if test_ema: 
        results["ema_loss"] = 0
        results["ema_L2_metric"] = 0
    if test_guide: 
        results["guide_loss"] = 0
        results["learner_guided_L2_metric"] = 0
    if test_ema and test_guide: 
        results["ema_guided_L2_metric"] = 0
    if test_ref: 
        results["ref_loss"] = 0
    if test_outer:
        results["learner_out_loss"] = 0
        results["learner_out_L2_metric"] = 0
        if test_ema:
            results["ema_out_loss"] = 0
            results["ema_out_L2_metric"] = 0
        if test_guide:
            results["guide_out_loss"] = 0
            results["learner_guided_out_L2_metric"] = 0
        if test_ema and test_guide:
            results["ema_guided_out_L2_metric"] = 0
        if test_ref:
            results["ref_out_loss"] = 0

    # Test loop
    if logging: progress_bar = tqdm.tqdm(range(n_epochs))
    else: progress_bar = range(n_epochs)
    for i_epoch in progress_bar:
        if logging: log.info("Iteration = %i", i_epoch)

        # Number of samples for this batch
        n_i_samples = min(n_samples - i_epoch*batch_size, batch_size)

        # Sample as in training
        test_sigma = (torch.randn(n_i_samples, device=device) * P_std + P_mean).exp()
        test_samples = gtd.sample(n_i_samples, test_sigma, generator=generator)

        # Also sample from the outer branches of the distribution
        if test_outer: out_test_samples = outd.sample(n_i_samples, test_sigma, generator=generator)

        # Evaluate scores
        gt_test_scores = gtd.score(test_samples, test_sigma)
        net_test_scores = net.score(test_samples, test_sigma)
        if test_ema: 
            ema_test_scores = ema.score(test_samples, test_sigma)
        if test_guide: 
            guide_test_scores = guide.score(test_samples, test_sigma)
        if test_ref and not test_guide and not test_ema:
            ref_test_scores = ref.score(test_samples, test_sigma)
        if test_outer:
            gt_out_test_scores = outd.score(out_test_samples, test_sigma)
            net_out_test_scores = net.score(out_test_samples, test_sigma)
            if test_ema: 
                ema_out_test_scores = ema.score(out_test_samples, test_sigma)
            if test_guide: 
                guide_out_test_scores = guide.score(out_test_samples, test_sigma)
            if test_ref and not test_guide and not test_ema: 
                ref_out_test_scores = ref.score(out_test_samples, test_sigma)

        # Evaluate loss
        net_test_loss = (test_sigma ** 2) * ((gt_test_scores - net_test_scores) ** 2).mean(-1)
        results["learner_loss"] += float(net_test_loss.mean())/n_epochs
        if test_ema:
            ema_test_loss = (test_sigma ** 2) * ((gt_test_scores - ema_test_scores) ** 2).mean(-1)
            results["ema_loss"] += float(ema_test_loss.mean())/n_epochs
        if test_guide: 
            guide_test_loss = (test_sigma ** 2) * ((gt_test_scores - guide_test_scores) ** 2).mean(-1)
            results["guide_loss"] += float(guide_test_loss.mean())/n_epochs
        if test_ref:
            if test_guide: 
                results["ref_loss"] += results["guide_loss"]
            elif test_ema:
                results["ref_loss"] += results["ema_loss"]
            else:
                ref_test_loss = (test_sigma ** 2) * ((gt_test_scores - ref_test_scores) ** 2).mean(-1)
                results["ref_loss"] += float(ref_test_loss.mean())/n_epochs
        if test_outer:
            net_out_test_loss = (test_sigma ** 2) * ((gt_out_test_scores - net_out_test_scores) ** 2).mean(-1)
            results["learner_out_loss"] += float(net_out_test_loss.mean())/n_epochs
            if test_ema:
                ema_out_test_loss = (test_sigma ** 2) * ((gt_out_test_scores - ema_out_test_scores) ** 2).mean(-1)
                results["ema_out_loss"] += float(ema_out_test_loss.mean())/n_epochs
            if test_guide:
                guide_out_test_loss = (test_sigma ** 2) * ((gt_out_test_scores - guide_out_test_scores) ** 2).mean(-1)
                results["guide_out_loss"] += float(guide_out_test_loss.mean())/n_epochs
            if test_ref:
                if test_guide:
                    results["ref_out_loss"] += results["guide_out_loss"]
                elif test_ema:
                    results["ref_out_loss"] += results["ema_out_loss"]
                else:
                    ref_out_test_loss = (test_sigma ** 2) * ((gt_out_test_scores - ref_out_test_scores) ** 2).mean(-1)
                    results["ref_out_loss"] += float(ref_out_test_loss.mean())/n_epochs

        # Sample from pure Gaussian noise
        test_samples = gtd.sample(n_i_samples, sigma_max, generator=generator)

        # Sample also from the outer branches of the distribution
        if test_outer: out_test_samples = outd.sample(n_i_samples, sigma_max, generator=generator)

        # Create full EMA samples using net for guidance
        gt_test_outputs = do_sample(net=gtd, x_init=test_samples, sigma_max=sigma_max)[-1]
        if test_ema:
            ema_test_outputs = do_sample(net=ema, x_init=test_samples, sigma_max=sigma_max)[-1]
            results["ema_L2_metric"] += float(torch.sqrt(((ema_test_outputs - gt_test_outputs) ** 2).sum(-1)).mean())/n_epochs
            if test_guide:
                guided_test_outputs = do_sample(net=ema, x_init=test_samples, 
                                                guidance=guidance_weight, gnet=guide, sigma_max=sigma_max)[-1]
                results["ema_guided_L2_metric"] += float(torch.sqrt(((guided_test_outputs - gt_test_outputs) ** 2).sum(-1)).mean())/n_epochs
            if test_outer:
                gt_out_test_outputs = do_sample(net=outd, x_init=out_test_samples, sigma_max=sigma_max)[-1]
                ema_out_test_outputs = do_sample(net=ema, x_init=out_test_samples, sigma_max=sigma_max)[-1]
                results["ema_out_L2_metric"] += float(torch.sqrt(((ema_out_test_outputs - gt_out_test_outputs) ** 2).sum(-1)).mean())/n_epochs
                if test_guide:
                    guided_out_test_outputs = do_sample(net=ema, x_init=out_test_samples, 
                                                        guidance=guidance_weight, gnet=guide, sigma_max=sigma_max)[-1]
                    results["ema_guided_out_L2_metric"] += float(torch.sqrt(((guided_out_test_outputs - gt_out_test_outputs) ** 2).sum(-1)).mean())/n_epochs
            if test_mandala:
                ema_mandala, ema_classification = mandala_score(ema, gtd, 
                                                                samples=ema_test_outputs,
                                                                sigma_max=sigma_max)
                results["ema_mandala_score"] = ema_mandala
                results["ema_classification_score"] = ema_classification
                if test_guide:
                    ema_mandala, ema_classification = mandala_score(
                        ema, gtd, samples=ema_test_outputs, sigma_max=sigma_max, 
                        guide=guide, guidance_weight=guidance_weight)
                    results["ema_guided_mandala_score"] = ema_mandala
                    results["ema_guided_classification_score"] = ema_classification

        # Create full learner samples using net for guidance
        test_outputs = do_sample(net=net, x_init=test_samples, sigma_max=sigma_max)[-1]
        results["learner_L2_metric"] += float(torch.sqrt(((test_outputs - gt_test_outputs) ** 2).sum(-1)).mean())/n_epochs
        if test_guide:
            guided_test_outputs = do_sample(net=net, x_init=test_samples, 
                                            guidance=guidance_weight, gnet=guide, sigma_max=sigma_max)[-1]
            results["learner_guided_L2_metric"] += float(torch.sqrt(((guided_test_outputs - gt_test_outputs) ** 2).sum(-1)).mean())/n_epochs
        if test_outer:
            out_test_outputs = do_sample(net=net, x_init=out_test_samples, sigma_max=sigma_max)[-1]
            results["learner_out_L2_metric"] += float(torch.sqrt(((out_test_outputs - gt_out_test_outputs) ** 2).sum(-1)).mean())/n_epochs
            if test_guide:
                guided_out_test_outputs = do_sample(net=net, x_init=out_test_samples, 
                                                    guidance=guidance_weight, gnet=guide, sigma_max=sigma_max)[-1]
                results["learner_guided_out_L2_metric"] += float(torch.sqrt(((guided_out_test_outputs - gt_out_test_outputs) ** 2).sum(-1)).mean())/n_epochs
        if test_mandala:
            net_mandala, net_classification = mandala_score(net, gtd, 
                                                            samples=test_outputs,
                                                            sigma_max=sigma_max)
            results["learner_mandala_score"] = net_mandala
            results["learner_classification_score"] = net_classification
            if test_guide:
                net_mandala, net_classification = mandala_score(
                    net, gtd, samples=test_outputs, sigma_max=sigma_max, 
                    guide=guide, guidance_weight=guidance_weight)
                results["learner_guided_mandala_score"] = net_mandala
                results["learner_guided_classification_score"] = net_classification

    return results

#----------------------------------------------------------------------------
# To run test outside of training, loading the models first

def do_test(net_path, ema_path=None, guide_path=None, acid=False, 
            classes='A', P_mean=-2.3, P_std=1.5, sigma_max=5, depth_sep=5,
            n_samples=4<<8, batch_size=4<<8, test_outer=False, test_mandala=False,
            guidance_weight=3,
            seed=None, generator=None,
            device=torch.device('cuda'),
            log_filename=None):

    # Set up the logger
    log = logs.create_logger(net_path)
    logging_to_file = log_filename is not None
    def set_up_logger(log_filename=None):
        log_level = logs.get_log_level(1)
        if logging_to_file:
            logs.set_log_file(log, log_filename)
            logs.set_log_format(log, color=False)
        logs.set_log_level(log_level, log)
    set_up_logger(log_filename)
    
    # Set random seed, if specified
    if generator is None and seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device).manual_seed(seed)
        np.random.seed(seed)
        log.info("Seed = %s", seed)
    
    # Log basic parameters
    n_epochs = n_samples//batch_size
    n_remainder = n_samples - n_epochs*batch_size
    if n_remainder > 0: n_epochs += 1
    log.info("Number of test epochs = %s", n_epochs)
    log.info("Test batch size = %s", batch_size)
    log.info("Number of test samples = %s", n_samples)
    
    # Load models
    with builtins.open(net_path, "rb") as f:
        net = pickle.load(f).to(device)
    if ema_path is not None:
        with builtins.open(ema_path, "rb") as f:
            ema = pickle.load(f).to(device)
    else: ema = None
    if guide_path is not None:
        with builtins.open(guide_path, "rb") as f:
            guide = pickle.load(f).to(device)
    else: guide = None
    set_up_logger(log_filename)
    log.warning("Model loaded from %s", net_path)
    if ema_path is not None: log.warning("EMA model loaded from %s", ema_path)
    if guide_path is not None: 
        log.warning("Guide model loaded from %s", guide_path)
        log.info("Guidance weight = %s", guidance_weight)
    if acid and guide_path is not None:
        ref = guide; log.warning("Guide model assigned as ACID reference")
    elif acid: 
        ref = ema; log.warning("EMA assigned as ACID reference")
    else: ref = None

    # Basic configuration
    test_ema = ema_path is not None
    test_guide = guide_path is not None
    test_ref = ref is not None

    # Run test
    results = run_test(net, ema, guide, ref, acid=acid,
        classes=classes, P_mean=P_mean, P_std=P_std, sigma_max=sigma_max, depth_sep=depth_sep,
        n_samples=n_samples, batch_size=batch_size, 
        test_outer=test_outer, test_mandala=test_mandala,
        guidance_weight=guidance_weight,
        generator=generator,
        device=device, logging=True)

    # Log test loss
    log.info("Average Test Learner Loss = %s", results["learner_loss"])
    log.info("Average Test EMA Loss = %s", results["ema_loss"])
    try: log.info("Average Test Guide Loss = %s", results["guide_loss"])
    except UnboundLocalError or KeyError: pass
    if acid:
        try: log.info("Average Test Ref Loss = %s", results["ref_loss"])
        except UnboundLocalError or KeyError: pass

    # Log test loss on outer branches of the distribution
    log.info("Average Outer Test Learner Loss = %s", results["learner_out_loss"])
    log.info("Average Outer Test EMA Loss = %s", results["ema_out_loss"])
    try: log.info("Average Outer Test Guide Loss = %s", results["guide_out_loss"])
    except UnboundLocalError: pass
    if acid:
        try: log.info("Average Outer Test Ref Loss = %s", results["ref_out_loss"])
        except UnboundLocalError: pass

    # Log test L2 metric
    if test_ema: 
        log.info("Average Test EMA L2 Distance = %s", results["ema_L2_metric"])
        if test_guide: 
            log.info("Average Test Guided EMA L2 Distance = %s", results["ema_guided_L2_metric"])
    log.info("Average Test Learner L2 Distance = %s", results["learner_L2_metric"])
    if test_guide: 
        log.info("Average Test Guided Learner L2 Distance = %s", results["learner_guided_L2_metric"])

    # Log metrics on outer branches of the distribution
    if test_outer:
        log.info("Average Outer Test Learner Loss = %s", results["learner_out_loss"])
        if test_ema: log.info("Average Outer Test EMA Loss = %s", results["ema_out_loss"])
        if test_guide: log.info("Average Outer Test Guide Loss = %s", results["guide_out_loss"])
        if test_ref: log.info("Average Outer Test Ref Loss = %s", results["ref_out_loss"])
        if test_ema:
            log.info("Average Outer Test EMA L2 Distance = %s", results["ema_out_L2_metric"])
            if test_guide:
                log.info("Average Outer Test Guided EMA L2 Distance = %s", results["ema_guided_out_L2_metric"])
        log.info("Average Outer Test Learner L2 Distance = %s", results["learner_out_L2_metric"])
        if test_guide: log.info("Average Outer Test Guided Learner L2 Distance = %s", results["learner_guided_out_L2_metric"])

    return results

#----------------------------------------------------------------------------
# Simulate the EDM sampling ODE for the given set of initial sample points.
# Adapted from generate_images.py.

def do_sample(net, x_init, guidance=1, gnet=None, num_steps=32, sigma_min=0.002, sigma_max=5, rho=7):
    # Guided denoiser.
    def denoise(x, sigma):
        score = net.score(x, sigma)
        if gnet is not None:
            score = gnet.score(x, sigma).lerp(score, guidance)
        return x + score * (sigma ** 2)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_init.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_cur = x_init
    trajectory = [x_cur]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

        # Euler step.
        d_cur = (x_cur - denoise(x_cur, t_cur)) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        # Record trajectory.
        x_cur = x_next
        trajectory.append(x_cur)
    return torch.stack(trajectory) # From sigma_max (random noise), to 0 (data distribution)

#----------------------------------------------------------------------------
# Draw the given set of plot elements using matplotlib.

DEVICE = torch.device('cuda')
FIG1_KWARGS = dict(view_x=0.30, view_y=0.30, view_size=1.2, num_samples=1<<14, device=DEVICE)
FIG2_KWARGS = dict(view_x=0.45, view_y=1.22, view_size=0.3, num_samples=1<<12, device=DEVICE, sample_distance=0.045, sigma_max=0.03)
GT_LOGP_LEVEL = -2.12

def do_plot(
    net=None, guidance=1, gnet=None, elems={'gt_uncond', 'gt_outline', 'samples'},
    view_x=0, view_y=0, view_size=1.6, grid_resolution=400, arrow_len=0.002,
    num_samples=1<<13, seed=1, sample_distance=0, sigma_max=5, depth_sep=5,
    ax=None, device=torch.device('cuda'),
):
    if seed is not None: 
        generator = torch.Generator(device).manual_seed(seed)
    else: generator = torch.Generator(device)

    # Initialize ground truth distribution
    allgtd, allgtcomps = gt('AB', device)
    gtcomps = [c for c in allgtcomps if c.cls=="A"]
    gtd = GaussianMixture([c.phi for c in gtcomps], 
                          [c.mu for c in gtcomps], 
                          [c.Sigma for c in gtcomps]).to(device)
    alloutcomps = [c for c in allgtcomps if c.depth>=depth_sep]
    alloutd = GaussianMixture([c.phi for c in alloutcomps], 
                              [c.mu for c in alloutcomps], 
                              [c.Sigma for c in alloutcomps]).to(device)
    outcomps = [c for c in alloutcomps if c.cls=="A"]
    outd = GaussianMixture([c.phi for c in outcomps], 
                           [c.mu for c in outcomps], 
                           [c.Sigma for c in outcomps]).to(device)
    
    # Generate initial samples.
    if any(x.startswith(y) for x in elems for y in ['samples', 'trajectories', 'scores']):
        samples = gtd.sample(num_samples, sigma_max, generator=generator)
        if sample_distance > 0:
            ok = torch.ones(len(samples), dtype=torch.bool, device=device)
            for i in range(1, len(samples)):
                ok[i] = (samples[i] - samples[:i][ok[:i]]).square().sum(-1).sqrt().min() >= sample_distance
            samples = samples[ok]
    if any(x.startswith(y) for x in elems for y in ['out_samples', 'out_trajectories']):
        out_samples = outd.sample(num_samples, sigma_max, generator=generator)
        if sample_distance > 0:
            ok = torch.ones(len(out_samples), dtype=torch.bool, device=device)
            for i in range(1, len(out_samples)):
                ok[i] = (out_samples[i] - out_samples[:i][ok[:i]]).square().sum(-1).sqrt().min() >= sample_distance
            out_samples = out_samples[ok]
        out_gt_trajectories = do_sample(net=outd, x_init=out_samples, sigma_max=sigma_max)

    # Run sampler.
    if any(x.startswith(y) for x in elems for y in ['samples', 'trajectories']):
        trajectories = do_sample(net=(net or gtd), x_init=samples, guidance=guidance, gnet=gnet, sigma_max=sigma_max)
    if any(x.startswith(y) for x in elems for y in ['out_samples', 'out_trajectories']):
        out_trajectories = do_sample(net=(net or outd), x_init=out_samples, guidance=guidance, gnet=gnet, sigma_max=sigma_max)

    # Initialize plot.
    if ax is None: 
        fig, ax = plt.subplots()
    gridx = torch.linspace(view_x - view_size, view_x + view_size, steps=grid_resolution, device=device)
    gridy = torch.linspace(view_y - view_size, view_y + view_size, steps=grid_resolution, device=device)
    gridxy = torch.stack(torch.meshgrid(gridx, gridy, indexing='xy'), axis=-1)
    ax.set_xlim(float(gridx[0]), float(gridx[-1]))
    ax.set_ylim(float(gridy[0]), float(gridy[-1]))
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Plot helper functions.
    def contours(values, levels, colors=None, cmap=None, alpha=1, linecolors='black', linealpha=1, linewidth=2.5):
        values = -(values.max() - values).sqrt().cpu().numpy()
        ax.contourf(gridx.cpu().numpy(), gridy.cpu().numpy(), values, levels=levels, antialiased=True, extend='max', colors=colors, cmap=cmap, alpha=alpha)
        ax.contour(gridx.cpu().numpy(), gridy.cpu().numpy(), values, levels=levels, antialiased=True, colors=linecolors, alpha=linealpha, linestyles='solid', linewidths=linewidth)
    def lines(pos, color='black', alpha=1):
        ax.plot(*pos.cpu().numpy().T, '-', linewidth=5, solid_capstyle='butt', color=color, alpha=alpha)
    def arrows(pos, dir, color='black', alpha=1):
        ax.quiver(*pos.cpu().numpy().T, *dir.cpu().numpy().T * arrow_len, scale=0.6, width=5e-3, headwidth=4, headlength=3, headaxislength=2.5, capstyle='round', color=color, alpha=alpha)
    def points(pos, color='black', alpha=1, size=30):
        ax.plot(*pos.cpu().numpy().T, '.', markerfacecolor=color, markeredgecolor='none', color=color, alpha=alpha, markersize=size)

    # Draw requested plot elements.
    if 'p_net' in elems:            contours(net.logp(gridxy, sigma_max), levels=np.linspace(-2.5, 2.5, num=20)[1:-1], cmap='Greens', linealpha=0.2)
    if 'p_gnet' in elems:           contours(gnet.logp(gridxy, sigma_max), levels=np.linspace(-2.5, 3.5, num=20)[1:-1], cmap='Reds', linealpha=0.2)
    if 'p_ratio' in elems:          contours(net.logp(gridxy, sigma_max) - gnet.logp(gridxy, sigma_max), levels=np.linspace(-2.2, 1.0, num=20)[1:-1], cmap='Blues', linealpha=0.2)
    if 'gt_uncond' in elems:        contours(allgtd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=[[0.9,0.9,0.9]], linecolors=[[0.7,0.7,0.7]], linewidth=1.5)
    if 'gt_outline' in elems:       contours(gtd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=[[1.0,0.8,0.6]], linecolors=[[0.8,0.6,0.5]], linewidth=1.5)
    if 'gt_smax' in elems:          contours(gtd.logp(gridxy, sigma_max), levels=[-1.41, 0], colors=['C1'], alpha=0.2, linealpha=0.2)
    if 'gt_shaded' in elems:        contours(gtd.logp(gridxy), levels=np.linspace(-2.5, 3.07, num=15)[1:-1], cmap='Oranges', linealpha=0.2)
    if 'trajectories' in elems:     lines(trajectories.transpose(0, 1), alpha=0.3)
    if 'scores_net' in elems:       arrows(samples, net.score(samples, sigma_max), color='C2')
    if 'scores_gnet' in elems:      arrows(samples, gnet.score(samples, sigma_max), color='C3')
    if 'scores_ratio' in elems:     arrows(samples, net.score(samples, sigma_max) - gnet.score(samples, sigma_max), color='C0')
    if 'samples' in elems:          points(trajectories[-1], size=15, alpha=0.25)
    if 'samples_before' in elems:   points(samples)
    if 'samples_after' in elems:    points(trajectories[-1])
    if 'samples_before_small' in elems: points(samples, alpha=0.5, size=15, color="m")
    if 'trajectories_small' in elems: lines(trajectories.transpose(0, 1), alpha=0.3, color="lightgrey")
    if 'out_gt_uncond' in elems:    contours(alloutd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=[[0.9,0.9,0.9]], linecolors=[[0.7,0.7,0.7]], linewidth=1.5)
    if 'out_gt_outline' in elems:   contours(outd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=[[1.0,0.8,0.6]], linecolors=[[0.8,0.6,0.5]], linewidth=1.5)
    if 'out_gt_after' in elems:     points(out_gt_trajectories[-1], alpha=0.35, size=15, color="b")
    if 'out_gt_trajectories_small' in elems: lines(out_gt_trajectories.transpose(0, 1), alpha=0.4, color="lightsteelblue")
    if 'out_samples' in elems:      points(out_trajectories[-1], size=15, alpha=0.35)
    if 'out_samples_before' in elems: points(out_samples)
    if 'out_samples_after' in elems:  points(out_trajectories[-1])
    if 'out_samples_before_small' in elems: points(out_samples, alpha=0.5, size=15, color="m")
    if 'out_trajectories' in elems:   lines(out_trajectories.transpose(0, 1), alpha=0.3)
    if 'out_trajectories_small' in elems: lines(out_trajectories.transpose(0, 1), alpha=0.4, color="lightgrey")
    if 'gt_uncond_transparent' in elems:  contours(allgtd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=["grey"], linecolors=["grey"], linewidth=1.5, alpha=0)
    if 'gt_outline_transparent' in elems: contours(gtd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=["black"], linecolors=["black"], linewidth=1.5, alpha=0)
    if 'gt_uncond_thin' in elems:  contours(allgtd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=["grey"], linecolors=["grey"], linewidth=1, alpha=0)
    if 'gt_outline_thin' in elems: contours(gtd.logp(gridxy), levels=[GT_LOGP_LEVEL, 0], colors=["black"], linecolors=["black"], linewidth=1, alpha=0)

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """2D toy example from the paper "Guiding a Diffusion Model with a Bad Version of Itself".

    Examples:

    \b
    # Visualize sampling distributions using autoguidance.
    python toy_example.py plot

    \b
    # Same, but save the plot as PNG instead of displaying it.
    python toy_example.py plot --save=out.png

    \b
    # Same, but specify the models explicitly.
    python toy_example.py plot \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim64/iter4096.pkl \\
        --gnet=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim32/iter0512.pkl \\
        --guidance=3

    \b
    # Same, but using classifier-free guidance.
    python toy_example.py plot \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim64/iter4096.pkl \\
        --gnet=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsAB-layers04-dim32/iter0512.pkl \\
        --guidance=4

    \b
    # Retrain the main model and visualize progress.
    python toy_example.py train

    \b
    # Retrain the main model and save snapshots.
    python toy_example.py train \\
        --outdir=toy-example/clsA-layers04-dim64 \\
        --cls=A --layers=4 --dim=64 --viz=false
    """
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------
# 'train' subcommand.

@cmdline.command()
@click.option('--outdir', help='Output directory', metavar='DIR',     type=str, default=None)
@click.option('--cls',    help='Target classes', metavar='A|B|AB',    type=str, default='A', show_default=True)
@click.option('--layers', help='Number of layers', metavar='INT',     type=int, default=4, show_default=True)
@click.option('--dim',    help='Hidden dimension', metavar='INT',     type=int, default=64, show_default=True)
@click.option('--total-iter', help='Number of training iterations', metavar='INT', 
                                                                      type=int, default=4<<10, show_default=True)
@click.option('--batch-size', help='Batch size', metavar='INT',       type=int, default=4<<10, show_default=True)
@click.option('--val/--no-val', help='Use validation?', metavar='BOOL', 
                                                                      type=bool, default=True, show_default=True)
@click.option('--test/--no-test', help='Run test?', metavar='BOOL',   type=bool, default=True, show_default=True)
@click.option('--guidance/--no-guidance', help='Use auto-guidance?', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--guide-path', help='Use auto-guidance?', metavar='PATH', 
                                                                      type=str, default=None, show_default=True)
@click.option('--interpol/--no-interpol', help='Use guide interpolation on training scores?', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--acid/--no-acid',   
                          help='Use ACID batch selection?', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--filt',   help='ACID filter ratio', metavar='FLOAT',  type=float, default=0.8, show_default=True)
@click.option('--n',      help='ACID chunk size', metavar='INT',      type=int, default=16, show_default=True)
@click.option('--diff/--no-diff',   
                          help='Use ACID learnability score?', metavar='BOOL', 
                                                                      type=bool, default=True, show_default=True)
@click.option('--invert/--no-invert', help='Use inverted ACID scores?', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--late/--no-late', help='Delay ACID start?', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--early/--no-early', help='Run ACID just at the beginning?', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--trick/--no-trick', help='Use the softmax stability trick?', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--seed',   help='Random seed', metavar='FLOAT',        type=int, default=None, show_default=True)
@click.option('--verbose/--no-verbose', help='Whether to log information messages or not', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--debug/--no-debug', help='Whether to log debug messages or not', metavar='BOOL', 
                                                                      type=bool, default=False, show_default=True)
@click.option('--logging', help='Log filename', metavar='DIR',        type=str, default=None)
@click.option('--viz/--no-viz', help='Visualize progress?', metavar='BOOL', 
                                                                      type=bool, default=True, show_default=True)
@click.option('--device', help='CUDA GPU id?', metavar='INT',         type=int, default=0, show_default=True)
def train(outdir, cls, layers, dim, total_iter, batch_size, val, test, viz, 
          guidance, guide_path, interpol, acid, n, filt, diff, invert, late, early, trick,
          seed, verbose, debug, logging, device):
    """Train a 2D toy model with the given parameters."""
    if debug: verbosity = 2
    elif verbose: verbosity = 1
    else: verbosity = 0
    if outdir is not None:
        outdir = os.path.join(dirs.MODELS_HOME, outdir)
    if guide_path is not None:
        guide_path = os.path.join(dirs.MODELS_HOME, guide_path)
    if logging is not None:
        log_filepath = os.path.join(dirs.MODELS_HOME, logging)
    else: log_filepath = None
    pkl_pattern = f'{outdir}/iter%04d.pkl' if outdir is not None else None
    viz_iter = 32 if viz else None
    device = torch.device("cuda:"+str(device))
    log.info('Training...')
    do_train(classes=cls, num_layers=layers, hidden_dim=dim, 
             total_iter=total_iter, batch_size=batch_size, seed=seed, 
             validation=val, testing=test, 
             guidance=guidance, guide_path=guide_path, guide_interpolation=interpol,
             acid=acid, acid_N=n, acid_f=filt, acid_diff=diff, acid_inverted=invert, 
             acid_late=late, acid_stability_trick=trick, acid_early=early,
             pkl_pattern=pkl_pattern, 
             viz_iter=viz_iter,
             verbosity=verbosity, log_filename=log_filepath, device=device)
    log.info('Done.')

#----------------------------------------------------------------------------
# 'test' subcommand.

@cmdline.command()
@click.option('--net-path', 
    help="Local path to the network's last checkpoint", metavar='PATH', type=str)
@click.option('--ema-path', 
    help="Local path to the EMA's last checkpoint", metavar='PATH', type=str, default=None, show_default=True)
@click.option('--guide-path', 
    help="Local path to the guide's checkpoint", metavar='PATH', type=str, default=None, show_default=True)
@click.option('--acid/--no-acid',   
    help='Was this trained using ACID batch selection?', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--n-samples', help='Number of samples', metavar='INT', type=int, default=4<<8, show_default=True)
@click.option('--batch-size', help='Batch size', metavar='INT', type=int, default=4<<8, show_default=True)
@click.option('--seed', help='Random seed', metavar='FLOAT', type=int, default=None, show_default=True)
@click.option('--logging', 
    help='Local path to logging file', metavar='DIR', type=str, default=None)
def test(net_path, ema_path, guide_path, acid, n_samples, batch_size, seed, logging):
    """Test a given model on a fresh batch of test data"""

    net_path = os.path.join(dirs.MODELS_HOME, net_path)
    if ema_path is not None:
        ema_path = os.path.join(dirs.MODELS_HOME, ema_path)
    if guide_path is not None:
        guide_path = os.path.join(dirs.MODELS_HOME, guide_path)
    if logging is not None:
        log_filepath = os.path.join(dirs.MODELS_HOME, logging)

    do_test(net_path, ema_path=ema_path, guide_path=guide_path, acid=acid, 
            batch_size=batch_size, n_samples=n_samples, seed=seed, log_filename=log_filepath)

#----------------------------------------------------------------------------
# 'plot' subcommand.

@cmdline.command()
@click.option('--net',      help='Main model  [default: download]', metavar='PKL|URL',          type=str, default='https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim64/iter4096.pkl')
@click.option('--gnet',     help='Guiding model  [default: autoguidance]', metavar='PKL|URL',   type=str, default='https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim32/iter0512.pkl')
@click.option('--guidance', help='Guidance weight', metavar='FLOAT',                            type=float, default=3, show_default=True)
@click.option('--save',     help='Save figure, do not display', metavar='PNG|PDF',              type=str, default=None)
def plot(net, gnet, guidance, save, device=torch.device('cuda')):
    """Visualize sampling distributions with and without guidance."""
    log.info('Loading models...')
    if isinstance(net, str):
        with dnnlib.util.open_url(net, cache_dir=PRETRAINED_HOME) as f:
            net = pickle.load(f).to(device)
    if isinstance(gnet, str):
        with dnnlib.util.open_url(gnet, cache_dir=PRETRAINED_HOME) as f:
            gnet = pickle.load(f).to(device)

    # Initialize plot.
    log.info('Drawing plots...')
    plt.rcParams['font.size'] = 28
    fig, ax = plt.subplots(figsize=[48, 25], dpi=40, tight_layout=True)

    # Draw first row.
    plt.subplot(2, 4, 1)
    plt.title('Ground truth distribution')
    do_plot(elems={'gt_uncond', 'gt_outline', 'samples'}, ax=ax, **FIG1_KWARGS)
    plt.subplot(2, 4, 2)
    plt.title('Sample distribution without guidance')
    do_plot(net=net, elems={'gt_uncond', 'gt_outline', 'samples'}, ax=ax, **FIG1_KWARGS)
    plt.subplot(2, 4, 3)
    plt.title('Sample distribution with guidance')
    do_plot(net=net, gnet=gnet, guidance=guidance, elems={'gt_uncond', 'gt_outline', 'samples'}, ax=ax, **FIG1_KWARGS)
    plt.subplot(2, 4, 4)
    plt.title('Trajectories without guidance')
    do_plot(net=net, elems={'gt_shaded', 'trajectories', 'samples_after'}, ax=ax, **FIG2_KWARGS)

    # Draw second row.
    plt.subplot(2, 4, 5)
    plt.title('PDF of main model')
    do_plot(net=net, elems={'p_net', 'gt_smax', 'scores_net', 'samples_before'}, ax=ax, **FIG2_KWARGS)
    plt.subplot(2, 4, 6)
    plt.title('PDF of guiding model')
    do_plot(net=net, gnet=gnet, elems={'p_gnet', 'gt_smax', 'scores_gnet', 'samples_before'}, ax=ax, **FIG2_KWARGS)
    plt.subplot(2, 4, 7)
    plt.title('PDF ratio (main / guiding)')
    do_plot(net=net, gnet=gnet, elems={'p_ratio', 'gt_smax', 'scores_ratio', 'samples_before'}, ax=ax, **FIG2_KWARGS)
    plt.subplot(2, 4, 8)
    plt.title('Trajectories with guidance')
    do_plot(net=net, gnet=gnet, guidance=guidance, elems={'gt_shaded', 'trajectories', 'samples_after'}, ax=ax, **FIG2_KWARGS)

    # Save or display.
    if save is not None:
        log.info('Saving to %s inside of results home folder', save)
        save = os.path.join(dirs.RESULTS_HOME, save)
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=80)
    else:
        log.info('Displaying...')
        plt.show()
    log.info('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------