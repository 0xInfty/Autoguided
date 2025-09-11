import torch
import numpy as np

from karras.dnnlib import EasyDict

REQUIRES_REF_LOSS = ["jointly_sample_batch"]

#----------------------------------------------------------------------------
# JEST & ACID's joint sampling batch selection method

def jointly_sample_batch(learner_loss, ref_loss, N=16, filter_ratio=0.8, 
                         learnability=True, inverted=False, numeric_stability_trick=False, 
                         device=None, **kwargs):
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

    # Change device, if needed
    if device is None: 
        device = learner_loss.get_device()
    else:
        learner_loss = learner_loss.detach().to(device)
        ref_loss = learner_loss.detach().to(device)

    # Construct the scores matrix
    learner_loss = learner_loss.reshape((B,1))
    ref_loss = ref_loss.reshape((1,B))
    scores = learner_loss - ref_loss if learnability else - ref_loss  # Shape (B,B)
    if inverted: scores = - scores
    # Rows use different learner loss ==> i is the learner/student's index
    # Columns use different reference loss ==> j is the reference/teacher's index

    # Extract basic score for each element of the super-batch
    logits_ii = torch.diag(scores) # Elements from the diagonal of the scores matrix
    #Q: Is this associated to the probability of picking learner i and ref j?
    
    # Draw the first mini-batch chunk using a uniform probability distribution
    indices = np.random.choice(B, b_over_N, replace=False)

    # Sample all the rest of the mini-batch chunks
    all_sum_of_probs = []
    for n in range(1, N):
        sampled_so_far = n*b_over_N

        # Get a binary mask that indicates which samples have been selected so far
        is_sampled = torch.eye(B).to(device)[indices].sum(axis=0) # (B,)
        
        # Mask scores to only keep learner rows k that have already been sampled
        logits_kj = (scores * is_sampled.view(B, 1)).sum(axis=0) # Sum over columns (B,)
        #Q: Associated to prob of picking learner i and an ref k that was already selected?

        # Mask scores to only keep ref columns k that have already been sampled
        logits_ik = (scores * is_sampled.view(1, B)).sum(axis=1) # Sum over rows (B,)
        #Q: Associated to prob of picking learner i and an ref k that was already selected?
        
        # Get conditional scores given past samples
        logits = logits_ii + logits_kj + logits_ik
        logits = logits / (2*sampled_so_far+1) # Normalize the logits by the number of terms added up
        logits = logits - is_sampled * 1e8 # Avoid sampling with replacement
        #Q: Why subtract that value, instead of setting all these to 0?

        # Sample new mini-batch chunk using the conditional probability distribution
        if numeric_stability_trick: logits = logits - max(logits)
        probabilities = np.exp(logits.detach().cpu().numpy())
        sum_of_probs = sum(probabilities)
        probabilities = probabilities / sum_of_probs
        new_indices = np.random.choice(np.arange(B), b_over_N, replace=False, p=probabilities)
        all_sum_of_probs.append(sum_of_probs)

        # Expand the array of sampled indices
        indices = np.concatenate((indices, new_indices))

    return indices # Gather the n chunks of size b/n and return mini-batch of size b

#----------------------------------------------------------------------------
# Just a random selection

def random_baseline(learner_loss, *args, selection_size=None, **kwargs):

    assert selection_size is not None, "Mini batch size is needed as a kwarg"
    
    # Define size of super-batch
    super_batch_size = int(learner_loss.numel()) # Size B of a super-batch

    # Create a list of random indices
    indices = np.random.choice(super_batch_size, selection_size, replace=False)

    return indices

#----------------------------------------------------------------------------
# A general function to get the selection size according to the function name

def get_selection_size(full_size, **selection_kwargs):
    
    if not isinstance(selection_kwargs, EasyDict):
        selection_kwargs = EasyDict(selection_kwargs)

    if "jointly_sample_batch" in selection_kwargs.func_name:
        # Each mini-batch of size b will be split in n chunks
        b_over_N = int(full_size * (1 - selection_kwargs.filter_ratio) / selection_kwargs.N) # Size b/N of each mini-batch chunk
        b = b_over_N * selection_kwargs.N # Size of the full mini-batch
        return b
    else:
        b = int(full_size * (1 - selection_kwargs.filter_ratio))
        return b

def infer_selection_params(params):

    batch_size = params["batch_size"]
    batch_gpu = params["batch_gpu"]
    selection_kwargs = params["selection_kwargs"]

    mini_batch_gpu = get_selection_size(batch_gpu, **selection_kwargs)
    mini_batch_size = int( mini_batch_gpu * batch_size / batch_gpu )

    selection = params["selection"] or False
    
    early = params["selection_early"] or False
    late = params["selection_late"] or False

    return {"selection":selection, "early":early, "late":late,
            "mini_batch_size":mini_batch_size, "batch_size":batch_size}