import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

import logs

log = logs.create_logger("errors")

#----------------------------------------------------------------------------
# JEST & ACID's joint sampling batch selection method

def jointly_sample_batch(learner_loss, ref_loss, N=16, filter_ratio=0.8, 
                         learnability=True, inverted=False, numeric_stability_trick=False, 
                         device=None, plotting=False, log=log, **kwargs):
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
    log.debug(*logs.get_stats_log("Scores", scores))
    if inverted: scores = - scores
    # Rows use different learner loss ==> i is the learner/student's index
    # Columns use different reference loss ==> j is the reference/teacher's index
    
    log.debug("> JEST Round 0")

    # Extract basic score for each element of the super-batch
    logits_ii = torch.diag(scores) # Elements from the diagonal of the scores matrix
    #Q: Is this associated to the probability of picking learner i and ref j?
    log.debug(*logs.get_stats_log("Logits ii", logits_ii))
    
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
        log.debug(*logs.get_stats_log("Logits kj", logits_kj))
        #Q: Associated to prob of picking learner i and an ref k that was already selected?

        # Mask scores to only keep ref columns k that have already been sampled
        logits_ik = (scores * is_sampled.view(1, B)).sum(axis=1) # Sum over rows (B,)
        log.debug(*logs.get_stats_log("Logits ik", logits_ik))
        #Q: Associated to prob of picking learner i and an ref k that was already selected?
        
        # Get conditional scores given past samples
        logits = logits_ii + logits_kj + logits_ik
        logits = logits / (2*sampled_so_far+1) # Normalize the logits by the number of terms added up
        logits = logits - is_sampled * 1e8 # Avoid sampling with replacement
        #Q: Why subtract that value, instead of setting all these to 0?

        # Sample new mini-batch chunk using the conditional probability distribution
        log.debug(*logs.get_stats_log("Logits", logits))
        if numeric_stability_trick: logits = logits - max(logits)
        probabilities = np.exp(logits.detach().cpu().numpy())
        sum_of_probs = sum(probabilities)
        log.debug(*logs.get_stats_log("Probabilities", probabilities))
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