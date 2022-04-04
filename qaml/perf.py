import qaml
import torch

import numpy as np

from collections import Counter

@torch.no_grad()
def distance_from_gibbs(model, samples, beta_range=None, num_samples=1e4, k=None):
    """ Test a range of inverse temperature values to find the closests match
        to the given distribution. Proximity to a Gibbs distribution doesn't
        directly
        Arg:
            model (qaml.nn.BoltzmannMachine):
            samples (tensor):
            beta_range (iterable or None):
            num_samples (int):
            k (int or None):
        Return:
            beta (float):
            distance (float):
    """
    if beta_range is None:
        beta_range = np.linspace(1,6,11)

    E_samples = model.energy(*samples)
    unique, counts = np.unique(E_samples.numpy(), return_counts=True)
    hist_samples = dict(zip(unique, counts/len(E_samples)))

    if k is None:
        ref_sampler = qaml.sampler.ExactNetworkSampler(model)
        sample = lambda n: ref_sampler(n)
    else:
        ref_sampler = qaml.sampler.GibbsNetworkSampler(model)
        sample = lambda n: ref_sampler(torch.rand(n,model.V),k=k)

    beta_eff = 1.0
    distance = float('inf')
    for beta_i in beta_range:
        ref_sampler.beta = beta_i
        vk,hk = sample(num_samples)
        E_gibbs = model.energy(vk.bernoulli(),hk.bernoulli())
        unique, counts = np.unique(E_gibbs.numpy(), return_counts=True)
        hist_gibbs = dict(zip(unique, counts/num_samples))
        E_set = set(hist_samples) | set(hist_gibbs)
        E_diff = {k:abs(hist_samples.get(k,0)-hist_gibbs.get(k,0)) for k in E_set}
        dist_i = sum(E_diff.values())/2
        if dist_i < distance:
            distance = dist_i
            beta_eff = beta_i

    return beta_eff, distance
