import torch
import dimod
import warnings
import minorminer
import dwave.system
import dwave.embedding

import numpy as np

class NetworkSampler(torch.nn.Module):
    r""" Sample generator for the probabilistic model provided.
    Args:
        model (e.g BotlzmannMachine): Generative Network Model
        beta (float, optional): Inverse temperature for the distribution.
    """

    scalar : float
    beta : float # Inverse-temperature

    def __init__(self, model, beta=1.0):
        super(NetworkSampler, self).__init__()
        self.model = model
        # Sampler stores states
        visible_unknown = torch.Tensor([float('NaN')]*model.V)
        self.prob_v = torch.nn.Parameter(visible_unknown, requires_grad=False)

        hidden_unknown = torch.Tensor([float('NaN')]*model.H)
        self.prob_h = torch.nn.Parameter(hidden_unknown, requires_grad=False)

        if torch.is_tensor(beta):
            self.register_buffer('beta', beta)
        else:
            self.beta = beta

    @property
    def binary_quadratic_model(self):
        bias_v = self.model.b.data.numpy()
        bias_h = self.model.c.data.numpy()
        W = self.model.W.data.numpy()
        V = self.model.V

        lin_V = {i: -b.item() for i,b in enumerate(bias_v)}
        lin_H = {j: -c.item() for j,c in enumerate(bias_h)}
        linear = {**lin_V,**{V+j: c for j,c in lin_H.items()}}

        # Values from RBM are for QUBO formulation i.e. BINARY [0,1]
        quadratic = {}
        for i in lin_V:
            for j in lin_H:
                weight = -W[j][i].item()
                if weight == 0: continue
                quadratic[(i,V+j)] = weight

        return dimod.BinaryQuadraticModel(linear,quadratic,'BINARY')

    def sample_visible(self):
        try:
            return self.prob_v.bernoulli()
        except RuntimeError as e:
            warnings.warn(f"Invalid probability vector: {self.prob_v}")
            return torch.zeros_like(self.prob_v)

    def sample_hidden(self):
        try:
            return self.prob_h.bernoulli()
        except RuntimeError as e:
            warnings.warn(f"Invalid probability vector: {self.prob_h}")
            return torch.zeros_like(self.prob_h)

class PersistentGibbsNetworkSampler(NetworkSampler):
    """ Sampler for Persistent Constrastive Divergence training with k steps.

        Args:
            model (torch.nn.Module): PyTorch `Module` with `forward` method.

            num_chains (int): PCD keeps N chains at all times. This number must
                match the batch size.

    """
    def __init__(self, model, num_chains):
        super(PersistentGibbsNetworkSampler, self).__init__(model)
        self.prob_v.data = torch.rand(num_chains,model.V)

    def forward(self, num_samples, k=1, init=None):
        prob_vk = self.prob_v if init is None else init
        prob_hk = self.model(prob_vk)

        for _ in range(k):
            prob_vk.data = self.model.generate(prob_hk)
            prob_hk = self.model.forward(prob_vk.bernoulli())

        self.prob_v.data = prob_vk
        self.prob_h.data = prob_hk
        return prob_vk[:num_samples], prob_hk[:num_samples]

class GibbsNetworkSampler(NetworkSampler):

    def __init__(self, model):
        super(GibbsNetworkSampler, self).__init__(model)

    def forward(self, v0, k=1):
        prob_vk = v0.clone()
        prob_hk = self.model(prob_vk)

        for _ in range(k):
            prob_vk.data = self.model.generate(prob_hk)
            prob_hk.data = self.model.forward(prob_vk.bernoulli())

        self.prob_v.data = prob_vk.data
        self.prob_h.data = prob_hk.data
        return prob_vk, prob_hk

class SimulatedAnnealingNetworkSampler(NetworkSampler,dimod.SimulatedAnnealingSampler):

    sa_kwargs = {"num_sweeps":1000}

    def __init__(self, model):
        NetworkSampler.__init__(self,model)
        dimod.SimulatedAnnealingSampler.__init__(self)

    def forward(self, num_reads=100, **kwargs):
        bqm = self.binary_quadratic_model
        sa_kwargs = {**self.sa_kwargs,**kwargs}
        sampleset = self.sample(bqm,num_reads=num_reads,**sa_kwargs)

        sampletensor = torch.Tensor(sampleset.record.sample.copy())
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h

class ExactNetworkSampler(NetworkSampler,dimod.ExactSolver):

    def __init__(self, model):
        NetworkSampler.__init__(self,model)
        dimod.ExactSolver.__init__(self)

    def forward(self, num_reads=100, **ex_kwargs):
        beta = self.beta
        bqm = self.binary_quadratic_model
        solutions = self.sample(bqm,**ex_kwargs)

        Z = sum(np.exp(-beta*E) for E in solutions.record['energy'])
        P = torch.Tensor([np.exp(-beta*E)/Z for E in solutions.record['energy']])
        samples = [solutions.record['sample'][i] for i in torch.multinomial(P,num_reads,replacement=True)]

        sampletensor = torch.Tensor(samples)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h

class QuantumAnnealingNetworkSampler(NetworkSampler,dwave.system.DWaveSampler):

    sample_kwargs = {"answer_mode":'raw',
                     "num_spin_reversal_transforms":0,
                     "anneal_schedule":[(0.0,0.0),(0.5,0.5),(10.5,0.5),(11.0,1.0)],
                     "auto_scale":False}

    embed_kwargs = {"chain_strength":1.6}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}

    def __init__(self, model, embedding=None, beta=1.0,
                 failover=False, retry_interval=-1, **config):
        NetworkSampler.__init__(self,model,beta=beta)
        dwave.system.DWaveSampler.__init__(self,failover,retry_interval,**config)
        self.networkx_graph = self.to_networkx_graph()
        self.sampleset = None
        if embedding is None:
            if 'Restricted' in repr(self.model):
                cache = minorminer.busclique.busgraph_cache(self.networkx_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
            else:
                S = self.binary_quadratic_model.quadratic
                embedding = minorminer.find_embedding(S,self.networkx_graph)
        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            self.embedding = dwave.embedding.EmbeddedStructure(self.networkx_graph.edges,embedding)
        else:
            self.embedding = embedding
        self.scalar = 1.0

    def embed_bqm(self, visible=None, hidden=None, auto_scale=False, **kwargs):
        embedding = self.embedding
        bqm = self.binary_quadratic_model
        embed_kwargs = {**self.embed_kwargs,**kwargs}

        if auto_scale:
            # Same as target auto_scale but retains scalar
            norm_args = {'bias_range':self.properties['h_range'],
                         'quadratic_range':self.properties['j_range']}
            self.scalar = bqm.normalize(**norm_args)
        else:
            bqm.scale(1.0/float(self.beta))

        target_bqm = self.embedding.embed_bqm(bqm, **embed_kwargs)

        return target_bqm

    def unembed_sampleset(self,**kwargs):
        unembed_kwargs = {**self.unembed_kwargs,**kwargs}

        sampleset = dwave.embedding.unembed_sampleset(self.sampleset,
                                                      self.embedding,
                                                      self.binary_quadratic_model,
                                                      **unembed_kwargs)
        return sampleset

    def forward(self, num_reads, visible=None, hidden=None, auto_scale=False,
                embed_kwargs={}, unembed_kwargs={}, **kwargs):

        sample_kwargs = {**self.sample_kwargs,**kwargs}
        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        unembed_kwargs = {**self.unembed_kwargs,**unembed_kwargs}

        target_bqm = self.embed_bqm(visible,hidden,auto_scale,**embed_kwargs)

        self.sampleset = self.sample(target_bqm,num_reads=num_reads,**sample_kwargs)

        sampleset = self.unembed_sampleset(**unembed_kwargs)

        sampletensor = torch.tensor(sampleset.record.sample,dtype=torch.float32)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h

QASampler = QuantumAnnealingNetworkSampler
