import copy
import torch
import dimod
import warnings
import minorminer
import dwave.system
import dwave.embedding

import numpy as np
import networkx as nx
import dwave_networkx as dnx

from random import random
from minorminer.utils.polynomialembedder import processor

class NetworkSampler(torch.nn.Module):
    r""" Sample generator for the probabilistic model provided.
    Args:
        model (e.g BotlzmannMachine): Generative Network Model
        beta (float, optional): Inverse temperature for the distribution.
    """

    beta : float # Inverse-temperature to match sampler

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
    def __init__(self, model, num_chains, beta=1.0):
        super(PersistentGibbsNetworkSampler, self).__init__(model,beta)
        self.prob_v.data = torch.rand(num_chains,model.V)

    def forward(self, num_samples, k=1, init=None):
        beta = self.beta
        prob_vk = self.prob_v.clone() if init is None else init.clone()
        prob_hk = self.model.forward(prob_vk.bernoulli(),scale=beta)

        for _ in range(k):
            prob_vk.data = self.model.generate(prob_hk.bernoulli(),scale=beta)
            prob_hk.data = self.model.forward(prob_vk.bernoulli(),scale=beta)

        self.prob_v.data = prob_vk.data
        self.prob_h.data = prob_hk.data
        return prob_vk[:num_samples], prob_hk[:num_samples]

class GibbsNetworkSampler(NetworkSampler):

    def __init__(self, model, beta=1.0):
        super(GibbsNetworkSampler, self).__init__(model,beta)

    @torch.no_grad()
    def reconstruct(self, input_data, k=1, mask=None):
        beta = self.beta

        if mask is None:
            mask = torch.ones_like(input_data)
        clamp = torch.mul(input_data,mask)

        prob_vk = clamp.clone().masked_fill_((mask==0),0.5)
        prob_hk = self.model.forward(prob_vk.bernoulli(),scale=beta)
        for _ in range(k):
            prob_vk.data = self.model.generate(prob_hk,scale=beta).data
            masked = clamp + (mask==0)*prob_vk.data
            prob_hk.data = self.model.forward(masked,scale=beta).data

        self.prob_v.data = masked.data
        self.prob_h.data = prob_hk.data
        return masked, prob_hk

    @torch.no_grad()
    def forward(self, v0, k=1):
        beta = self.beta
        prob_vk = v0.clone()
        prob_hk = self.model.forward(prob_vk.bernoulli(),scale=beta)

        for _ in range(k):
            prob_vk.data = self.model.generate(prob_hk.bernoulli(),scale=beta)
            prob_hk.data = self.model.forward(prob_vk.bernoulli(),scale=beta)

        self.prob_v.data = prob_vk.data
        self.prob_h.data = prob_hk.data
        return prob_vk, prob_hk

""" The next samplers formulate the model as a Binary Quadratic Model (BQM) """
class BinaryQuadraticModelSampler(NetworkSampler):

    _qubo = None
    _ising = None
    _networkx_graph = None

    sampleset = None

    def __init__(self, model, beta=1.0):
        super(BinaryQuadraticModelSampler, self).__init__(model,beta)

    def to_qubo(self):
        bias_v = self.model.b.data.numpy()
        bias_h = self.model.c.data.numpy()
        W = self.model.W.data.detach().clone()
        V = self.model.V
        # Linear biases
        lin_V = {i: -float(b) for i,b in enumerate(bias_v)}
        lin_H = {j: -float(c) for j,c in enumerate(bias_h)}
        linear = {**lin_V,**{V+j: c for j,c in lin_H.items()}}

        # To prune BQM from mask
        mask = self.model.state_dict().get('W_mask',torch.ones_like(W))

        # Quadratic weights
        quadratic = {}
        for i in lin_V:
            for j in lin_H:
                if mask[j][i]:
                    quadratic[(i,V+j)] = -float(W[j][i])

        self._qubo = dimod.BinaryQuadraticModel(linear,quadratic,'BINARY')
        return self._qubo

    @property
    def qubo(self):
        if self._qubo is None:
            return self.to_qubo()
        else:
            return self._qubo

    def to_ising(self):
        """When converting a Boltzmann Machine (BM) model to Ising, first
        formulate as Quadratic Unconstrained Binary Optimization (QUBO) and then
        transform to Ising."""
        self._ising = self.to_qubo().change_vartype('SPIN',inplace=False)
        return self._ising

    @property
    def ising(self):
        if self._ising is None:
            return self.to_ising()
        else:
            return self._ising

    def to_networkx_graph(self):
        self._networkx_graph = self.bqm.to_networkx_graph()
        return self._networkx_graph

    @property
    def networkx_graph(self):
        if self._networkx_graph is None:
            return self.to_networkx_graph()
        else:
            return self._networkx_graph

BQMSampler = BinaryQuadraticModelSampler

class SimulatedAnnealingNetworkSampler(BinaryQuadraticModelSampler):
    sa_kwargs = {"num_sweeps":1000}

    def __init__(self, model, beta=1.0, **kwargs):
        BinaryQuadraticModelSampler.__init__(self,model,beta)
        self.sampler = dimod.SimulatedAnnealingSampler(**kwargs)

    def forward(self, num_reads=100, **kwargs):
        bqm = self.to_qubo()
        bqm.scale(float(self.beta))
        sa_kwargs = {**self.sa_kwargs,**kwargs}
        sampleset = self.sampler.sample(bqm,num_reads=num_reads,**sa_kwargs)
        samples = sampleset.record.sample.copy()
        sampletensor = torch.tensor(samples,dtype=torch.float32)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        self.sampleset = sampleset
        return samples_v, samples_h

SASampler = SimulatedAnnealingNetworkSampler

class ExactNetworkSampler(BinaryQuadraticModelSampler):

    def __init__(self, model, beta=1.0):
        BinaryQuadraticModelSampler.__init__(self,model,beta)
        self.sampler = dimod.ExactSolver()

    def forward(self, num_reads=None, **ex_kwargs):
        beta = self.beta
        bqm = self.to_qubo()

        solutions = self.sampler.sample(bqm,**ex_kwargs)
        energies = solutions.record['energy']
        Z = np.exp(-beta*energies).sum()
        P = torch.Tensor(np.exp(-beta*energies)/Z)

        if num_reads is None:
            tensorset = torch.Tensor(solutions.record.sample)
            prob = torch.matmul(P,tensorset).unsqueeze(0)
            vs,hs = prob.split([self.model.V,self.model.H],1)
        else:
            samples = [solutions.record.sample[i]
                       for i in torch.multinomial(P,num_reads,replacement=True)]
            tensorset = torch.Tensor(samples)
            vs,hs = tensorset.split([self.model.V,self.model.H],1)

        self.sampleset = solutions
        return vs, hs

class ExactEmbeddedNetworkSampler(BinaryQuadraticModelSampler):

    def __init__(self, model, beta=1.0, target_graph=None, embedding=None):
        BinaryQuadraticModelSampler.__init__(self,model,beta)

        if target_graph is None:
            target_graph = dnx.chimera_graph(16,16,4)
        self._networkx_graph = target_graph

        struct_sampler = dimod.StructureComposite(dimod.ExactSolver(),
                                                  list(target_graph.nodes),
                                                  list(target_graph.edges))

        if embedding is None:
            if 'Restricted' in repr(self.model):
                cache = minorminer.busclique.busgraph_cache(self.networkx_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
            else:
                S = self.qubo.quadratic
                embedding = minorminer.find_embedding(S,self.networkx_graph)
            if not embedding:
                warnings.warn("Embedding not found")

        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            edgelist = self.networkx_graph.edges
            embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        self.embedding = embedding
        self.sampler = dwave.system.FixedEmbeddingComposite(struct_sampler,
                                                            embedding)

    def forward(self, num_reads=None, **ex_kwargs):
        beta = self.beta
        bqm = self.to_qubo()

        solutions = self.sampler.sample(bqm,**ex_kwargs)
        energies = solutions.record['energy']
        Z = np.exp(-beta*energies).sum()
        P = torch.Tensor(np.exp(-beta*energies)/Z)

        if num_reads is None:
            tensorset = torch.Tensor(solutions.record.sample)
            prob = torch.matmul(P,tensorset).unsqueeze(0)
            vs,hs = prob.split([self.model.V,self.model.H],1)
        else:
            samples = [solutions.record.sample[i]
                       for i in torch.multinomial(P,num_reads,replacement=True)]
            tensorset = torch.Tensor(samples)
            vs,hs = tensorset.split([self.model.V,self.model.H],1)

        self.sampleset = solutions
        return vs, hs


class QuantumAnnealingNetworkSampler(BinaryQuadraticModelSampler):

    sample_kwargs = {"annealing_time":20.0,"label":"QARBM-DEV"}

    embed_kwargs = {"chain_strength":1.6}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}


    scalar : float # Scaling factor to fit sampler's range
    embedding = None

    def __init__(self, model, embedding=None, beta=1.0, failover=False,
                 retry_interval=-1, **config):
        BinaryQuadraticModelSampler.__init__(self,model,beta=beta)
        self.sampler = dwave.system.DWaveSampler(failover,retry_interval,**config)
        if embedding is None:
            if 'Restricted' in repr(self.model):
                cache = minorminer.busclique.busgraph_cache(self.networkx_graph)
                embedding = cache.find_biclique_embedding(model.V,model.H)
            else:
                S = self.qubo.quadratic
                embedding = minorminer.find_embedding(S,self.networkx_graph)
            if not embedding:
                warnings.warn("Embedding not found")
        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            edgelist = self.networkx_graph.edges
            embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)
        self.embedding = embedding
        self.scalar = 1.0

    def to_networkx_graph(self):
        self._networkx_graph = self.sampler.to_networkx_graph()
        return self._networkx_graph

    def embed_bqm(self, bqm=None, auto_scale=False, **embed_kwargs):
        if bqm is None:
            bqm = self.to_ising().copy()

        embedding = self.embedding

        target_bqm = embedding.embed_bqm(bqm,**embed_kwargs)
        ignoring = [e for u in embedding for e in embedding.chain_edges(u)]
        scale_kwargs = {'ignored_interactions':ignoring}
        if auto_scale:
            # Same as target auto_scale but retains scalar
            scale_kwargs.update({'bias_range':self.sampler.properties['h_range'],
                               'quadratic_range':self.sampler.properties['j_range']})
            self.scalar = target_bqm.normalize(**scale_kwargs)
        else:
            target_bqm.scale(1.0/float(self.beta),**scale_kwargs)
        return target_bqm

    def sample_rbm(self, embed_kwargs={}, unembed_kwargs={}, **sample_kwargs):
        bqm = self.to_ising().copy()
        embedding = self.embedding

        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        sample_kwargs = {**self.sample_kwargs,**sample_kwargs}
        unembed_kwargs = {**self.unembed_kwargs,**unembed_kwargs}

        responses = []
        flipped_bqm = bqm.copy()
        transform = {v: False for v in bqm.variables}
        num_spin_reversal_transforms = 4

        num_reads = sample_kwargs.pop('num_reads',100)
        auto_scale = sample_kwargs.pop('auto_scale',False)
        num_spinrevs = sample_kwargs.pop('num_spin_reversal_transforms',0)

        if num_spinrevs>1:
            reads_per_transform = num_reads//num_spinrevs
            iter_num_reads = [reads_per_transform]*(num_spinrevs-1)
            iter_num_reads += [reads_per_transform+(num_reads%num_spinrevs)]
        else:
            iter_num_reads = [num_reads]

        for num_reads in iter_num_reads:
            # Don't flip if num_spin_reversal_transforms is 0
            if num_spinrevs>0:
                for v in list(bqm.variables):
                    if random() > .5:
                        transform[v] = not transform[v]
                        flipped_bqm.flip_variable(v)

            target_bqm = self.embed_bqm(flipped_bqm,auto_scale,**embed_kwargs)
            target_response = self.sampler.sample(target_bqm,auto_scale=False,
                                          num_reads=num_reads,answer_mode='raw',
                                          num_spin_reversal_transforms=0,
                                          **sample_kwargs)
            target_response.resolve()
            target_response.change_vartype('BINARY',inplace=True)

            flipped_response = self.unembed_sampleset(target_response,**unembed_kwargs)

            tf_idxs = [flipped_response.variables.index(v)
                       for v, flip in transform.items() if flip]

            flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]

            responses.append(flipped_response)

        return dimod.sampleset.concatenate(responses)

    def unembed_sampleset(self, response, **unembed_kwargs):
        sampleset = dwave.embedding.unembed_sampleset(response,self.embedding,
                                                     self.qubo,**unembed_kwargs)
        return sampleset

    def forward(self, num_reads, embed_kwargs={}, unembed_kwargs={}, **kwargs):


        kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        self.sampleset = self.sample_rbm(embed_kwargs,unembed_kwargs,**kwargs)


        samples = self.sampleset.record.sample.copy()
        sampletensor = torch.tensor(samples,dtype=torch.float32)
        samples_v,samples_h = sampletensor.split([self.model.V,self.model.H],1)

        return samples_v, samples_h

QASampler = QuantumAnnealingNetworkSampler
