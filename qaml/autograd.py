import torch

class ConstrastiveDivergence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pos_phase, neg_phase, bias_v, bias_h, weights):
        v0, prob_h0 = pos_phase
        prob_vk, prob_hk = neg_phase
        # Values for gradient
        ctx.save_for_backward(v0, prob_h0, prob_vk, prob_hk)
        return torch.nn.functional.mse_loss(v0, prob_vk, reduction='sum')


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve positive and negative phase values
        v0, prob_h0, prob_vk, prob_hk = ctx.saved_tensors

        # Data batch size
        D = len(v0)

        # for j = 1,...,m do
        #     \Delta b_j += v_j^{0} - v_j^{k}
        v_grad = -grad_output*torch.mean(v0 - prob_vk, dim=0)

        # for i = 1,...,n do
        #     \Delta c_i += p(H_i = 1 | v^{0}) - p(H_i = 1 | v^{k})
        h_grad = -grad_output*torch.mean(prob_h0 - prob_hk, dim=0)

        # for i = 1,...,n, j = 1,...,m do
        #     \Delta w_{ij} += p(H_i=1|v^{0})*v_j^{0} - p(H_i=1|v^{k})*v_j^{k}
        W_grad = -grad_output*(torch.matmul(prob_h0.T,v0) - torch.matmul(prob_hk.T,prob_vk))/D

        return None, None, v_grad, h_grad, W_grad

class SampleBasedConstrastiveDivergence(torch.autograd.Function):
    """A sample-based CD trainer is necessary when the number of samples doesn't
    match the size of the batch, therefore needing to average the collection of
    values independently.
    Args:



    """
    @staticmethod
    def forward(ctx, pos_phase, neg_phase, bias_v, bias_h, weights):
        samples_v0, samples_h0 = pos_phase
        samples_vk, samples_hk = neg_phase

        expect_0 = torch.mean(samples_v0,dim=0)
        expect_k = torch.mean(samples_vk,dim=0)

        # Values for gradient
        ctx.save_for_backward(samples_v0,samples_h0,samples_vk,samples_hk)
        return torch.nn.functional.l1_loss(expect_k, expect_0, reduction='sum')


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve positive and negative phase values
        samples_v0, samples_h0, samples_vk, samples_hk = ctx.saved_tensors

        # Data batch size
        D = len(samples_v0)
        # Sampleset size
        S = len(samples_vk)

        v_grad = -grad_output*(torch.mean(samples_v0, dim=0) - torch.mean(samples_vk, dim=0))

        h_grad = -grad_output*(torch.mean(samples_h0,dim=0) - torch.mean(samples_hk, dim=0))

        W_grad = -grad_output*(torch.matmul(samples_h0.T,samples_v0)/D - torch.matmul(samples_hk.T,samples_vk)/S)

        return None, None, v_grad, h_grad, W_grad


class AdaptiveBeta(torch.autograd.Function):
    """ Adaptive hyperparameter (beta) updating using the method from [1]. This
    is useful when dealing with a sampler that has an unknown effective inverse
    temperature (beta), such as quantum annealers.

    [1] Xu, G., Oates, W.S. Adaptive hyperparameter updating for training
    restricted Boltzmann machines on quantum annealers. Sci Rep 11, 2727 (2021).
    https://doi.org/10.1038/s41598-021-82197-1
    """
    @staticmethod
    def forward(ctx, energies_0, energies_k, beta):
        # Values for gradient
        energy_avg_0 = torch.mean(energies_0)
        energy_avg_k = torch.mean(energies_k)
        ctx.save_for_backward(energy_avg_0,energy_avg_k,beta)
        return torch.nn.functional.l1_loss(energy_avg_0, energy_avg_k, reduction='sum')

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve energy average from data and samples
        energy_avg_0, energy_avg_k, beta = ctx.saved_tensors

        beta_grad = -(energy_avg_0-energy_avg_k)/(beta**2)

        return  None, None, beta_grad
