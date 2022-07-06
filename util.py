import os
import time 
import functools
import numpy as np
from math import cos, pi, floor, sin
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from stft_loss import MultiResolutionSTFTLoss


def flatten(v):
    return [x for y in v for x in y]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path):
    """
    Find latest checkpoint
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            number = f[:-4]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch


def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True, end="; ")
        
        if keyword is not None:
            keyword_parameters = [p for name, p in net.named_parameters() if p.requires_grad and keyword in name]
            params = sum([np.prod(p.size()) for p in keyword_parameters])
            print("{} Parameters: {:.6f}M".format(
                keyword, params / 1e6), flush=True, end="; ")
        
        print(" ")


####################### lr scheduler: Linear Warmup then Cosine Decay #############################

# Adapted from https://github.com/rosinality/vq-vae-2-pytorch

# Original Copyright 2019 Kim Seonghyeon
#  MIT License (https://opensource.org/licenses/MIT)


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cosine(start, end, proportion):
    cos_val = cos(pi * proportion) + 1
    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, cur_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = cur_iter

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class LinearWarmupCosineDecay:
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        iteration=0,
        divider=25,
        warmup_proportion=0.3,
        phase=('linear', 'cosine'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {'linear': anneal_linear, 'cosine': anneal_cosine}

        cur_iter_phase1 = iteration
        cur_iter_phase2 = max(0, iteration - phase1)
        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, cur_iter_phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, cur_iter_phase2, phase_map[phase[1]]),
        ]

        if iteration < phase1:
            self.phase = 0
        else:
            self.phase = 1

    def step(self):
        lr = self.lr_phase[self.phase].step()

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            self.phase = 0

        return lr


####################### model util #############################

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)


@torch.no_grad()
def sampling(net, noisy_audio):
    """
    Perform denoising (forward) step
    """

    return net(noisy_audio)


def loss_fn(net, X, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
    """
    Loss function in CleanUNet

    Parameters:
    net: network
    X: training data pair (clean audio, noisy_audio)
    ell_p: \ell_p norm (1 or 2) of the AE loss
    ell_p_lambda: factor of the AE loss
    stft_lambda: factor of the STFT loss
    mrstftloss: multi-resolution STFT loss function

    Returns:
    loss: value of objective function
    output_dic: values of each component of loss
    """

    assert type(X) == tuple and len(X) == 2
    
    clean_audio, noisy_audio = X
    B, C, L = clean_audio.shape
    output_dic = {}
    loss = 0.0
    
    # AE loss
    denoised_audio = net(noisy_audio)  

    if ell_p == 2:
        ae_loss = nn.MSELoss()(denoised_audio, clean_audio)
    elif ell_p == 1:
        ae_loss = F.l1_loss(denoised_audio, clean_audio)
    else:
        raise NotImplementedError
    loss += ae_loss * ell_p_lambda
    output_dic["reconstruct"] = ae_loss.data * ell_p_lambda

    if stft_lambda > 0:
        sc_loss, mag_loss = mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
        loss += (sc_loss + mag_loss) * stft_lambda
        output_dic["stft_sc"] = sc_loss.data * stft_lambda
        output_dic["stft_mag"] = mag_loss.data * stft_lambda

    return loss, output_dic

