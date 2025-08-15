

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
import os
import sys
from .unet import UNet
from resize_right import resize


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffCorr(nn.Module):
    def __init__(self, AM, device, num_steps=3, beta_start=1e-4, beta_end=0.02):
        super(DiffCorr, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.AM = AM
        self.num_steps = num_steps
        # print("111",num_steps) ##0.0001
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.time_embedding = SinusoidalTimeEmbeddings(AM)

        self.reverse_model = UNet(AM, AM)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(x_start.device).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device).view(-1, 1, 1, 1)

        return sqrt_recip_alphas_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t):
        # time_emb = self.time_embedding(torch.tensor([t], device=x.device).repeat(x_t.size(0)))
        time_emb = self.time_embedding(torch.tensor([t]).repeat(x_t.size(0)))
        pred_noise = self.reverse_model(x_t, time_emb)

        alphas_t = self.alphas[t].view(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        c0 = 1.0 / torch.sqrt(alphas_t)
        c1 = (1 - alphas_t) / torch.sqrt(1 - alphas_cumprod_t)

        if t > 0:
            rand_noise = torch.randn_like(x_t)
        else:
            rand_noise = torch.zeros_like(x_t)

        x_t_minus_1 = c0 * (x_t - c1 * pred_noise) + rand_noise * sqrt_one_minus_alphas_cumprod_t

        return x_t_minus_1

    def forward(self, x, L, timesteps=None):
        target_len = 64
        x1 = x
        x = resize(x, out_shape=(x.shape[0], target_len, target_len))  ##[1, 1, 120, 120]
        if timesteps is None:
            timesteps = self.num_steps

        batch_size = x.size(0)
        t = torch.randint(0, timesteps, (batch_size,), device=x.device).long()

        x_t = self.q_sample(x, t)
        x = x_t
        for t in reversed(range(timesteps)):
            x = self.p_sample(x, t)

        x = resize(x.squeeze(0), out_shape=(x.shape[1], L, L))
        x = x + x1
        return x








