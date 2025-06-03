# ######################第四次diffussion############################
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# import math
# import numpy as np
# import os
# import sys
# from .unet import UNet
# from resize_right import resize
#
# class SinusoidalTimeEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings
#
# class ResizeConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, target_size):
#         super(ResizeConv2d, self).__init__()
#         self.target_size = target_size
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.interpolate = nn.Upsample(size=(target_size, target_size), mode='bilinear', align_corners=False)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.interpolate(x)
#         return x
#
# class DiffCorr(nn.Module):
#     def __init__(self, AM, device,num_steps=100, beta_start=1e-4, beta_end=0.02):
#         super(DiffCorr, self).__init__()
#         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动获取设备
#         self.device = device  # 保存设备信息
#         self.AM = AM
#         self.num_steps = num_steps
#         # print("111",num_steps) ##0.0001
#         self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device)
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         # self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
#         # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
#         self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)  # 确保在正确的设备上
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
#         self.time_embedding = SinusoidalTimeEmbeddings(AM)
#
#         self.resize_conv = ResizeConv2d(AM, AM, target_size=480)  # zsyyyy
#         self.reverse_model = UNet(AM, AM)
#
#     def q_sample(self, x_start, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_start)
#
#         # sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
#         # sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
#         # 确保sqrt_recip_alphas_t与t在相同的设备上
#         sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(x_start.device).view(-1, 1, 1, 1)
#         sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device).view(-1, 1, 1, 1)
#
#         return sqrt_recip_alphas_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
#
#     def p_sample(self, x_t, t):
#         # time_emb = self.time_embedding(torch.tensor([t], device=x.device).repeat(x_t.size(0)))
#         time_emb = self.time_embedding(torch.tensor([t]).repeat(x_t.size(0)))
#         pred_noise = self.reverse_model(x_t, time_emb)
#
#         # 反向扩散公式
#         alphas_t = self.alphas[t].view(-1, 1, 1, 1)
#         alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
#         sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
#         sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
#
#         c0 = 1.0 / torch.sqrt(alphas_t)
#         c1 = (1 - alphas_t) / torch.sqrt(1 - alphas_cumprod_t)
#
#         if t > 0:
#             rand_noise = torch.randn_like(x_t)
#         else:
#             rand_noise = torch.zeros_like(x_t)
#
#         x_t_minus_1 = c0 * (x_t - c1 * pred_noise) + rand_noise * sqrt_one_minus_alphas_cumprod_t
#
#         return x_t_minus_1
#
#     def forward(self, x, L):
#         # print("!!!!", x.shape)
#         x1 = x
#
#         x = self.resize_conv(x)
#
#         timesteps = self.num_steps
#
#         batch_size = x.size(0)
#         t = torch.randint(0, timesteps, (batch_size,), device=x.device).long()
#
#         x_t = self.q_sample(x, t)
#
#         for t in reversed(range(timesteps)):
#             x = self.p_sample(x_t, t)
#
#         # x = resize(x.squeeze(0), out_shape=(x.shape[1], L, L))
#         x = F.interpolate(x, size=(L,L), mode='bilinear', align_corners=False)
#         x = x + x1
#         return x



# ######################第三次diffussion############################


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
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动获取设备
        self.device = device  # 保存设备信息
        self.AM = AM
        self.num_steps = num_steps
        # print("111",num_steps) ##0.0001
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)  # 确保在正确的设备上
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.time_embedding = SinusoidalTimeEmbeddings(AM)

        self.reverse_model = UNet(AM, AM)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        # sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        # 确保sqrt_recip_alphas_t与t在相同的设备上
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(x_start.device).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device).view(-1, 1, 1, 1)

        return sqrt_recip_alphas_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t):
        # time_emb = self.time_embedding(torch.tensor([t], device=x.device).repeat(x_t.size(0)))
        time_emb = self.time_embedding(torch.tensor([t]).repeat(x_t.size(0)))
        pred_noise = self.reverse_model(x_t, time_emb)

        # 反向扩散公式
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
        # print("xxxxxxxxx:",x.shape)
        # x = x.unsqueeze(dim=0)   ##[1, 1, 1, 120, 120]
        # print("xxxxxxxxx:",x.shape)
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
#
#
# if __name__ == '__main__':
#
#     device = 'cuda:0'
#     d_model = 256
#     num_steps = 1000
#     beta_start = 1e-4
#     beta_end = 0.02
#     AM = 64
#     # L = 100
#     # x = torch.randn([64, L, L]).to(device)
#
#     # if True == True:
#     #     L = x.size(-1)
#     #     print("LLLLLLLL:",L)
#     #     print("xxxxxxxxx:",x.shape)
#     #     diffC = DiffCorr(AM, num_steps, beta_start, beta_end).to(device)
#     #     x = diffC(x, L)
#     #     print("111", x.shape)       # [1, 8, 100, 100]
#     #     x = x.squeeze(dim=0)
#
#     #     print(x.shape)
#
#     #  加载目录下的 .npy 文件
#     directory_path = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/fuse_3/zyf/rep2/'
#     x_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if
#                filename.endswith('.npy')]
#     x_data = [torch.tensor(np.load(file)).to(device) for file in x_files]
#
#     for x in x_data:
#         L = x.size(-1)
#         print("LLLLLLLL:", L)
#         print("xxxxxxxxx:", x.shape)
#         diffC = DiffCorr(AM, num_steps, beta_start, beta_end).to(device)
#         x = diffC(x, L)
#         print("111", x.shape)  ##[1, 8, 100, 100]
#         x = x.squeeze(dim=0)
#
#         print(x.shape)

























# ####################第一次diffusion####################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# import math
# import numpy as np

# class SinusoidalTimeEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings

# class UNet(nn.Module):
#     def __init__(self, d_model):
#         super(UNet, self).__init__()

#         self.down1 = nn.Linear(d_model, 512)
#         self.down2 = nn.Linear(512, 1024)

#         self.up1 = nn.Linear(1024, 512)
#         self.up2 = nn.Linear(512, d_model)

#         self.norm1 = nn.LayerNorm(512)
#         self.norm2 = nn.LayerNorm(1024)
#         self.time_linear = nn.Linear(d_model, d_model)

#         self._initialize_weights()

#     def forward(self, x, time_embed):
#         time_embed = self.time_linear(time_embed)
#         time_embed = time_embed.unsqueeze(1).expand(x.size(0), x.size(1), -1)
#         x = x + time_embed

#         d1 = self.norm1(F.relu(self.down1(x)))
#         d2 = self.norm2(F.relu(self.down2(d1)))

#         u1 = self.norm1(F.relu(self.up1(d2)))
#         u2 = self.up2(u1)

#         return u2

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

# class DiffCorr(nn.Module):
#     def __init__(self, AM, d_model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
#         super(DiffCorr, self).__init__()
#         self.AM = AM
#         # self.L = L
#         self.num_steps = num_steps

#         self.input = nn.Sequential(
#             # nn.Linear(self.AM * self.L, 512),
#             nn.Linear(self.AM * 100, 512),  # 这里的 100 是占位符
#             nn.ReLU(True),
#             nn.LayerNorm(512),
#             nn.Dropout(p=0.1, inplace=False),
#             nn.Linear(512, 256),
#             nn.ReLU(True),
#             nn.LayerNorm(256)
#         )

#         self.betas = torch.linspace(beta_start, beta_end, num_steps)
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
#         self.reverse_model = UNet(d_model)
#         self.time_embedding = SinusoidalTimeEmbeddings(d_model)

#         self.sims = Similarity_matrix()

#     def q_sample(self, x_start, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_start)

#         return (
#             self.sqrt_recip_alphas[t] * x_start +
#             self.sqrt_one_minus_alphas_cumprod[t] * noise
#         )

#     def forward(self, x,L,timesteps=None):
#         # x = x.unsqueeze(dim=0)
#         x = x.permute(0, 2, 3, 1) ##[1, L, L, 64]  ##64是MSA的条数 也是训练输入MSA特征的维度

#         x = x.flatten(start_dim=2)  ##[1, L, 7488]

#         # 使用动态 L 更新 input 层的输入尺寸
#         self.input[0] = nn.Linear(self.AM * L, 512).to(x.device)

#         x = self.input(x)
    

#         if timesteps is None:
#             timesteps = self.num_steps

#         for t in reversed(range(timesteps)):
#             time_emb = self.time_embedding(torch.tensor([t], device=x.device).repeat(x.size(0)))
#             noise = self.reverse_model(x, time_emb)
#             x = self.q_sample(x, t, noise)

#         x_sims = F.relu(self.sims(x, x, x))
#         return x_sims

#     def sample(self, x_start, step_lr=1e-5):
#         batch_size = x_start.size(0)
#         x_T = torch.randn_like(x_start).to(x_start.device)

#         traj = {self.num_steps: x_T}

#         for t in tqdm(range(self.num_steps, 0, -1)):
#             time = torch.full((batch_size,), t, device=x_start.device)
#             time_emb = self.time_embedding(time)

#             alphas = self.alphas[t]
#             alphas_cumprod = self.alphas_cumprod[t]

#             c0 = 1.0 / torch.sqrt(alphas)
#             c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

#             # Corrector
#             rand_noise = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
#             step_size = step_lr * (self.sqrt_one_minus_alphas_cumprod[t] / self.sqrt_one_minus_alphas_cumprod[0]) ** 2
#             std_noise = torch.sqrt(2 * step_size)

#             pred_noise = self.reverse_model(x_T, time_emb)
#             x_t_minus_05 = x_T - step_size * pred_noise + std_noise * rand_noise

#             # Predictor
#             adjacent_sigma_x = self.sqrt_one_minus_alphas_cumprod[t-1]
#             step_size = (self.sqrt_one_minus_alphas_cumprod[t] ** 2 - adjacent_sigma_x ** 2)
#             std_noise = torch.sqrt((adjacent_sigma_x ** 2 * (self.sqrt_one_minus_alphas_cumprod[t] ** 2 - adjacent_sigma_x ** 2)) / (self.sqrt_one_minus_alphas_cumprod[t] ** 2))

#             pred_noise = self.reverse_model(x_t_minus_05, time_emb)
#             x_T = x_t_minus_05 - step_size * pred_noise + std_noise * rand_noise

#             traj[t - 1] = x_T

#         traj_stack = torch.stack([traj[i] for i in range(self.num_steps, -1, -1)])
#         return traj[0], traj_stack


# class Attention(nn.Module):
#     """Scaled dot-product attention mechanism."""
#     def __init__(self, scale=128, att_dropout=None):
#         super().__init__()
#         # self.dropout = nn.Dropout(attention_dropout)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(att_dropout)
#         self.scale = scale

#     def forward(self, q, k, v, attn_mask=None):
#         # q: [B, head, F, model_dim]
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)  # [B,Head, F, F]
#         if attn_mask:
#             scores = scores.masked_fill_(attn_mask, -np.inf)
#         scores = self.softmax(scores)
#         scores = self.dropout(scores)  # [B,head, F, F]
#         # context = torch.matmul(scores, v)  # output
#         return scores  # [B,head,F, F]

# class Similarity_matrix(nn.Module):
#     ''' buliding similarity matrix by self-attention mechanism '''
#     def __init__(self, num_heads=8, model_dim=256):
#         super().__init__()

#         # self.dim_per_head = model_dim // num_heads
#         self.num_heads = num_heads
#         self.model_dim = model_dim
#         self.input_size = model_dim
#         self.linear_q = nn.Linear(self.input_size, model_dim)
#         self.linear_k = nn.Linear(self.input_size, model_dim)
#         self.linear_v = nn.Linear(self.input_size, model_dim)

#         self.attention = Attention(att_dropout=0.1)
#         # self.out = nn.Linear(model_dim, model_dim)
#         # self.layer_norm = nn.LayerNorm(model_dim)

#     def forward(self, query, key, value, attn_mask=None):
#         batch_size = query.size(0)
#         # dim_per_head = self.dim_per_head
#         num_heads = self.num_heads
#         # linear projection
#         query = self.linear_q(query)  # [B,F,model_dim]
#         key = self.linear_k(key)
#         value = self.linear_v(value)
#         # split by heads
#         # [B,F,model_dim] ->  [B,F,num_heads,per_head]->[B,num_heads,F,per_head]
#         query = query.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
#         key = key.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
#         value = value.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
#         # similar_matrix :[B,H,F,F ]
#         matrix = self.attention(query, key, value, attn_mask)
#         return matrix
    
# import os

# if __name__ == '__main__':

#     device = 'cuda:0'
#     d_model = 256
#     num_steps = 1000
#     beta_start = 1e-4
#     beta_end = 0.02
#     AM = 64
# #     L = 100
# #     # x = torch.randn([64, L, L]).to(device)

#     # 加载目录下的 .npy 文件
#     directory_path = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/fuse_3/zyf/rep2/'
#     x_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.npy')]

#     x_data = [torch.tensor(np.load(file)).to(device) for file in x_files]
    
   

#     for x in x_data:
#         L=x.size(-1)
#         print("LLLLLLLL:",L)
#         print("xxxxxxxxx:",x.shape)
#         diffC = DiffCorr(AM,L, d_model, num_steps, beta_start, beta_end).to(device)
#         x = diffC(x)
#         print("111",x.shape) ##[1, 8, 100, 100]
#         x = x.squeeze(dim=0)

#         print(x.shape)
















# # ######################第二次diffussion############################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# import math
# import numpy as np
# import os
# import sys
# from .unet import UNet
# from resize_right import resize
#
# class SinusoidalTimeEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings
#
#
# class DiffCorr(nn.Module):
#     def __init__(self, AM,device,num_steps=1000, beta_start=1e-4, beta_end=0.02):
#         super(DiffCorr, self).__init__()
#         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动获取设备
#         self.device = device  # 保存设备信息
#         self.AM = AM
#         self.num_steps = num_steps
#         # print("111",num_steps) ##0.0001
#         self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device)
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         # self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
#         # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
#         self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)  # 确保在正确的设备上
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
#         self.time_embedding = SinusoidalTimeEmbeddings(AM)
#
#         self.reverse_model = UNet(AM, AM)
#
#     def q_sample(self, x_start, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_start)
#
#         # sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
#         # sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
#         # 确保sqrt_recip_alphas_t与t在相同的设备上
#         sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(x_start.device).view(-1, 1, 1, 1)
#         sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device).view(-1, 1, 1, 1)
#
#         return sqrt_recip_alphas_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
#
#     def p_sample(self, x_t, t):
#         # time_emb = self.time_embedding(torch.tensor([t], device=x.device).repeat(x_t.size(0)))
#         time_emb = self.time_embedding(torch.tensor([t]).repeat(x_t.size(0)))
#         pred_noise = self.reverse_model(x_t, time_emb)
#
#         # 反向扩散公式
#         alphas_t = self.alphas[t].view(-1, 1, 1, 1)
#         alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
#         sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
#         sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
#
#         c0 = 1.0 / torch.sqrt(alphas_t)
#         c1 = (1 - alphas_t) / torch.sqrt(1 - alphas_cumprod_t)
#
#         if t > 0:
#             rand_noise = torch.randn_like(x_t)
#         else:
#             rand_noise = torch.zeros_like(x_t)
#
#         x_t_minus_1 = c0 * (x_t - c1 * pred_noise) + rand_noise * sqrt_one_minus_alphas_cumprod_t
#
#         return x_t_minus_1
#
#     def forward(self, x, L, timesteps=None):
#         target_len = 120
#         x = resize(x, out_shape=(x.shape[0], target_len, target_len))  ##[1, 1, 120, 120]
#         # print("xxxxxxxxx:",x.shape)
#         # x = x.unsqueeze(dim=0)   ##[1, 1, 1, 120, 120]
#         # print("xxxxxxxxx:",x.shape)
#         if timesteps is None:
#             timesteps = self.num_steps
#
#         batch_size = x.size(0)
#         t = torch.randint(0, timesteps, (batch_size,), device=x.device).long()
#
#         x_t = self.q_sample(x, t)
#
#         for t in reversed(range(timesteps)):
#             x = self.p_sample(x_t, t)
#
#         x = resize(x.squeeze(0), out_shape=(x.shape[1], L, L))
#         return x
#
# if __name__ == '__main__':
#
#     device = 'cuda:0'
#     d_model = 256
#     num_steps = 1000
#     beta_start = 1e-4
#     beta_end = 0.02
#     AM = 64
#     # L = 100
#     # x = torch.randn([64, L, L]).to(device)
#
#
#     # if True == True:
#     #     L = x.size(-1)
#     #     print("LLLLLLLL:",L)
#     #     print("xxxxxxxxx:",x.shape)
#     #     diffC = DiffCorr(AM, num_steps, beta_start, beta_end).to(device)
#     #     x = diffC(x, L)
#     #     print("111", x.shape)       # [1, 8, 100, 100]
#     #     x = x.squeeze(dim=0)
#
#     #     print(x.shape)
#
#     #  加载目录下的 .npy 文件
#     directory_path = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/fuse_3/zyf/rep2/'
#     x_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.npy')]
#     x_data = [torch.tensor(np.load(file)).to(device) for file in x_files]
#
#     for x in x_data:
#         L=x.size(-1)
#         print("LLLLLLLL:",L)
#         print("xxxxxxxxx:",x.shape)
#         diffC = DiffCorr(AM, num_steps, beta_start, beta_end).to(device)
#         x = diffC(x, L)
#         print("111",x.shape) ##[1, 8, 100, 100]
#         x = x.squeeze(dim=0)
#
#         print(x.shape)








