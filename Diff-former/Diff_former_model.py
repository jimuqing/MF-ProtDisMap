from .Diff_former_parts import *
from .tri import *
from .diffusion import DiffCorr
import numpy as np
import os

class Diff_former(nn.Module):
    def __init__(self, n_channels, L, bilinear=False):
        super(Diff_former, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.layer1 = Forward_conv(n_channels, 48)
        self.tri_1 = TriangularSelfAttentionBlock(48)
        self.layer2 = Forward_former(48, 128)
        self.layer3 = Mix_former(128, 48, bilinear)
        self.layer4 = Mix_conv(48, n_channels, bilinear)

        self.lr = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.pMAE = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        e_layers = 2
        device = 'cuda:0'
        self.diff = DiffCorr(24, device).to(device)


    def forward(self, xE, xT, L, name=None):

        L = int(L.item())

        device = xT.device
        xE = xE.to(device)


        xD = self.diff(xT, L)  # Diffusion + Corr  torch.Size([64, 117, 117])

        save_dir = "/"

        os.makedirs(save_dir, exist_ok=True)
        if name:
            base_name = os.path.splitext(name)[0]
            xt_filename = f"{base_name}_T.txt"
            xd_filename = f"{base_name}_D.txt"
        else:
            xt_filename = "matrix_T_avg.txt"
            xd_filename = "matrix_D_avg.txt"

        matrix_T = xT.mean(dim=1).squeeze(0).detach().cpu().numpy()
        matrix_D = xD.mean(dim=1).squeeze(0).detach().cpu().numpy()

        np.savetxt(os.path.join(save_dir, xt_filename), matrix_T, fmt='%.6f')
        np.savetxt(os.path.join(save_dir, xd_filename), matrix_D, fmt='%.6f')


        x1 = torch.cat((xD, xE), dim=1)

        x4 = self.layer1(x1)

        x4 = self.tri_1(x4.permute(0, 2, 3, 1))

        x4 = x4.permute(0, 3, 1, 2)
        x5 = self.layer2(x4)

        x = self.layer3(x5, x4)
        x = self.layer4(x, x1)

        pMAE = self.pMAE(x)
        x = self.lr(x)

        x = self.sigmoid(x)
        pMAE = self.sigmoid(pMAE)
        return x, pMAE
