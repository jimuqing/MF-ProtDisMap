# ##############第四次扩散修改 2025年1月14日#############
#
# from .R_former_parts import *
# from .tri import *
# from .diffusion import DiffCorr
#
#
# class R_former(nn.Module):
#     def __init__(self, n_channels, L, bilinear=False):
#         super(R_former, self).__init__()
#         self.n_channels = n_channels
#         self.bilinear = bilinear
#
#         self.layer1 = Forward_conv(n_channels, 48)
#         self.tri_1 = TriangularSelfAttentionBlock(48)
#         self.layer2 = Forward_former(48, 128)
#         self.layer3 = Mix_former(128, 48, bilinear)
#         self.layer4 = Mix_conv(48, n_channels, bilinear)
#
#         self.lr = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
#         self.pMAE = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         # #self.diff = DiffCorr(64, d_model=256)
#         # self.diff = DiffCorr(64,device).to(device)
#         e_layers = 2
#         device = 'cuda:0'
#
#         self.diff1 = DiffCorr(24, device, num_steps=100)
#         # self.diff2 = DiffCorr(64, device, num_steps=50)
#
#     def forward(self, xE, xT, L):
#         L = int(L.item())
#
#         # print("我是xE", xE.shape)
#         # print("我是xT", xT.shape)
#
#         device = xT.device
#         xE = xE.to(device)
#
#         xD = self.diff1(xT, L)  # Diffusion + Corr  torch.Size([64, 117, 117])
#         # xD = self.diff2(xD, L)
#         # xD = xD.unsqueeze(dim=0)
#         # print("我是xD", xD.shape)
#         """
#         xE: [1, 33, L, L]
#         xD: [1, 8,  L, L]
#         xT: [1, 64,  L, L]
#         """
#         xT = xT + xD
#         # print("我是新的MSA特征",xT.shape)  ## torch.Size([1, 64, 117, 117])
#
#         x1 = torch.cat((xT, xE), dim=1)
#         # print("我是融合后特征", x1.shape)  ## torch.Size([1, 97, 117, 117])
#
#         x4 = self.layer1(x1)
#
#         x4 = self.tri_1(x4.permute(0, 2, 3, 1))
#
#         x4 = x4.permute(0, 3, 1, 2)
#         x5 = self.layer2(x4)
#
#         x = self.layer3(x5, x4)
#         x = self.layer4(x, x1)
#
#         pMAE = self.pMAE(x)
#         x = self.lr(x)
#
#         x = self.sigmoid(x)
#         pMAE = self.sigmoid(pMAE)
#         return x, pMAE















##############第三次扩散修改 2025年1月4日#############
from .R_former_parts import *
from .tri import *
from .diffusion import DiffCorr


class R_former(nn.Module):
    def __init__(self, n_channels, L, bilinear=False):
        super(R_former, self).__init__()
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
        # self.diff = nn.ModuleList([DiffCorr(64, device) for _ in range(e_layers)])
        self.diff = DiffCorr(24, device).to(device)


    def forward(self, xE, xT, L):  # xE  xT ->(split) xTr  xTc    xTr-Linear xTc-Linear xT=xTr+xTc
        L = int(L.item())

        # print("我是xE",xE.shape)
        # print("我是xT",xT.shape)

        device = xT.device
        xE = xE.to(device)
        print("我是xT",xT.shape)


        xD = self.diff(xT, L)  # Diffusion + Corr  torch.Size([64, 117, 117])


        # # 定义文件路径
        # file_path = '/data/home/scyb035/zhangyufei/FreeProtMap-diffusion/model/R_former/test1.txt'
        #
        # # 打开文件并写入xT的内容和注释
        # with open(file_path, 'w') as f:
        #     f.write(f"# xT的形状: {xT.shape}\n")  # 写入注释
        #     # f.write(f"{xT}\n")  # 写入xT的内容
        #     f.write(f"# xT的第一个样本的第一个通道内容（形状: {xT[0, 0].shape}）:\n")
        #     f.write(f"{xT[0, 0]}\n")  # 写入xT的第一个样本的第一个通道内容
        #
        # # 计算xD
        # xD = self.diff(xT, L)  # Diffusion + Corr  torch.Size([64, 117, 117])
        # print("我是xD", xD.shape)
        #
        # # 追加写入xD的内容和注释
        # with open(file_path, 'a') as f:  # 使用 'a' 模式追加内容
        #     f.write(f"# xD的形状: {xD.shape}\n")  # 写入注释
        #     # f.write(f"{xD}\n")  # 写入xD的内容
        #     f.write(f"# xD的第一个样本的第一个通道内容（形状: {xD[0, 0].shape}）:\n")
        #     f.write(f"{xD[0, 0]}\n")  # 写入xD的第一个样本的第一个通道内容




        """
        xE: [1, 33, L, L]  ## ESM2
        xD: [1, 8,  L, L]  ## Diffusion
        xT: [1, 64,  L, L] ## MSA + Diff
        """
        # xT = xT + xD
        # print("我是新的MSA特征",xT.shape)  ## torch.Size([1, 64, 117, 117])

        x1 = torch.cat((xD, xE), dim=1)
        # print("我是融合后特征",x1.shape)   ## torch.Size([1, 97, 117, 117])

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

















#############第一次扩散
# from .R_former_parts import *
# from .tri import *
# from .diffusion import DiffCorr


# class R_former(nn.Module):
#     def __init__(self, n_channels,L,bilinear=False):
#         super(R_former, self).__init__()
#         self.n_channels = n_channels
#         self.bilinear = bilinear

#         self.layer1 = Forward_conv(n_channels, 48)
#         self.tri_1 = TriangularSelfAttentionBlock(48)
#         self.layer2 = Forward_former(48, 128)
#         self.layer3 = Mix_former(128, 48, bilinear)
#         self.layer4 = Mix_conv(48, n_channels, bilinear)
        
        
#         self.lr = nn.Conv2d(n_channels, 1, kernel_size=1,bias=False)
#         self.pMAE = nn.Conv2d(n_channels, 1, kernel_size=1,bias=False)
#         self.sigmoid = torch.nn.Sigmoid()

#         self.diff = DiffCorr(24, d_model=256)

#     def forward(self, xE, xT, L):
#         L = int(L.item())
#         xD = self.diff(xT,L)  # Diffusion + Corr
       
#         """
#         xE: [1, 33, L, L]
#         xD: [1, 8,  L, L]
#         xT: [1, 64,  L, L]
#         """

#         x1 = torch.cat((xT, xD, xE),dim=1)

#         x4 = self.layer1(x1)
     
#         x4 = self.tri_1(x4.permute(0,2,3,1))

#         x4 = x4.permute(0,3,1,2)
#         x5 = self.layer2(x4)
       
#         x = self.layer3(x5, x4)
#         x = self.layer4(x, x1)
  

#         pMAE = self.pMAE(x)
#         x = self.lr(x)
        
#         x = self.sigmoid(x)
#         pMAE = self.sigmoid(pMAE)                
#         return x,pMAE

    












# ##############第二次扩散修改#############
# from .R_former_parts import *
# from .tri import *
# from .diffusion import DiffCorr
#
#
# class R_former(nn.Module):
#     def __init__(self, n_channels,L,bilinear=False):
#         super(R_former, self).__init__()
#         self.n_channels = n_channels
#         self.bilinear = bilinear
#
#         self.layer1 = Forward_conv(n_channels, 48)
#         self.tri_1 = TriangularSelfAttentionBlock(48)
#         self.layer2 = Forward_former(48, 128)
#         self.layer3 = Mix_former(128, 48, bilinear)
#         self.layer4 = Mix_conv(48, n_channels, bilinear)
#
#
#         self.lr = nn.Conv2d(n_channels, 1, kernel_size=1,bias=False)
#         self.pMAE = nn.Conv2d(n_channels, 1, kernel_size=1,bias=False)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         # self.diff = DiffCorr(64, d_model=256)
#         device = 'cuda:0'
#         self.diff = DiffCorr(24,device).to(device)
#         # self.diff = DiffCorr(64).to(device)
#
#     def forward(self, xE, xT, L):
#         L = int(L.item())
#
#         # print("我是xE",xE.shape)
#         # print("我是xT",xT.shape)
#
#         device = xT.device
#         xE = xE.to(device)
#
#
#         xD = self.diff(xT, L) # Diffusion + Corr  torch.Size([64, 117, 117])
#         xD = xD.unsqueeze(dim=0)
#         # print("我是xD",xD.shape)
#         """
#         xE: [1, 33, L, L]
#         xD: [1, 8,  L, L]
#         xT: [1, 64,  L, L]
#         """
#         xT = xT + xD
#         # print("我是新的MSA特征",xT.shape)  ## torch.Size([1, 64, 117, 117])
#
#         x1 = torch.cat((xT, xE),dim=1)
#         # print("我是融合后特征",x1.shape)   ## torch.Size([1, 97, 117, 117])
#
#         x4 = self.layer1(x1)
#
#         x4 = self.tri_1(x4.permute(0,2,3,1))
#
#         x4 = x4.permute(0, 3, 1, 2)
#         x5 = self.layer2(x4)
#
#         x = self.layer3(x5, x4)
#         x = self.layer4(x, x1)
#
#
#         pMAE = self.pMAE(x)
#         x = self.lr(x)
#
#         x = self.sigmoid(x)
#         pMAE = self.sigmoid(pMAE)
#         return x, pMAE
#