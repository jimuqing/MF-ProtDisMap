import os
import sys
import torch
import torch.nn as nn
sys.path.append("..")
from argparse import ArgumentParser
from inputs.inputs import *
from train.train_process import train
from model.R_former.R_former_model import R_former
torch.cuda.device_count()    ##8
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#####对最基础的 只有行+列 24层
parser = ArgumentParser(description='')               ##日志文件路径
parser.add_argument('--log', default="/data/home/scyb035/zhangyufei/FreeProtMap-diffusion/weights/weight.pth")
parser.add_argument('-train_num', default=14930)      ##14930
parser.add_argument('-train_batchsize', default=1)    ##训练批次大小，默认值为1。

parser.add_argument('--ESM', default='/data/home/scyb035/zhangyufei/datasets/feature_ESM/rep1_t36_1/')          ##特征文件路径，33层
parser.add_argument('--MSA', default='/data/home/scyb035/zhangyufei/datasets/feature_MSA/R_C_t24/')    ##行+列,24层
parser.add_argument('--label', default='/data/home/scyb035/zhangyufei/datasets/npz/')            ##标签文件路径
parser.add_argument('--save_dir', default='/data/home/scyb035/zhangyufei/datasets/train_result/diff_250311/')         ##保存目录路径
parser.add_argument('-epoch', default=200)             ##训练的轮数,作者是30





if __name__ =="__main__":

        args = parser.parse_args()

        traindata = MyData(args.train_num, args.ESM, args.MSA, args.label, get_train_data)
        traindata_loader = DataLoader(traindata, batch_size=args.train_batchsize, shuffle=False, drop_last=True, collate_fn=collate_fn)
        L = traindata.L

        model = R_former(60, L).cuda() ##36是通道数 n_channels 原本预训练模型是36层 我用的33层的
        model = nn.DataParallel(model)  ##实现GPU并行功能


        criterion_L1 = nn.SmoothL1Loss(beta=1.0, reduction='mean')

        ##lr是学习率
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

        train(model,criterion_L1,optimizer,args,traindata_loader)







####################从当前权重继续训练#####################
#
# import os
# import sys
# import torch
# import torch.nn as nn
#
# sys.path.append("..")
# from argparse import ArgumentParser
# from inputs.inputs import *
# from train.train_process import train
# from model.R_former.R_former_model import R_former
#
# torch.cuda.device_count()  ##8
# # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#
# #####对最基础的 只有行+列 24层
# parser = ArgumentParser(description='')  ##日志文件路径
# parser.add_argument('--log', default="/data/home/scyb035/zhangyufei/FreeProtMap-diffusion/weights/weight.pth")
# parser.add_argument('-train_num', default=14930)  ##14930
# parser.add_argument('-train_batchsize', default=1)  ##训练批次大小，默认值为1。
#
# parser.add_argument('--ESM', default='/data/home/scyb035/zhangyufei/datasets/feature_ESM/rep1_t36_1/')  ##特征文件路径，33层
# parser.add_argument('--MSA', default='/data/home/scyb035/zhangyufei/datasets/feature_MSA/R_C_t24/')  ##行+列,24层
# parser.add_argument('--label', default='/data/home/scyb035/zhangyufei/datasets/npz/')  ##标签文件路径
# parser.add_argument('--save_dir',
#                     default='/data/home/scyb035/zhangyufei/datasets/train_result/diff_250311/')  ##保存目录路径
# parser.add_argument('-epoch', default=3)  ##训练的轮数,作者是30
# parser.add_argument('--resume',
#                     default='/data/home/scyb035/zhangyufei/datasets/train_result/diff_t36_250117/99.pth')  # 加载的权重路径
#
# if __name__ == "__main__":
#         args = parser.parse_args()
#
#         traindata = MyData(args.train_num, args.ESM, args.MSA, args.label, get_train_data)
#         traindata_loader = DataLoader(traindata, batch_size=args.train_batchsize, shuffle=False, drop_last=True,
#                                       collate_fn=collate_fn)
#         L = traindata.L
#
#         model = R_former(60, L).cuda()  ##36是通道数 n_channels 原本预训练模型是36层 我用的33层的
#         model = nn.DataParallel(model)  ##实现GPU并行功能
#
#         # 加载权重
#         if os.path.exists(args.resume):
#                 print(f"Loading model weights from {args.resume}...")
#                 # 加载整个模型对象
#                 loaded_model = torch.load(args.resume)
#
#                 # 如果加载的模型是 DataParallel 包装过的，提取内部模型
#                 if isinstance(loaded_model, nn.DataParallel):
#                         model = loaded_model
#                 else:
#                         model.module.load_state_dict(loaded_model.state_dict())
#         else:
#                 print(f"No weights found at {args.resume}. Starting from scratch.")
#
#         criterion_L1 = nn.SmoothL1Loss(beta=1.0, reduction='mean')
#
#         ##lr是学习率
#         optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
#
#         train(model, criterion_L1, optimizer, args, traindata_loader)
