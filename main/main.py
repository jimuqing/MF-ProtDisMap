import os
import sys
import torch
import torch.nn as nn
sys.path.append("..")
from argparse import ArgumentParser
from inputs.inputs import *
from train.train_process import train
from model.R_former.R_former_model import R_former
torch.cuda.device_count()
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


parser = ArgumentParser(description='')
parser.add_argument('--log', default="/weight.pth")
parser.add_argument('-train_num', default=15000)
parser.add_argument('-train_batchsize', default=4)

parser.add_argument('--ESM', default='/data/……')
parser.add_argument('--MSA', default='/data/……')
parser.add_argument('--label', default='/data/……')
parser.add_argument('--save_dir', default='/data/……')
parser.add_argument('-epoch', default=200)





if __name__ =="__main__":

        args = parser.parse_args()

        traindata = MyData(args.train_num, args.ESM, args.MSA, args.label, get_train_data)
        traindata_loader = DataLoader(traindata, batch_size=args.train_batchsize, shuffle=False, drop_last=True, collate_fn=collate_fn)
        L = traindata.L
        name = traindata.files


        model = R_former(60, L).cuda()
        model = nn.DataParallel(model)


        criterion_L1 = nn.SmoothL1Loss(beta=1.0, reduction='mean')

        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

        train(model,criterion_L1,optimizer,args,traindata_loader)




