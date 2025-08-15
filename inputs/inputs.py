from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import os

def get_L(dir_attention, name):
    inputs_attention = np.load(dir_attention+name,allow_pickle=True)
    inputs = torch.from_numpy(inputs_attention)
    return inputs.shape[2]

def get_train_data(ESM_atteention,MSA_atteention,distance_dir,files,item):
    name = files[item]

    # inputs_attention = np.load(dir_attention+name,allow_pickle=True)
    # inputs = torch.from_numpy(inputs_attention)

    input_ESM = np.load(ESM_atteention+name,allow_pickle=True)   # path of ESM
    input_ESM = torch.from_numpy(input_ESM)   ##torch.Size([33, 117, 117])
    input_MSA = np.load(MSA_atteention+name,allow_pickle=True)   # path of MSA
    input_MSA = torch.from_numpy(input_MSA)   ##torch.Size([64, 117, 117])

    label = np.load(distance_dir+ name.split('.')[0]+'.npz',allow_pickle=True)
    label = label['dist_ca']

    L     =  input_ESM.shape[2]
    L     =  torch.ones(1)*L

    label = torch.from_numpy(label).squeeze(0)

    return input_ESM.cuda(),input_MSA.cuda(),label.cuda(),L.cuda(),name

def get_test_data(ESM_feature,MSA_feature,dir_label,files,item):
    name = files[item]

    inputs_ESM = np.load(ESM_feature + name,allow_pickle=True)
    inputs_ESM = torch.from_numpy(inputs_ESM)
    inputs_MSA = np.load(MSA_feature+name,allow_pickle=True)   # path of MSA
    inputs_MSA = torch.from_numpy(inputs_MSA)

    name = name.split('.')[0]+'.txt'
    label = np.loadtxt(dir_label + name)
    # label = np.loadtxt(dir_label +'label/'+ name)
    L     =  inputs_ESM.shape[2]
    L     =  torch.ones(1)*L

    label = torch.from_numpy(label).squeeze(0)

    return inputs_ESM.cuda(),inputs_MSA.cuda(),label.cuda(),L.cuda(), name


def collate_fn(batch):
    pad_index = 0
    lens = [ESM.size(2) for ESM,MSA,label,L,file_name in batch]
    lens.sort()
    seq_len = lens[-1]

    pad_ESM_list = []
    pad_MSA_list = []
    pad_label_list = []
    pad_L_list = []

    file_name_list = []

    for ESM,MSA,label,L,file_name in batch:

        pad_len = seq_len - ESM.size(2)

        label = F.pad(label, (0,pad_len,0,pad_len), "constant", 0)
        ESM = F.pad(ESM, (0,pad_len,0,pad_len), "constant", 0)
        MSA = F.pad(MSA, (0,pad_len,0,pad_len), "constant", 0)


        pad_ESM_list.append(ESM)
        pad_MSA_list.append(MSA)
        pad_label_list.append(label)
        pad_L_list.append(L)
        file_name_list.append(file_name)



    pad_ESM = torch.stack(pad_ESM_list, 0)
    pad_MSA = torch.stack(pad_MSA_list, 0)
    pad_label = torch.stack(pad_label_list, 0)
    pad_L = torch.stack(pad_L_list, 0)

    # return pad_inputs.cuda(),pad_label.cuda(),pad_L.cuda()
    return pad_ESM.cuda(),pad_MSA.cuda(), pad_label.cuda(), pad_L.cuda(),file_name_list



class MyData(Dataset):
    def __init__(self,train_num,ESM,MSA,label,loader,transform=None, target_transform=None):
        super(MyData,self).__init__()

        self.ESM =  ESM
        self.MSA =  MSA
        self.label =   label

        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader
        self.train_num = train_num

        self.files = os.listdir(self.ESM)
        self.L = get_L(self.ESM, self.files[0])


    def __getitem__(self, item):

        file_name = self.files[item]

        input_ESM,input_MSA,label,L,file_name= self.loader(self.ESM,self.MSA,self.label,self.files,item)
        return input_ESM, input_MSA,label, L, file_name

    def __len__(self):
        return self.train_num

class TestData(Dataset):
    def __init__(self,loader, test_ESM,test_MSA,test_label,transform=None, target_transform=None):
        super(TestData,self).__init__()

        self.test_ESM = test_ESM
        self.test_MSA = test_MSA
        self.test_label = test_label


        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


        self.files = os.listdir(self.test_ESM)
        self.L = get_L(self.test_ESM, self.files[0])

    def __getitem__(self, item):

        input_ESM,input_MSA,label,L, name = self.loader(self.test_ESM,self.test_MSA,self.test_label,self.files,item)
        return input_ESM,input_MSA,label,L, name

    def __len__(self):
        return len(self.files)


