# # import esm.esm.data as esm
import os
import sys
sys.path.append("..") 
from einops import rearrange
from argparse import ArgumentParser
import torch 
import torch.nn as nn
from inputs.inputs import *
import numpy as np
import esm


group_dim = 36
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
parser = ArgumentParser(description='')
parser.add_argument('-source_dir', default="")
parser.add_argument('-save_dir', default="")
args = parser.parse_args()


Files = os.listdir(args.source_dir)
model = model.cuda()

with open("sequence_length_info.txt", 'w') as log_file:
    for File in Files:
        name = os.path.basename(File)
        File = open(args.source_dir+File,'r')
        File = File.readlines()
        # name = File[0][1:].strip()
        name = os.path.splitext(name)[0]
        # print("name:",name)

        seq = File[1].strip()
        
        seq_len = len(seq)
        print("L：",seq_len)

        log_file.write(f"Name: {name}, Length: {seq_len}\n")

        if seq_len > 1022:
            log_file.write(f"Sequence {name} is longer than 1000, skipping this file.\n")
            continue
        
        
        data = [(name, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)     

        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        # print("Results keys:", results.keys())
        representations=results["representations"]
        attention=results["attentions"]
        contacts=results["contacts"]

        # print("Shape of attention:",attention.shape)

        outputs =  results["attentions"][:,:,:, 1: batch_lens-1,1: batch_lens-1]

        outputs =  outputs.squeeze()

        outputs = rearrange(outputs, 'l c h w -> (l c) h w')
        outputs =  rearrange(outputs, '(g c) h w -> g c h w', g =group_dim)

        pred_all = outputs.detach().cpu().numpy()

        pred = np.max(pred_all, axis=1)

        np.save(args.save_dir+name.strip(),pred)

        
        torch.cuda.empty_cache()

print("运行完成")
