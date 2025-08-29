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

MAX_LEN = 1024
OVERLAP = 128


def sliding_window(seq, max_len=MAX_LEN, overlap=OVERLAP):
    step = max_len - overlap
    slices = []
    for start in range(0, len(seq), step):
        end = min(start + max_len, len(seq))
        slices.append((start, seq[start:end]))
        if end == len(seq):
            break
    return slices


with open("sequence_length_info.txt", 'w') as log_file:
    for File in Files:
        name = os.path.basename(File)
        File = open(args.source_dir + File, 'r')
        File = File.readlines()
        name = os.path.splitext(name)[0]
        seq = File[1].strip()

        seq_len = len(seq)
        print("L：", seq_len)
        log_file.write(f"Name: {name}, Length: {seq_len}\n")

        if seq_len <= MAX_LEN:
            segments = [(0, seq)]
        else:
            segments = sliding_window(seq)

        all_embeddings = []
        all_positions = []

        for (start_idx, subseq) in segments:
            data = [(name, subseq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                batch_tokens = batch_tokens.cuda()
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)

            outputs = results["attentions"][:, :, :, 1:batch_lens - 1, 1:batch_lens - 1]
            outputs = outputs.squeeze()
            outputs = rearrange(outputs, 'l c h w -> (l c) h w')
            outputs = rearrange(outputs, '(g c) h w -> g c h w', g=group_dim)

            pred_all = outputs.detach().cpu().numpy()
            pred = np.max(pred_all, axis=1)

            all_embeddings.append(pred)
            all_positions.append((start_idx, start_idx + len(subseq)))

            torch.cuda.empty_cache()

        full_len = seq_len
        final_matrix = np.zeros((full_len, full_len))
        counts = np.zeros((full_len, full_len))

        for (seg, (s, e)) in zip(all_embeddings, all_positions):
            seg_len = e - s
            final_matrix[s:e, s:e] += seg[:seg_len, :seg_len]
            counts[s:e, s:e] += 1

        counts[counts == 0] = 1
        final_matrix = final_matrix / counts

        np.save(os.path.join(args.save_dir, name.strip()), final_matrix)

print("successful")
