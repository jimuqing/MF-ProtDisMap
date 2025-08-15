
import torch
import torch.nn as nn
import esm
import pickle
import numpy as np
import os
import torch.nn.functional as F
from einops import rearrange

from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from scipy.spatial.distance import cdist
from Bio import SeqIO


torch.set_grad_enabled(False)


class esmmsa1b(object):

    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    def __init__(self, msa):  # str->file
        self.msa = self.read_msa(msa)

    def remove_insertions(self, sequence: str) -> str:
        return sequence.translate(self.translation)

    def read_msa(self, filename: str) -> List[Tuple[str, str]]:
        return [(record.description, self.remove_insertions(str(record.seq))) for record in
                SeqIO.parse(filename, "fasta")]

    def greedy_select(self, msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
        assert mode in ("max", "min")

        if len(msa) <= num_seqs:
            return msa

        array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

        optfunc = np.argmax if mode == "max" else np.argmin
        all_indices = np.arange(len(msa))
        indices = [0]
        pairwise_distances = np.zeros((0, len(msa)))
        for _ in range(num_seqs - 1):
            dist = cdist(array[indices[-1:]], array, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = optfunc(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        return [msa[idx] for idx in indices]

    def get_embedding(self, num_seqs=64):
        msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        msa_transformer = msa_transformer.eval().cuda()
 
        msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

        inputs = self.greedy_select(self.msa, num_seqs=num_seqs)
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter(
            [inputs])
        batch_lens = (msa_transformer_batch_tokens != msa_transformer_alphabet.padding_idx).sum(-1)
        msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)

        if msa_transformer_batch_tokens.shape[2] > 1024:
            index = msa_transformer_batch_tokens.shape[2] - 1024
            msa_transformer_batch_tokens1 = msa_transformer_batch_tokens[:, :, :1024]
            msa_transformer_batch_tokens2 = msa_transformer_batch_tokens[:, :, -1024:]
            results1 = msa_transformer(msa_transformer_batch_tokens1, repr_layers=[12], return_contacts=True)
            results2 = msa_transformer(msa_transformer_batch_tokens2, repr_layers=[12], return_contacts=True)
            results = torch.cat((results1[:, :, :index, :],
                                 (results1[:, :, index:, :] + results2[:, :, :-index, :]) / 2,
                                 results2[:, :, -index:, :]), dim=2)
        else:
            results = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12], return_contacts=True)

        return results,batch_lens


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('The code runs on {}'.format(device))
    protein_embeddings = {}

    def traverse_directories(start_path):
        for dirpath, dirnames, filenames in os.walk(start_path):
            for filename in filenames:
                print(f"文件: {filename}")
                a = os.path.join(dirpath,filename)
                results,batch_lens = esmmsa1b(a).get_embedding()
                

                row_attention = results["row_attentions"]
                row_attention = row_attention.squeeze(0)

                row_attention = row_attention[:, :, 1:batch_lens[0,0], 1: batch_lens[0,0]]
                # print("Shape of row_attention:",row_attention.shape)
                row_pred_all = row_attention.detach().cpu().numpy()
                row_pred = np.max(row_pred_all, axis=1)

                col_attention = results["col_attentions"]

                target_size = col_attention.size(3) 
                weidu_size = col_attention.size(-1)
                L = target_size-1
                col_attention = col_attention.squeeze(0)
                col_attention = col_attention[:, :, 1:batch_lens[0,0], :,:]

                x = col_attention.view(col_attention.size(0),col_attention.size(1),col_attention.size(2), -1)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                linear_layer = nn.Linear(weidu_size*weidu_size, L).to(device)
                x = x.to(device)
                pred_mapped = linear_layer(x)
                relu = nn.ReLU().to(device)
                col_attention = relu(pred_mapped)
                
                pred_all = pred_mapped.detach().cpu().numpy()
                pred = np.max(pred_all, axis=1)
                

                output_folder = '/data/'
                fused_array = np.concatenate((row_pred, pred), axis=0)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                filename= os.path.splitext(filename)[0]
                save_path = os.path.join(output_folder, filename.strip() + ".npy")
                np.save(save_path, fused_array)
                print(f"row_col_attention  {save_path}")
                
                torch.cuda.empty_cache()


    start_path = '/data/home/'
    traverse_directories(start_path)
   
