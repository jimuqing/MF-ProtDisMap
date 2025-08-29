import torch
import torch.nn as nn
import esm
import numpy as np
import os
from scipy.spatial.distance import cdist
from Bio import SeqIO
import string

torch.set_grad_enabled(False)


class esmmsa1b(object):
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    def __init__(self, msa_file):
        self.msa = self.read_msa(msa_file)

    def remove_insertions(self, sequence: str) -> str:
        return sequence.translate(self.translation)

    def read_msa(self, filename: str) -> List[Tuple[str, str]]:
        return [(record.description, self.remove_insertions(str(record.seq)))
                for record in SeqIO.parse(filename, "fasta")]

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

        return [msa[idx] for idx in sorted(indices)]

    def get_embedding(self, num_seqs=64, window_size=1024, overlap=128):

        msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        msa_transformer = msa_transformer.eval().cuda()
        msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

        inputs = self.greedy_select(self.msa, num_seqs=num_seqs)
        batch_labels, batch_strs, batch_tokens = msa_transformer_batch_converter([inputs])
        batch_lens = (batch_tokens != msa_transformer_alphabet.padding_idx).sum(-1)
        batch_tokens = batch_tokens.to(next(msa_transformer.parameters()).device)

        seq_length = batch_tokens.shape[2]

        if seq_length <= window_size:
            results = msa_transformer(batch_tokens, repr_layers=[12], return_contacts=True)
            return results, batch_lens

        step = window_size - overlap  
        num_chunks = (seq_length - window_size) // step + 1  

        all_representations = []
        all_row_attentions = []
        all_col_attentions = []

        for i in range(num_chunks):
            start = i * step
            end = start + window_size

            if end > seq_length:
                end = seq_length
                start = end - window_size

            chunk_tokens = batch_tokens[:, :, start:end]

            chunk_results = msa_transformer(chunk_tokens, repr_layers=[12], return_contacts=True)

            all_representations.append(chunk_results["representations"][12])
            all_row_attentions.append(chunk_results["row_attentions"])
            all_col_attentions.append(chunk_results["col_attentions"])

        merged_repr = self.merge_chunks(all_representations, window_size, overlap, step, seq_length)
        merged_row_attn = self.merge_chunks(all_row_attentions, window_size, overlap, step, seq_length,
                                            is_attention=True, attention_type="row")
        merged_col_attn = self.merge_chunks(all_col_attentions, window_size, overlap, step, seq_length,
                                            is_attention=True, attention_type="col")

        results = {
            "representations": {12: merged_repr},
            "row_attentions": merged_row_attn,
            "col_attentions": merged_col_attn
        }

        return results, batch_lens

    def merge_chunks(self, chunks, window_size, overlap, step, seq_length, is_attention=False, attention_type=None):
   
        if is_attention:
   
            if attention_type == "row":

                batch_size, num_layers, num_heads, _, _ = chunks[0].shape
                merged = torch.zeros(
                    (batch_size, num_layers, num_heads, seq_length, seq_length),
                    device=chunks[0].device
                )
   
                weights = torch.zeros_like(merged)
            else:  # col_attention

                batch_size, num_layers, num_heads, _, *rest_dims = chunks[0].shape
                merged = torch.zeros(
                    (batch_size, num_layers, num_heads, seq_length, *rest_dims),
                    device=chunks[0].device
                )
                weights = torch.zeros_like(merged)
        else:
            batch_size, num_seqs, _, feat_dim = chunks[0].shape
            merged = torch.zeros(
                (batch_size, num_seqs, seq_length, feat_dim),
                device=chunks[0].device
            )
            weights = torch.zeros_like(merged)

        overlap_weights = torch.linspace(0, 1, overlap, device=chunks[0].device)

        for i, chunk in enumerate(chunks):
            start = i * step
            end = start + window_size

            if end > seq_length:
                end = seq_length
                start = end - window_size
                current_window_size = window_size
            else:
                current_window_size = window_size

            chunk_weights = torch.ones(current_window_size, device=chunk.device)

            if i > 0: 
                chunk_weights[:overlap] = overlap_weights
            if i < len(chunks) - 1:  
                chunk_weights[-overlap:] = 1 - overlap_weights

 
            if is_attention:
                if attention_type == "row":

                    chunk_weights_2d = torch.outer(chunk_weights, chunk_weights)
                    expanded_weights = chunk_weights_2d[None, None, :, :]  # [1, 1, window, window]
                    expanded_weights = expanded_weights.expand_as(chunk)
                else:
  
                    expanded_weights = chunk_weights[None, None, :, None]  # [1, 1, window, 1, ...]
                    expanded_weights = expanded_weights.expand_as(chunk)
            else:
                expanded_weights = chunk_weights[None, None, :, None]  # [1, 1, window, 1]
                expanded_weights = expanded_weights.expand_as(chunk)

            merged[:, :, start:end] += chunk * expanded_weights
            weights[:, :, start:end] += expanded_weights

        merged = merged / weights.clamp(min=1e-8)  

        return merged


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'The code runs on {device}')


    def traverse_directories(start_path):
        for dirpath, dirnames, filenames in os.walk(start_path):
            for filename in filenames:
                print(f"Processing file: {filename}")
                file_path = os.path.join(dirpath, filename)

                results, batch_lens = esmmsa1b(file_path).get_embedding()

                row_attention = results["row_attentions"].squeeze(0)  
                row_attention = row_attention[:, :, 1:batch_lens[0, 0], 1:batch_lens[0, 0]]
                row_pred_all = row_attention.detach().cpu().numpy()
                row_pred = np.max(row_pred_all, axis=1)  

                col_attention = results["col_attentions"].squeeze(0)  
                target_size = col_attention.size(2)  
                weidu_size = col_attention.size(-1)  
                L = target_size - 1 
                col_attention = col_attention[:, :, 1:batch_lens[0, 0], :, :]

                x = col_attention.view(col_attention.size(0), col_attention.size(1), col_attention.size(2), -1)
                linear_layer = nn.Linear(weidu_size * weidu_size, L).to(device)
                x = x.to(device)
                pred_mapped = linear_layer(x)
                relu = nn.ReLU().to(device)
                col_attention = relu(pred_mapped)

                pred_all = pred_mapped.detach().cpu().numpy()
                pred = np.max(pred_all, axis=1)  

                output_folder = ''
                fused_array = np.concatenate((row_pred, pred), axis=0)
                os.makedirs(output_folder, exist_ok=True)

                base_filename = os.path.splitext(filename)[0]
                save_path = os.path.join(output_folder, f"{base_filename.strip()}.npy")
                np.save(save_path, fused_array)
                print(f"Saved row_col_attention to {save_path}")

                torch.cuda.empty_cache()


    start_path = ''
    traverse_directories(start_path)

