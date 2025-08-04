import os
import argparse
import subprocess
import numpy as np
import numba
import math
from Bio import SeqIO
from numpy.typing import NDArray
from typing import Tuple

ceo_clustering_bin = 'ceo-clustering'

alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
int_type = np.uint16
float_type = np.float64

NPZ_onehot = "onehot"
NPZ_to_alignment_column = "to_alignment_column"
NPZ_N_ia = "N_ia"
NPZ_A = "A"
NPZ_Stilde_km_cache = "Stilde_km_cache"
NPZ_dS0 = "dS0"
NPZ_dS_km = "dS_km"
NPZ_dQ_km = "dQ_km"
NPZ_merge_km = "merge_km"
NPZ_cluster_assignments = "cluster_assignments"
NPZ_N_k = "N_k"
NPZ_N_ia = "N_ia"
NPZ_f_ia = "f_ia"
NPZ_N_kia = "N_kia"
NPZ_f_kia = "f_kia"


def parse_A(A_str: str) -> NDArray:
    A_begin, A_end, A_step = map(float, A_str.split(','))
    lenA = math.floor((A_end - A_begin) / A_step) + 1
    return np.arange(A_begin, A_begin + lenA * A_step, A_step)


def extract_msa_cols(msa_file: str, _alphabet: NDArray, max_col_gap: float) -> Tuple[NDArray, NDArray]:
    aln = list(SeqIO.parse(msa_file, 'fasta'))
    a = np.array(aln, dtype='S1').view(np.int8)
    to_alignment_column = np.nonzero(
        (np.count_nonzero(a == 45, axis=0) <= a.shape[0] * max_col_gap) &
        np.isin(a[0], _alphabet)
    )[0].astype(int_type)
    return a[:, to_alignment_column], to_alignment_column


def assign_cluster(onehot: NDArray, merge_km: NDArray, full_dS0: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    def _select_clustering_steps(full_dS0: NDArray, N: int):
        dS0 = full_dS0.reshape(-1, N - 1)
        begins = np.arange(0, len(full_dS0), N - 1, dtype=int)
        m = dS0 == full_dS0.min()
        p = np.any(m, axis=1)
        num_clusters = 1 + np.argmax(m[p, ::-1], axis=1)
        begins = begins[p]
        ends = N - 1 - num_clusters + begins
        i = np.argmin(full_dS0[ends])
        return begins[i], ends[i]

    @numba.jit(nopython=True, nogil=True, fastmath=True, error_model='numpy')
    def numba_assign_clusters(N, merge_km, begin, end):
        cluster_assignments = np.arange(N, dtype=int_type)
        cluster_sizes = np.ones(N, dtype=int_type)
        new_cluster = N - 2
        for i in range(begin, end + 1):
            k = merge_km[i, 0]
            m = merge_km[i, 1]
            
            new_cluster_size = cluster_sizes[k] + cluster_sizes[m]

            for j in range(k + 1, m):
                cluster_sizes[j - 1] = cluster_sizes[j]
            for j in range(m + 1, new_cluster + 2):
                cluster_sizes[j - 2] = cluster_sizes[j]
            cluster_sizes[new_cluster] = new_cluster_size

            for j in range(N):
                if (cluster_assignments[j] == k) or (cluster_assignments[j] == m):
                    cluster_assignments[j] = new_cluster
                elif (cluster_assignments[j] > k) and (cluster_assignments[j] < m):
                    cluster_assignments[j] -= 1
                elif cluster_assignments[j] > m:
                    cluster_assignments[j] -= 2
            
            new_cluster -= 1
        cluster_sizes = cluster_sizes[0 : new_cluster + 2]
        num_clusters = len(cluster_sizes)

        # relabel clusters by descending size
        # ensure lead sequence is in cluster 0
        cluster_sizes[cluster_assignments[0]] += N
        sorted_indices = cluster_sizes.argsort()
        cluster_sizes[cluster_assignments[0]] -= N
        relabel_mapping = np.empty(num_clusters, dtype=int_type)
        for i in range(num_clusters):
            relabel_mapping[sorted_indices[i]] = num_clusters - 1 - i
        
        for i in range(N):
            cluster_assignments[i] = relabel_mapping[cluster_assignments[i]]

        N_k = np.empty(num_clusters, dtype=int_type)
        for i in range(num_clusters):
            N_k[relabel_mapping[i]] = cluster_sizes[i]
        
        return cluster_assignments, N_k

    @numba.jit(nopython=True, nogil=True, error_model='numpy')
    def calc_N_kia(onehot, cluster_assignments, M):
        N, L, D = onehot.shape
        N_kia = np.zeros((M, L, D), dtype=float_type)
        for n in range(N):
            N_kia[cluster_assignments[n]] += onehot[n, :, :]
        return N_kia

    onehot = np.swapaxes(onehot, 1, 2)
    N, L, D = onehot.shape
    N_ia = onehot.sum(axis=0)
    f_ia = N_ia / N

    begin, end = _select_clustering_steps(full_dS0, N)
    cluster_assignments, N_k = numba_assign_clusters(N, merge_km, begin, end)
    num_clusters = len(N_k)
    N_kia = calc_N_kia(onehot, cluster_assignments, num_clusters)
    f_kia = N_kia / N_k[:, np.newaxis, np.newaxis]
    return cluster_assignments, N_k, N_ia, f_ia, N_kia, f_kia


def ceo(msa_file: str, max_col_gap = 0.3, A_str = '0.5,0.975,0.025', num_threads = 4):
    ceo_dir = os.path.dirname(msa_file)
    clustering_npz_file = os.path.splitext(msa_file)[0] + '_ceo_clustering.npz'

    _alphabet = np.fromiter(alphabet, dtype='S1').view(np.int8)
    a, to_alignment_column = extract_msa_cols(msa_file, _alphabet, max_col_gap)

    in_npz_file = os.path.join(ceo_dir, '_clustering_in.npz')
    onehot = (a[:, np.newaxis, :] == _alphabet[np.newaxis, :, np.newaxis]).astype(int_type)
    in_npz = {
        NPZ_A: parse_A(A_str), 
        NPZ_onehot: onehot, 
        NPZ_to_alignment_column: to_alignment_column,
    }
    np.savez(in_npz_file, **in_npz)

    out_npz_file = os.path.join(ceo_dir, '_clustering_out.npz')
    subprocess.run([ceo_clustering_bin, '--threads', str(num_threads), '--in-npz', in_npz_file, '--out-npz', out_npz_file], check=True)
    out_npz = dict(np.load(out_npz_file))
    merge_km = out_npz[NPZ_merge_km]
    full_dS0 = out_npz[NPZ_dS0]
    
    cluster_assignments, N_k, N_ia, f_ia, N_kia, f_kia = assign_cluster(onehot, merge_km, full_dS0)
    clustering_npz = {
        NPZ_cluster_assignments: cluster_assignments, 
        NPZ_N_k: N_k, 
        # NPZ_N_ia: N_ia, # already in out_npz 
        NPZ_f_ia: f_ia, 
        NPZ_N_kia: N_kia, 
        NPZ_f_kia: f_kia
    }
    
    np.savez_compressed(clustering_npz_file, allow_pickle=True, **in_npz, **out_npz, **clustering_npz)
    os.remove(in_npz_file)
    os.remove(out_npz_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CEO")
    parser.add_argument('msa_file', type=str)
    parser.add_argument('--max-col-gap', type=float, default=0.3)
    parser.add_argument('-A', type=str, default='0.5,0.975,0.025')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    ceo(
        msa_file = args.msa_file, 
        max_col_gap = args.max_col_gap, 
        A_str = args.A, 
        num_threads = args.threads
    )

