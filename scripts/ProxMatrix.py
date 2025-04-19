import numpy as np
from numba import njit, prange


@njit(parallel=True)
def get_upper_prox_matrix(leaves):
    n_examples = leaves.shape[0]
    prox_matrix = np.zeros((n_examples, n_examples), dtype=np.uint16)
    for i in prange(n_examples):
        for j in range(i + 1, n_examples):
            prox_matrix[i, j] = np.sum(leaves[i] == leaves[j])
    return prox_matrix


def get_dist_matrix(upper_prox_matrix, n_tree: int):
    n_examples = upper_prox_matrix.shape[0]
    prox_matrix = (upper_prox_matrix + upper_prox_matrix.T).astype(np.float32)
    prox_matrix /= n_tree
    prox_matrix[range(n_examples), range(n_examples)] = 1
    dist_matrix = 1 - prox_matrix
    return dist_matrix
