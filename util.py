import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(eps={self.eps})'


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


def load_checkpoint(args, model, checkpoint):
    state_dict = torch.load(checkpoint)
    if 'state_dict' in state_dict:
        state_dict = torch.load(checkpoint)['state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = torch.load(checkpoint)['model_state_dict']
    elif 'teacher' in state_dict:
        state_dict = torch.load(checkpoint)['teacher']

    if args.model == 'desc_1st':
        model.load_state_dict(state_dict)
    else:
        # model.module.load_state_dict(state_dict)
        model.module.load_state_dict(state_dict, strict=False)

    return model


def negative_embedding_subtraction(
        embedding: np.ndarray,
        negative_embeddings: np.ndarray,
        faiss_index: faiss.IndexFlatIP,
        num_iter: int = 3,
        k: int = 10,
        beta: float = 0.35,
) -> np.ndarray:
    """
    Post-process function to obtain more discriminative image descriptor.

    Parameters
    ----------
    embedding : np.ndarray of shape (n, d)
        Embedding to be subtracted.
    negative_embeddings : np.ndarray of shape (m, d)
        Negative embeddings to be subtracted.
    faiss_index : faiss.IndexFlatIP
        Index to be used for nearest neighbor search.
    num_iter : int, optional
        Number of iterations. The default is 3.
    k : int, optional
        Number of nearest neighbors to be used for each iteration. The default is 10.
    beta : float, optional
        Parameter for the weighting of the negative embeddings. The default is 0.35.

    Returns
    -------
    np.ndarray of shape (n, d)
        Subtracted embedding.
    """
    for _ in range(num_iter):
        _, topk_indexes = faiss_index.search(embedding, k=k)  # search for hard negatives
        topk_negative_embeddings = negative_embeddings[topk_indexes]

        embedding -= (topk_negative_embeddings.mean(axis=1) * beta)  # subtract by hard negative embeddings
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)  # L2-normalize

    return embedding.astype('float32')
