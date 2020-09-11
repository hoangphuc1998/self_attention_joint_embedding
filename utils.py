import os
import numpy as np
import torch

def seed_everything(seed=100):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def normalize(X):
    mean = torch.mean(X, dim = 1, keepdim=True)
    std = torch.std(X, dim = 1, keepdim=True)
    return (X - mean) / std

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def batch_l2norm(X):
    """L2 normalization for batch of image regions (B, N, D)
    """
    norm = torch.pow(X, 2).sum(dim=2, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def get_top_k_eval(texts, images, k):
    dists = 1 - (texts.mm(images.t())/torch.norm(texts, p=2, dim=1, keepdim=True))/(torch.norm(images, p=2, dim=1, keepdim=True).t())
    _, indices = torch.topk(dists, k,  largest = False)
    return indices

def cosine_sim(query, retrio):
    """Cosine similarity between all the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)
    return query.mm(retrio.t())