import torch
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sparse_autoencoders import JumpReLUSAE, MatryoshkaSAE, TopKSAE, VanillaSAE
import pandas as pd
import numpy as np
from rich import print

try:
    # Try to import cuML and check for a CUDA device
    import cuml # type: ignore
    print("imported cuml")
    #from cuml.cluster import KMeans as CumlKMeans
    #from cuml.cluster import AgglomerativeClustering as CumlAgglomerativeClustering
    #import numba.cuda

    #print("Error Before")

    if torch.cuda.is_available():
        KMeans = cuml.cluster.KMeans
        AgglomerativeClustering = cuml.cluster.AgglomerativeClustering
        print("using CUDA")
    else:
        raise ImportError("CUDA not available")
except ImportError:
    # Fallback to scikit-learn if cuML or CUDA is not available
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    print("using CPU")

def diffs(
    acts: torch.Tensor,
    n_pairs: int = None,
    n_dims : int = 100,
    return_encoder: bool = False,
    **kwargs
):
    """Our method, based on representing differences between samples.
    Parameters:
        - acts (torch.Tensor): 2 dimensional matrix of activations
    Returns:
        - _ (lambda) : Projection function into the extracted concept space
    
    """
    print('--- DIFFS ---')
    diffs = []
    if n_pairs is None: n_pairs = len(acts)
    for _ in tqdm(range(n_pairs), desc='Sampling pairs'):
        indices = torch.randperm(len(acts))[:2]
        sampled_rows = acts[indices]
        d = (acts[_ % len(acts)] - sampled_rows[1]).unsqueeze(0)
        diffs.append(d)
        del d
        del sampled_rows
    diffs = torch.vstack(diffs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    diffs = diffs.to(device)

    P = acts.to(device) @ diffs.T
    P = P.T
    mean = P.mean(dim=1, keepdim=True)
    std = P.std(dim=1, unbiased=False, keepdim=True) + 1e-20
    skew = (((P - mean) / std) ** 3).mean(dim=1)
    del P

    diffs *= torch.sign(skew).unsqueeze(1)
    skew_inv = torch.nan_to_num(1/skew.abs(), nan=0, posinf=0, neginf=0)
    projection_matrix = KMeans(n_clusters=n_dims).fit(diffs.cpu().detach().numpy(), sample_weight=skew_inv.abs().cpu().detach().numpy()).cluster_centers_.T


    if return_encoder: return projection_matrix
    return lambda x: x @ projection_matrix


def large_diffs(acts: torch.Tensor, **kwargs): return diffs(acts, None, n_dims=6144) 
def diffs50(acts: torch.Tensor, **kwargs): return diffs(acts, None, n_dims=50) 
def diffs125(acts: torch.Tensor, **kwargs): return diffs(acts, None, n_dims=125) 
def diffs250(acts: torch.Tensor, **kwargs): return diffs(acts, None, n_dims=250) 
def diffs768(acts: torch.Tensor, **kwargs): return diffs(acts, None, n_dims=768) 
def diffs1250(acts: torch.Tensor, **kwargs): return diffs(acts, None, n_dims=1250) 
def diffs2500(acts: torch.Tensor, **kwargs): return diffs(acts, None, n_dims=2500) 



def memsafediffs(
    acts: torch.Tensor,
    n_pairs: int = None,
    n_dims : int = 100,
    return_encoder: bool = False,
    **kwargs
):
    """Memory efficient version of our method. May be slower on some CPU configurations, but is less likely to cause OOM errors"""
    print('--- DIFFS ---')
    diffs = []
    if n_pairs is None: n_pairs = len(acts)
    for _ in tqdm(range(n_pairs), desc='Sampling pairs'):
        indices = torch.randperm(len(acts))[:2]
        sampled_rows = acts[indices]
        d = (acts[_ % len(acts)] - sampled_rows[1]).unsqueeze(0)
        diffs.append(d)
        del d
        del sampled_rows
    diffs = torch.vstack(diffs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    diffs = diffs.to(device)

    skew = torch.zeros((len(diffs),)).to(device)
    for i in tqdm(range(len(skew)), desc='Computing Skewness'):
        mean = diffs[i].mean()
        std = diffs[i].std(unbiased=False) + 1e-20
        skew[i] = (((diffs[i] - mean) / std)**3).mean()

    diffs *= torch.sign(skew).unsqueeze(1)
    skew_inv = torch.nan_to_num(1/skew.abs(), nan=0, posinf=0, neginf=0)
    projection_matrix = KMeans(n_clusters=n_dims).fit(diffs.cpu().detach().numpy(), sample_weight=skew_inv.abs().cpu().detach().numpy()).cluster_centers_.T


    if return_encoder: return projection_matrix
    return lambda x: x @ projection_matrix

def large_diffs_mem(acts: torch.Tensor, **kwargs): return memsafediffs(acts, None, n_dims=6144) 


def topksae(acts: torch.Tensor, **kwargs):
    """.
    Parameters:
        - acts (torch.Tensor): 2 dimensional matrix of activations
    Returns:
        - _ (lambda) : Projection function into the extracted concept space
    
    """
    print('--- SAE ---')
    acts = acts[torch.randperm(acts.shape[0])]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sae = TopKSAE(
        d_vit=acts.shape[1],
        expansion_factor=8,
        top_k=32
    ).to(device)

    LR = 1e-5

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    #BS = 32
    BS = 1
    for i, row in tqdm(enumerate(
        torch.chunk(acts.to(device),
        chunks=len(acts)//BS)), total=len(acts)//BS):
        loss = sae.loss(row)
        loss.backward()
        optimizer.step()
    sae = sae.to('cpu')

    return lambda x:  (x.to('cpu') - sae.b_dec) @ sae.W_enc

def vanillasae(acts: torch.Tensor, **kwargs):
    """
    Parameters:
        - acts (torch.Tensor): 2 dimensional matrix of activations
    Returns:
        - _ (lambda) : Projection function into the extracted concept space
    
    """
    print('--- SAE ---')
    acts = acts[torch.randperm(acts.shape[0])]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sae = VanillaSAE(
        d_vit=acts.shape[1],
        expansion_factor=8,
        l1_lambda=1e-8
    ).to(device)

    LR = 1e-5

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    #BS = 32
    BS = 1
    for i, row in tqdm(enumerate(
        torch.chunk(acts.to(device),
        chunks=len(acts)//BS)), total=len(acts)//BS):
        #sae_out = sae(row)
        loss = sae.loss(row)
        #if i % 100 == 0: print(loss.item())
        loss.backward()
        optimizer.step()
    sae = sae.to('cpu')

    return lambda x:  (x.to('cpu') - sae.b_dec) @ sae.W_enc


def matrysae(acts: torch.Tensor, **kwargs):
    """
    Parameters:
        - acts (torch.Tensor): 2 dimensional matrix of activations
    Returns:
        - _ (lambda) : Projection function into the extracted concept space
    
    """
    print('--- SAE ---')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    acts = acts.to(device)
    acts = acts[torch.randperm(acts.shape[0])]

    BS = 1
    sae = MatryoshkaSAE(
        d_vit=acts.shape[1],
        expansion_factor=8,
        n_steps=len(acts)//BS
    ).to(device)
    sae.train()

    
    for i, row in tqdm(enumerate(
        torch.chunk(acts.to(device),
        chunks=len(acts)//BS)), total=len(acts)//BS):
        sae.step(row)

        del row
    sae = sae.to(device)
    sae = sae.to('cpu')

    return lambda x : sae.get_acts(x)# lambda x:  (x.to('cpu') - sae.b_dec) @ sae.W_enc

def topksaediffs(acts: torch.Tensor, **kwargs):
    """Used only for ablation, learning SAEs on differences in activations
    Parameters:
        - acts (torch.Tensor): 2 dimensional matrix of activations
    Returns:
        - _ (lambda) : Projection function into the extracted concept space
    
    """
    print('--- DIFFS ---')
    diffs = []
    n_pairs = len(acts)
    for _ in tqdm(range(n_pairs), desc='Sampling pairs'):
        indices = torch.randperm(len(acts))[:2]
        sampled_rows = acts[indices]
        d = (acts[_ % len(acts)] - sampled_rows[1]).unsqueeze(0)
        diffs.append(d)
        del d
        del sampled_rows
    diffs = torch.vstack(diffs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffs = diffs.to(device)
    return topksae(acts=diffs, LR=5e-5)


def unweighteddiffs(
    acts: torch.Tensor,
    n_pairs: int = None,
    n_dims : int = 6144,
    return_encoder: bool = False,
    **kwargs
):
    """Used only for ablation
    Parameters:
        - acts (torch.Tensor): 2 dimensional matrix of activations
    Returns:
        - _ (lambda) : Projection function into the extracted concept space
    
    """
    print('--- UNWEIGHTED DIFFS ---')
    diffs = []
    if n_pairs is None: n_pairs = len(acts)
    for _ in tqdm(range(n_pairs), desc='Sampling pairs'):
        indices = torch.randperm(len(acts))[:2]
        sampled_rows = acts[indices]
        d = (acts[_ % len(acts)] - sampled_rows[1]).unsqueeze(0)
        diffs.append(d)
        del d
        del sampled_rows
    diffs = torch.vstack(diffs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    diffs = diffs.to(device)

    P = acts.to(device) @ diffs.T
    P = P.T
    mean = P.mean(dim=1, keepdim=True)
    std = P.std(dim=1, unbiased=False, keepdim=True) + 1e-20
    skew = (((P - mean) / std) ** 3).mean(dim=1)
    del P

    diffs *= torch.sign(skew).unsqueeze(1)
    projection_matrix = KMeans(n_clusters=n_dims).fit(diffs.cpu().detach().numpy()).cluster_centers_.T


    if return_encoder: return projection_matrix
    return lambda x: x @ projection_matrix


def unweighted_memsafediffs(
    acts: torch.Tensor,
    n_pairs: int = None,
    n_dims : int = 100,
    return_encoder: bool = False,
    **kwargs
):
    print('--- DIFFS ---')
    diffs = []
    if n_pairs is None: n_pairs = len(acts)
    for _ in tqdm(range(n_pairs), desc='Sampling pairs'):
        indices = torch.randperm(len(acts))[:2]
        sampled_rows = acts[indices]
        d = (acts[_ % len(acts)] - sampled_rows[1]).unsqueeze(0)
        diffs.append(d)
        del d
        del sampled_rows
    diffs = torch.vstack(diffs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    diffs = diffs.to(device)

    skew = torch.zeros((len(diffs),)).to(device)
    for i in tqdm(range(len(skew)), desc='Computing Skewness'):
        mean = diffs[i].mean()
        std = diffs[i].std(unbiased=False) + 1e-20
        skew[i] = (((diffs[i] - mean) / std)**3).mean()

    diffs *= torch.sign(skew).unsqueeze(1)
    projection_matrix = KMeans(n_clusters=n_dims).fit(diffs.cpu().detach().numpy()).cluster_centers_.T


    if return_encoder: return projection_matrix
    return lambda x: x @ projection_matrix

def kmeansacts(
    acts: torch.Tensor,
    n_pairs: int = None,
    n_dims : int = 6144,
    return_encoder: bool = False,
    **kwargs
):
    """Used only for ablation, direclty performing KMeans on activations
    Parameters:
        - acts (torch.Tensor): 2 dimensional matrix of activations
    Returns:
        - _ (lambda) : Projection function into the extracted concept space
    
    """
    print('--- DIRECT KMEANS ---')

    projection_matrix = KMeans(n_clusters=n_dims).fit(acts.cpu().detach().numpy()).cluster_centers_.T


    if return_encoder: return projection_matrix
    return lambda x: x @ projection_matrix


if __name__ == '__main__':
    
    train = torch.load("activations/mini-imagenet_ClipVisionB_-1/train.pt")
    test  = torch.load("activations/mini-imagenet_ClipVisionB_-1/test.pt")

    proj = diffs(train, n_dims=50, return_encoder=True)

    print(torch.nn.MSELoss()(
        train @ proj @ torch.pinverse(torch.tensor(proj)),
        train)
    )
    
    print(torch.nn.MSELoss()(
        test @ proj @ torch.pinverse(torch.tensor(proj)),
        test)
    )