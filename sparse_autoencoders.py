import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod, ABC
import math
import warnings
warnings.filterwarnings("ignore", message="^.*instability in Pearson correlationcoefficient.*$")

def safe_mse(x_hat, x, norm=False):
    """Safe MSE, in case of very large values of x_hat or x"""

    upper = x.abs().max()
    x = x / upper
    x_hat = x_hat / upper

    mse = (x_hat - x) ** 2
    # (sam): I am now realizing that we normalize by the L2 norm of x.
    if norm:
        mse /= torch.linalg.norm(x, axis=-1, keepdim=True) + 1e-12
        return mse * upper

    return mse * upper * upper


    @torch.no_grad()
    def get_features(self, x):
        """Get SAE features for a batch of activations"""
        h_pre = self.enc(x)
        f_x = torch.nn.functional.relu(h_pre)
        return f_x
    

class SparseAutoencoder(nn.Module, ABC):
    """Base class for Advanced Sparse Autoencoder"""
    def __init__(self, d_vit = 768, expansion_factor = 16, **kwargs):
        """Initialize W_enc = W_dec.T, according to 
        "Scaling and evaluating sparse autoencoders" (Gao et al.)"""
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        d_sae = d_vit * expansion_factor
        
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_vit, d_sae))
        )

        self.b_dec = nn.Parameter(torch.zeros(d_vit))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_vit))
        )

        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.num_batches_not_active = torch.zeros(d_sae).to(self.device)

    @abstractmethod
    def forward(self, x):
        """
        Compute reconstruction of input x
        Parameters:
            - x (tensor): batch of vit activations
        Return:
            - loss (tensor) : reconstruction loss
            - metrics (dict) : metrics to log to a tensorboard or equivalent
        """
        pass

    @abstractmethod
    def get_features(self, x):
        """Get SAE features for a batch of activations"""
        pass

    def update_inactive_features(self, acts):
        acts_sum = acts.sum(0).sum(0) # Sum per feature
        self.num_batches_not_active += (acts_sum == 0).float()
        self.num_batches_not_active[acts_sum > 0] = 0

    def make_wdec_grad_unit(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @abstractmethod
    def loss(self, x):
        pass

class TopKSAE(SparseAutoencoder):
    """SAE constraining directly L0 over features, per patch"""
    def __init__(self, d_vit=768, expansion_factor=16, top_k=32, 
                aux_penalty=1/32, aux_top_k=512,n_batches_to_dead=5,**kwargs):
        super().__init__(d_vit, expansion_factor, **kwargs)
        self.top_k = top_k
        self.aux_penalty = aux_penalty
        self.aux_top_k = aux_top_k
        self.n_batches_to_dead = n_batches_to_dead

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = torch.topk(acts, self.top_k, dim=-1)
        features = torch.zeros_like(acts).scatter(
            -1, features.indices, features.values
        )
        self.update_inactive_features(features)

        x_hat = features @ self.W_dec + self.b_dec
        return x_hat
    
    def get_features(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = torch.topk(acts, self.top_k, dim=-1)
        features = torch.zeros_like(acts).scatter(
            -1, features.indices, features.values
        )
        return features
    def get_cls_features(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)

        n_patches = acts.shape[-2]
        if n_patches-1 != math.isqrt(n_patches) ** 2: # n_patches is perfect square <=> no CLS token : Avg
            acts[...,0,:] = acts.mean(dim=-2) # Average across patches

        #features = torch.topk(acts, self.top_k, dim=-1)
        #features = torch.zeros_like(acts).scatter(
        #    -1, features.indices, features.values
        #)
        features = acts
        return features[...,0,:]
    
    def loss(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = torch.topk(acts, self.top_k, dim=-1)
        features = torch.zeros_like(acts).scatter(
            -1, features.indices, features.values
        )
        self.update_inactive_features(features)

        x_hat = features @ self.W_dec + self.b_dec
        mse_loss = nn.MSELoss().to(self.device)(x, x_hat)
        aux_loss = self.auxiliary_loss(x, x_hat, acts)

        return mse_loss #+ aux_loss

    def auxiliary_loss(self, x, x_hat, acts):
        """Reconstruction MSE using `self.aux_top_k` dead latents"""
        dead_features = self.num_batches_not_active >= self.n_batches_to_dead
        if dead_features.sum() > 0:
            residual = x.float() - x_hat.float()
            acts_topk_aux = torch.topk(
                acts[..., dead_features],
                min(self.aux_top_k, dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[..., dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.aux_penalty * (
                    x_reconstruct_aux.float() - residual.float()
                ).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
        


class VanillaSAE(SparseAutoencoder):
    def __init__(self, d_vit=768, expansion_factor=8, l1_lambda=1e-6, **kwargs):
        super().__init__(d_vit, expansion_factor, **kwargs)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        x_hat = acts @ self.W_dec + self.b_dec
        return x_hat
    def get_features(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        return acts
    
    def get_cls_features(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        return acts[...,0,:]

    def loss(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        x_hat = acts @ self.W_dec + self.b_dec

        l2_loss = (x_hat.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = l1_norm * self.l1_lambda

        #print(f'L2 : {l2_loss} | L1 : {l1_loss} | L0 : {torch.count_nonzero(acts[...,0])}')

        return l2_loss + l1_loss
    

class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
class RectangleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        print('Threshold', threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        print("AUFHIUAZHFIUZH")
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)
    

class JumpReLUSAE(SparseAutoencoder):
    def __init__(self, d_vit=768, expansion_factor=16, bandwidth=1e-3, lambda_l0=1e-8,**kwargs):
        super().__init__(d_vit, expansion_factor, **kwargs)
        self.jumprelu = JumpReLU(
            feature_size=expansion_factor*d_vit,
            bandwidth=bandwidth,
            #device=cfg["device"]
        )
        self.bandwidth = bandwidth
        self.lambda_l0 = lambda_l0
    def forward(self, x):
        mean = x.mean()
        std = x.std()
        x_std = (x - mean) / std
        x_cent = x_std - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = self.jumprelu(acts)
        x_hat = features @ self.W_dec + self.b_dec
        x_hat = x_hat * std + mean
        return x_hat
    
    def get_features(self, x):
        mean = x.mean()
        std = x.std()
        x_std = (x - mean) / std
        x_cent = x_std - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = self.jumprelu(acts)
        return features

    def get_cls_features(self, x):
        mean = x.mean()
        std = x.std()
        x_std = (x - mean) / std
        x_cent = x_std - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = self.jumprelu(acts)
        return features[...,0,:]
    
    def loss(self, x):
        mean = x.mean()
        std = x.std()
        x_std = (x - mean) / std
        x_cent = x_std - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = self.jumprelu(acts)
        x_hat = features @ self.W_dec + self.b_dec
        x_hat = x_hat * std + mean
        l2_loss = (x_hat.float() - x.float()).pow(2).mean()
        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.bandwidth).sum(dim=-1).mean()
        l0_loss = l0 * self.lambda_l0
        #print(l2_loss, l0, l0_loss, self.jumprelu.log_threshold)
        return l2_loss + l0_loss



    
    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive


import matryoshka
MatryoshkaSAE = matryoshka.MatryoshkaSAE