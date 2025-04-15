'''edit from https://github.com/snu-ccl/approxCNN'''
import torch
import torch.nn as nn
import numpy as np
import math
import itertools

class rangeException(Exception):
    def __init__(self, type, val):
        self.type = type
        self.val = val

    def show(self):
        if self.type == 'relu':
            print("STOP! There is an input value", self.val.item(), "for the approximate ReLU function.")
        elif self.type == 'max':
            print("STOP! There is an input value", self.val.item(), "for the approximate max-pooling function.")

class ReLU_approx_module(nn.Module):
    def __init__(self):
        super(ReLU_approx_module, self).__init__()
        self.relu_dict = {'alpha': 6, 'B': 100.0, 'type': 'proposed'}

    def forward(self, x):
        return ReLU_approx(x, self.relu_dict)

class ReLU_square_module(nn.Module):
    def __init__(self):
        super(ReLU_square_module, self).__init__()

    def forward(self, x):
        return x ** 2

class ReLU_AQ_module(nn.Module):
    def __init__(self):
        super(ReLU_AQ_module, self).__init__()

    def forward(self, x):
        return (x**2 / 8.0 + x / 2.0 + 1 / 4.0)


def ReLU_maker(relu_dict = {'alpha': 6, 'B': 100.0, 'type': 'proposed'}):
    if relu_dict['type'] == 'pure':
        return nn.ReLU(inplace=True)
    elif relu_dict['type'] == 'proposed':
        return ReLU_approx_module
    elif relu_dict['type'] == 'square':
        return ReLU_square_module()
    elif relu_dict['type'] == 'relu_aq':
        return ReLU_AQ_module()
    return 0

def poly_eval(x, coeff):
    coeff = torch.tensor(coeff).cuda()


    if len(x.size()) == 2:
        return torch.sum(x[:, :, None] ** torch.arange(coeff.size(0)).cuda()[None, None, :] * coeff, dim=-1)


    elif len(x.size()) == 3:
        return torch.sum(x[:, :, :, None] ** torch.arange(coeff.size(0)).cuda()[None, None, None, :] * coeff, dim=-1)

    elif len(x.size()) == 4:
        return torch.sum(x[:, :, :, :, None] ** torch.arange(coeff.size(0)).cuda()[None, None, None, None, :] * coeff, dim=-1)

    else:
        raise ValueError(f"Unsupported input dimension: {len(x.size())}. Only dimensions 2, 3, and 4 are supported.")



def sgn_approx(x, relu_dict):
    alpha = relu_dict['alpha']
    B = torch.tensor(relu_dict['B']).cuda().double()

    # Get degrees
    f = open('./approxCNN/degreeResult/deg_' + str(alpha) + '.txt')
    readed = f.readlines()
    comp_deg = [int(i) for i in readed]

    # Get coefficients
    f = open('./approxCNN/coeffResult/coeff_' + str(alpha) + '.txt')
    coeffs_all_str = f.readlines()
    coeffs_all = [torch.tensor(np.double(i), dtype=torch.double) for i in coeffs_all_str]
    i = 0

    if (torch.sum(torch.abs(x) > B) != 0.0):
        max_val = torch.max(torch.abs(x))
        raise rangeException('relu', max_val)

    x.double()
    x = torch.div(x, B)

    for deg in comp_deg:
        coeffs_part = coeffs_all[i:(i + deg + 1)]
        x = poly_eval(x, coeffs_part)
        torch.cuda.empty_cache()
        i += deg + 1

    return x.float()

def ReLU_approx(x, relu_dict):
    sgnx = sgn_approx(x, relu_dict)
    return x * (1.0 + sgnx) / 2.0



class PowerSoftmax(nn.Module):
    def __init__(self, dim=-1, eps=1e-2, c:float=1.0, p:int = 6):
        super(PowerSoftmax, self).__init__()
        self.dim = dim
        self.eps = eps
        self.c = c
        self.p = p

    def forward(self, x):
        L = x.size(self.dim)
        numerator = ((x/self.c) ** self.p)
        denominator = torch.sum((x/self.c) ** self.p, dim=self.dim, keepdim=True) + self.eps*L
        return numerator / denominator


class Attention_approx(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            approx_p: int = 6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.powersoftmax = PowerSoftmax(p = approx_p)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        #attn = attn.softmax(dim=-1)
        attn = self.powersoftmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
