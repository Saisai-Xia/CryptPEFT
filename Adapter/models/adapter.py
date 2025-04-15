# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F
from timm.layers import use_fused_attn
from timm.layers.helpers import to_2tuple
from collections import OrderedDict

from functools import partial
from typing import Any, Callable

from Adapter.models.approx_module import ReLU_maker, Attention_approx



from torchvision.ops.misc import  MLP
class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2
    

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, act_layer: nn.Module = nn.GELU):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=act_layer, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        approx = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.approx = approx
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        #self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        if approx != None:
            print("Using Approximation" + str(approx))
            self.self_attention = Attention_approx(dim=hidden_dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout, approx_p = approx)
        else:
            self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        if approx != None:
            self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, act_layer=nn.ReLU)
        else:
            self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        #self.mlp = Mlp(hidden_dim, mlp_dim, hidden_dim, act_layer=ReLU_maker())

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        if self.approx != None:
            x = self.self_attention(x)
        else:
            x, _ = self.self_attention(x, x, x, need_weights=False)
        #x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        #x = input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

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

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#adaptformer 
class adaptformer(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_scaler="1.0",):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        if adapter_scaler == "learnable_scaler":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scaler)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):


        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        output = up

        return output

#LoRA for vit
class lora(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.dropout = dropout
        self.scale = float(1.0/bottleneck)

        self.lora_A = nn.Linear(in_features=self.n_embd, out_features=self.down_size)
         
        self.lora_B = nn.Linear(in_features=self.down_size, out_features=self.n_embd)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            nn.init.zeros_(self.lora_A.bias)
            nn.init.zeros_(self.lora_B.bias)


    def forward(self, x):
        x = self.lora_A(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lora_B(x) * self.scale
        return x

class CryptPEFT_adapter(nn.Module):
    def __init__(self,
                 num_heads: int,
                 attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_scaler="1.0",
                 mlp_dim=None,
                 num_blk=1,
                 approx=None):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.dropout = dropout
        self.scale = float(adapter_scaler)

        self.lora_A = nn.Linear(in_features=self.n_embd, out_features=self.down_size)

        self.lora_B = nn.Linear(in_features=self.down_size, out_features=self.n_embd)
        blks: OrderedDict[str, nn.Module] = OrderedDict()

        for i in range(num_blk):
            blks[f"blk_{i}"] = EncoderBlock(num_heads=num_heads, hidden_dim=self.down_size, mlp_dim=mlp_dim, dropout=dropout, attention_dropout=attention_dropout, norm_layer=norm_layer, approx=approx)
        
        self.blks = nn.Sequential(blks) 

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            nn.init.zeros_(self.lora_A.bias)
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x):
        x = self.lora_A(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.blks(x)
        x = self.lora_B(x)
        x = x * self.scale
        return x
