# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from collections import OrderedDict

from functools import partial
from typing import Any, Callable

from Adapter.models.approx_module import LinAtten


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
    """Transformer encoder block with LinAtten."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = LinAtten(dim=hidden_dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)

        #self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)

        #MPC-friendly MLP
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, act_layer=nn.ReLU)


    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        #x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


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
                 num_blk=1,):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.dropout = nn.Dropout(dropout)
        self.scale = float(adapter_scaler)

        self.lora_A = nn.Linear(in_features=self.n_embd, out_features=self.down_size)

        self.lora_B = nn.Linear(in_features=self.down_size, out_features=self.n_embd)
        blks: OrderedDict[str, nn.Module] = OrderedDict()

        for i in range(num_blk):
            blks[f"blk_{i}"] = EncoderBlock(num_heads=num_heads, hidden_dim=self.down_size, mlp_dim=mlp_dim, dropout=dropout, attention_dropout=attention_dropout, norm_layer=norm_layer)
        
        self.blks = nn.Sequential(blks) 

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            nn.init.zeros_(self.lora_A.bias)
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x):
        x = self.lora_A(x)
        x = self.dropout(x)
        x = self.blks(x)
        x = self.lora_B(x)
        x = x * self.scale
        return x
