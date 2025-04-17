import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torchvision

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

from .adapter import CryptPEFT_adapter, adaptformer, lora

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

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
        config = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.config = config

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        if self.config.adapt_on and not self.config.fulltune and self.config.adapter_type == "adaptformer":
            self.adaptmlp = adaptformer(dropout=0.1, d_model=hidden_dim, bottleneck=64, adapter_scaler=0.1)

        if self.config.adapt_on and not self.config.fulltune and self.config.adapter_type == "lora":
            self.adaptmlp = lora(dropout=0.1, d_model=hidden_dim, bottleneck=8)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        if self.config.adapt_on and not self.config.fulltune and self.config.adapter_type in ["adaptformer", "lora"]:
            adapt = self.adaptmlp(x)
        else:
            adapt = 0

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y + adapt
    

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        config = None
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.first_layer=config.first_layer
        self.layer_id = config.layer_id
        self.config = config
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        adapters: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                config = config,
            )
            if self.config.adapt_on and not self.config.fulltune and self.config.adapter_type == "CryptPEFT":
                if i >=self.first_layer:
                    if self.config.adapter_arch == "CryptPEFT":
                        adapters[f"adapter_layer_{i}"] = CryptPEFT_adapter(num_heads=self.config.num_head, attention_dropout=0.0, norm_layer=norm_layer,d_model=hidden_dim,
                                                                    bottleneck=self.config.bottleneck, dropout=0.1, adapter_scaler=self.config.adapter_scaler, mlp_dim=self.config.bottleneck, num_blk=self.config.num_repeat_blk)
                    elif self.config.adapter_arch == "adaptformer":
                        adapters[f"adapter_layer_{i}"] = adaptformer(dropout=0.1, d_model=hidden_dim, bottleneck=64, adapter_scaler=0.1)
                    elif self.config.adapter_arch == "lora":
                        adapters[f"adapter_layer_{i}"] = lora(dropout=0.1, d_model=hidden_dim, bottleneck=8)
                else:
                    adapters[f"adapter_layer_{i}"] = nn.Identity()
            #baseline
            if self.config.adapt_on and not self.config.fulltune and self.config.adapter_type == "single_adapter":
                if i == self.layer_id:
                    if self.config.adapter_arch == "CryptPEFT":
                        print("hellp")
                        adapters[f"adapter_layer_{i}"] = CryptPEFT_adapter(num_heads=self.config.num_head, attention_dropout=0.0, norm_layer=norm_layer,d_model=hidden_dim,
                                                                    bottleneck=self.config.bottleneck, dropout=0.1, adapter_scaler=self.config.adapter_scaler, mlp_dim=self.config.bottleneck, num_blk=self.config.num_repeat_blk)
                else:
                    adapters[f"adapter_layer_{i}"] = nn.Identity()
        
        self.layers = nn.Sequential(layers)

        if self.config.adapt_on and not self.config.fulltune and self.config.adapter_type in ["CryptPEFT", "single_adapter"]:
            self.adapters = nn.Sequential(adapters)

        self.ln = norm_layer(hidden_dim)
        

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)

        if self.config.adapt_on and not self.config.fulltune and self.config.adapter_type == "CryptPEFT":
            adapter_output = torch.tensor(0)
            i = 0
            for layer, adapter in zip(self.layers, self.adapters):
                if i < self.first_layer:
                    input = layer(input)
                else:  
                    adapter_input = adapter_output + input
                    input = layer(input)
                    adapter_output = adapter(adapter_input) + adapter_input
                i += 1
                
            input = input + adapter_output
            #input = adapter_output
        else:
            input = self.layers(input)

        out = self.ln(input)
        return out
    

class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        config = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.adapter = []
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            config=config,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
    
    def freeze_backbone_only(self):
        for name, p in self.named_parameters():
            if(name in self.adapter):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # #head
        for _, p in self.heads.named_parameters():
            p.requires_grad = True
        # final ln
        for _, p in self.encoder.ln.named_parameters():
            p.requires_grad = True

    def freeze_layers(self, finetune_layer_num):
        #freeze all layers except >= 12-finetune_layer_num
        finetune_layer_name = [f"encoder_layer_{i}" for i in range(12-finetune_layer_num,12)]
        for name, p in self.named_parameters():
            for layer in finetune_layer_name:
                if(layer in name):
                    p.requires_grad = True
                    break
                else:
                    p.requires_grad = False
        # #head
        for _, p in self.heads.named_parameters():
            p.requires_grad = True
        # final ln
        for _, p in self.encoder.ln.named_parameters():
            p.requires_grad = True

def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        msg = model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)
        model.adapter = msg.missing_keys

    return model
 
def Vit_B_16(pretrained = True, num_classes = 100, **kwargs: Any):

    if(pretrained):
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torchvision.models.ViT_B_16_Weights.verify(weights)

    # param
    hidden_dim = 768
    patch_size = 16
    num_layers = 12
    mlp_dim = 3072
    num_heads = 12
    
    model = _vision_transformer(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, weights, progress=True,**kwargs,)

    # resize header
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    representation_size = kwargs.get('representation_size', None)
    if representation_size is None:
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
    else:
        heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(representation_size, num_classes)
    model.heads = nn.Sequential(heads_layers)

    return model



