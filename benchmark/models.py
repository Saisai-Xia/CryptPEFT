import math
import time

import torch
import torch.nn.functional as F

import crypten
import crypten.nn as cnn
import crypten.communicator as comm

class PowerSoftmax(cnn.Module):
    def __init__(self, dim=-1, eps=1e-2, c:float=1.0, p:int = 6):
        super(PowerSoftmax, self).__init__()
        print("approx",p)
        self.dim = dim
        self.eps = eps
        self.c = c
        self.p = torch.tensor(p).item()
        self.sum = cnn.Sum(dim=self.dim, keepdim=True)
        self.pow = cnn.Pow()

    def forward(self, x):
        L = x.size(self.dim)
        numerator = self.pow((x/self.c, self.p))
        denominator = self.sum(self.pow((x/self.c, self.p))) + self.eps*L
        return numerator / denominator

class MLP(cnn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.dim = args.dim
        self.mlp_hidden_dim = args.mlp_dim
        dropout = args.mlp_drop
        
        self.fc1 = cnn.Linear(self.dim, self.mlp_hidden_dim)
        self.act = cnn.ReLU()
        self.drop1 = cnn.Dropout(p = dropout)
        self.norm = cnn.BatchNorm2d(num_features=self.mlp_hidden_dim, eps=args.layer_norm_eps)
        self.fc2 = cnn.Linear(self.mlp_hidden_dim, self.dim)
        self.drop2 = cnn.Dropout(p = dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x_size = x.size()
        x = x.view(-1, self.mlp_hidden_dim)
        x = self.norm(x).view(x_size)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Attention(cnn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        assert args.dim % args.num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = args.num_heads
        self.head_dim = args.dim // args.num_heads
        self.scale = math.sqrt(self.head_dim)
        attn_drop = args.attn_drop
        proj_drop = args.proj_drop

        self.q = cnn.Linear(args.dim, args.dim)
        self.k = cnn.Linear(args.dim, args.dim)
        self.v = cnn.Linear(args.dim, args.dim)
        self.attn_drop = cnn.Dropout(p = attn_drop)
        self.proj = cnn.Linear(args.dim, args.dim)
        self.proj_drop = cnn.Dropout(p = proj_drop)

        self.softmax = PowerSoftmax(p=args.degree) if args.use_approx else cnn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        x_size = x.size()
        q_layer = self.transpose_for_scores(self.q(x))
        k_layer = self.transpose_for_scores(self.k(x))
        v_layer = self.transpose_for_scores(self.v(x))

        attention_scores = q_layer.matmul(k_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_drop(attention_probs)
        x = attention_probs.matmul(v_layer)
        x = x.transpose(1,2)
        x = x.reshape(x_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class EncoderBlock(cnn.Module):
    """Transformer encoder block."""
    def __init__(self, args):
        super(EncoderBlock, self).__init__()
        self.dim = args.dim
        self.ln_1 = cnn.BatchNorm2d(self.dim, eps=args.layer_norm_eps)
        self.self_attention = Attention(args)
        self.dropout = cnn.Dropout(args.encoder_drop)

        self.ln_2 = cnn.BatchNorm2d(self.dim, eps=args.layer_norm_eps)
        self.mlp = MLP(args)
        if args.use_PEFT == 'adaptformer':
            self.adapter = adaptformer(args)
        elif args.use_PEFT == 'lora':
            self.adapter = lora(args)
        
    
    def forward(self, x):
        assert len(x.size()) == 3, f"Expected (batch_size, seq_length, dim) got {x.size()}"
        input = x
        x_size = x.size()
        x = x.view(-1, self.dim)
        x = self.ln_1(x).view(x_size)
        
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        if hasattr(self, 'adapter'):
            adapt_mlp = self.adapter(x)
        y_size = x.size()
        y = x.view(-1, self.dim)
        y = self.ln_2(y).view(y_size)
        y = self.mlp(y)
        if hasattr(self, 'adapter'):
            y = y + adapt_mlp
        return x + y

class VisionTransformer(cnn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        self.dim = args.dim

        conv_in_channels = 3
        self.conv_proj = cnn.Conv2d(
                in_channels=conv_in_channels, out_channels=self.dim, kernel_size=self.patch_size, stride=self.patch_size
            )

        self.seq_length = (self.image_size // self.patch_size) ** 2
        
        self.class_token = cnn.Parameter(torch.zeros(1, self.dim))
        self.seq_length = self.seq_length + 1
        args.seq_length = self.seq_length

        self.encoder = Encoder(args)

        self.head = cnn.Linear(self.dim, args.num_classes)

        with crypten.no_grad():
            cnn.init.zeros_(self.head.weight)
            cnn.init.zeros_(self.head.bias)

            # Init the patchify stem
            fan_in = conv_in_channels * self.patch_size * self.patch_size
            cnn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                cnn.init.zeros_(self.conv_proj.bias)

    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!"
        assert w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!"
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]
    
        batch_class_token = crypten.stack([self.class_token.data for i in range(n)])
        x = crypten.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = x[:,0]

        x = self.head(x)

        return x


class Encoder(cnn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.seq_length = args.seq_length
        self.dim = args.dim
        
        self.pos_embedding = cnn.Parameter(torch.empty(1, self.seq_length, self.dim).normal_(std=0.02))
        self.dropout = cnn.Dropout(p=args.encoder_drop)
        self.layers = cnn.ModuleList()
        for i in range(args.num_encoderblk):
            self.layers.append(EncoderBlock(args=args))
        
        self.ln = cnn.BatchNorm2d(num_features=self.dim, eps=args.layer_norm_eps)

    def forward(self, x):
        assert len(x.size()) == 3, f"Expected (batch_size, seq_length, dim) got {x.size()}"
        x = x + self.pos_embedding.data
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        
        x_size = x.size()
        x = x.view(-1, self.dim)
        out = self.ln(x).view(x_size)
        return out
    
class transfer_scope_baseline(cnn.Module):
    def __init__(self,args):
        super(transfer_scope_baseline, self).__init__()
        self.dim = args.dim
        self.layers = cnn.ModuleList()
        for i in range(args.transfer_scope):
            self.layers.append(EncoderBlock(args=args))
        
        self.ln = cnn.BatchNorm2d(num_features=self.dim, eps=args.layer_norm_eps)
        self.head = cnn.Linear(self.dim, args.num_classes)

        with crypten.no_grad():
            cnn.init.zeros_(self.head.weight)
            cnn.init.zeros_(self.head.bias)

    def forward(self, x):
        assert len(x.size()) == 3, f"Expected (batch_size, seq_length, dim) got {x.size()}"

        for layer in self.layers:
            x = layer(x)
        
        x_size = x.size()
        x = x.view(-1, self.dim)
        x = self.ln(x).view(x_size)

        x = x[:,0]

        x = self.head(x)

        return x
    


class Adapter(cnn.Module):
    """Adapter block for secure inference"""
    def __init__(self,args):
        super(Adapter, self).__init__()
        self.dim = args.dim
        self.mlp_dim = args.mlp_dim
        self.down_size = args.bottleneck
        self.dropout = cnn.Dropout(p=args.adjuster_drop)
        self.scale = float(args.adjuster_scale)
        args.dim = args.bottleneck
        args.mlp_dim = args.bottleneck

        self.down_proj = cnn.Linear(self.dim, self.down_size)
        
        self.blks = cnn.ModuleList()
        for i in range(args.adjuster_n_blks):
            self.blks.append(EncoderBlock(args))
        
        self.up_proj = cnn.Linear(self.down_size, self.dim)

        with crypten.no_grad():
            cnn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            cnn.init.zeros_(self.up_proj.weight)
            cnn.init.zeros_(self.down_proj.bias)
            cnn.init.zeros_(self.up_proj.bias)
        
        #reset
        args.dim = self.dim
        args.mlp_dim = self.mlp_dim

        
    def forward(self, x):
        x = self.down_proj(x)
        x = self.dropout(x)
        for blk in self.blks:
            x = blk(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x

class Adjuster(cnn.Module):
    def __init__(self,args):
        super(Adjuster, self).__init__()
        self.dim = args.dim
        self.layers = cnn.ModuleList()
        for i in range(args.transfer_scope):
            self.layers.append(Adapter(args))
        
        self.ln = cnn.BatchNorm2d(num_features=self.dim, eps=args.layer_norm_eps)
        self.head = cnn.Linear(self.dim, args.num_classes)

        with crypten.no_grad():
            cnn.init.zeros_(self.head.weight)
            cnn.init.zeros_(self.head.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        x_size = x.size()
        x = x.view(-1, self.dim)
        x = self.ln(x).view(x_size)

        x = x[:,0]

        x = self.head(x)
        return x

    
class adaptformer(cnn.Module):
    def __init__(self, args):
        super(adaptformer, self).__init__()
        self.dim = args.dim
        self.r = 64
        self.Down = cnn.Linear(self.dim, self.r)
        self.act = cnn.ReLU()
        self.Up = cnn.Linear(self.r, self.dim)

    def forward(self, x):
        return self.Up(self.act(self.Down(x)))
    
class lora(cnn.Module):
    def __init__(self, args):
        super(lora, self).__init__()
        self.dim = args.dim
        self.r = 8
        self.A = cnn.Linear(self.dim, self.r)

        self.B = cnn.Linear(self.r, self.dim)

    def forward(self, x):
        return self.B(self.A(x))

