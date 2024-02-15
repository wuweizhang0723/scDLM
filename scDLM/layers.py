import math
import torch
from torch import einsum
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import numpy as np

# Activation function ----------------------
class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class EXP(nn.Module):
    def forward(self, x):
        return torch.exp(x)
# -------------------------------------------


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    


def ConvBlock(
    in_channels,
    out_channels,
    kernel_size,
    padding=0,
    stride=1,
    dilation=1,
    activation="relu",
):
    if activation == "elu":
        act_func = nn.ELU()
    elif activation == "sigmoid":
        act_func = nn.Sigmoid()
    elif activation == "gelu":
        act_func = GELU()
    elif activation == "exp":
        act_func = EXP()
    else:
        act_func = nn.ReLU()

    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        ),
        nn.BatchNorm1d(out_channels),
        act_func,
    )



# relative positional encoding functions  (from enformer-pytorch)--------------------------

def exists(val):
    return val is not None


# positional encoding

def get_positional_features_exponential(
    positions, features, seq_len, min_half_life=3.0
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8
):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings



def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# Attention Layer -----------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        dim,  # the input has shape (batch_size, len, dim) = (b, n, dim)
        *,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
        num_rel_pos_features=66,
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        # Q, K, V

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features  ###########

        self.to_rel_k = nn.Linear(
            self.num_rel_pos_features, dim_key * heads, bias=False
        )

        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits  # shape (b, h, n, n)
        attn = logits.softmax(dim=-1)  # softmax over the last dimension
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)  # (b, n, dim)