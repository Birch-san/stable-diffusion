from dataclasses import dataclass
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from einops import rearrange, repeat
from typing import Optional

from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.tome.layer import SpatialTransformerLayerLocation, SpatialTransformerSelfAttnLocation, UNetLayerLocation
from ldm.modules.tome.merge import SourceDims, bipartite_soft_matching, kth_bipartite_soft_matching, merge_source, merge_wavg, random_bipartite_soft_matching
from ldm.modules.tome.params import BipartiteParams, GetMergeParams, KthBipartiteParams, MergeParams, RandomBipartiteParams
from ldm.modules.tome.tome_info import ToMeInfo

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    _tome_info: ToMeInfo
    location: SpatialTransformerSelfAttnLocation
    def __init__(self, query_dim, _tome_info: ToMeInfo, location: SpatialTransformerSelfAttnLocation, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self._tome_info = _tome_info
        self.location = location
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, get_merge_params: Optional[GetMergeParams]=None) -> Tensor:
        is_self_attention = context is None
        h = self.heads

        q = self.to_q(x)
        context: Tensor = default(context, x)
        del x
        batch_size, token_count, _ = context.shape
        tome_source_dims = SourceDims(batch_size, token_count)
        tome_source_device = context.device
        k = self.to_k(context)
        v = self.to_v(context)
        del context

        if is_self_attention and callable(get_merge_params):
            merge_params: Optional[MergeParams] = get_merge_params(token_count=token_count, layer=self.location)
            if merge_params is not None:
                # Apply ToMe here
                k_heads_extracted: Tensor = rearrange(k, 'b n (h d) -> b n h d', h=h)
                k_mean = k_heads_extracted.mean(-2)
                del k_heads_extracted

                match merge_params:
                    case BipartiteParams(r):
                        merge, _ = bipartite_soft_matching(
                            k_mean,
                            r
                        )
                    case RandomBipartiteParams(r):
                        merge, _ = random_bipartite_soft_matching(
                            k_mean,
                            r
                        )
                    case KthBipartiteParams(k_):
                        merge, _ = kth_bipartite_soft_matching(
                            k_mean,
                            k_
                        )
                    case _:
                        raise TypeError(f"That ({merge_params}) ain't no MergeParams I ever heard of")
                
                del k_mean
                if self._tome_info.trace_source:
                    self._tome_info.source = merge_source(
                        merge, tome_source_dims, self._tome_info.source, device=tome_source_device
                    )
                k = merge(k, mode='mean')
                v = merge(v, mode='mean')
                del merge

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
            del mask

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        del sim

        out = einsum('b i j, b j d -> b i d', attn, v)
        del attn, v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    _tome_info: ToMeInfo
    location: SpatialTransformerSelfAttnLocation
    def __init__(self, dim, n_heads, d_head, _tome_info: ToMeInfo, location: SpatialTransformerSelfAttnLocation, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.location=location
        self._tome_info=_tome_info
        self.location=location
        self.attn1 = CrossAttention(query_dim=dim, _tome_info=_tome_info, location=location, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, _tome_info=_tome_info, location=location, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, get_merge_params: Optional[GetMergeParams]=None):
        return checkpoint(self._forward, (x, context, get_merge_params), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, get_merge_params: Optional[GetMergeParams]=None):
        x = x.contiguous() if x.device.type == 'mps' else x
        x = self.attn1(self.norm1(x), get_merge_params=get_merge_params) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    _tome_info: ToMeInfo
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, _tome_info: ToMeInfo, unet_location: UNetLayerLocation,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self._tome_info = _tome_info
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, _tome_info=self._tome_info, location=SpatialTransformerSelfAttnLocation(
                unet_location=unet_location,
                spatial_transformer_location=SpatialTransformerLayerLocation(
                    spatial_transformer_block_depth=d
                )
            ), dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, get_merge_params: Optional[GetMergeParams]=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x, context=context, get_merge_params=get_merge_params)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in