from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


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


# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.scale = dim_head ** -0.5
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
# #         self.attn_soft = nn.Softmax(dim=-1)
# #         self.attn_soft = nn.Identity() 
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, context=None, mask=None):
#         h = self.heads

#         q = self.to_q(x)
#         context = default(context, x)
#         k = self.to_k(context)
#         v = self.to_v(context)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#         if exists(mask):
#             mask = rearrange(mask, 'b ... -> b (...)')
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b j -> (b h) () j', h=h)
#             sim.masked_fill_(~mask, max_neg_value)

#         # attention, what we cannot get enough of
# #         attn = self.attn_soft(sim)
#         attn = sim.softmax(dim=-1)
# #         attn = self.attn_soft(attn)
#         out = einsum('b i j, b j d -> b i d', attn, v)
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#         return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
#         self.attn_soft = nn.Softmax(dim=-1)
#         self.attn_soft = nn.Identity() 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, prompt=None):
        h = self.heads

        if prompt is None:
            """
            x: [b, hw, dx] i.e., [2, 4096, 320] where hw is the input size, dx is input dimension/depth
            context: [b, m, dc] i.e., [2, 77, 768] where m is the sequence length, dc is the context dimension
            prompt: None
            """
            
            # if context is not None:
            #     import pdb; pdb.set_trace()

            query = self.to_q(x) # [2, 4096, 320]
            new_context = default(context, x)
            key = self.to_k(new_context) # [2, 77, 320]
            value = self.to_v(new_context) # [2, 77, 320]

            query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (query, key, value)) 
            # after rearrange: query: [16, 4096, 40], key: [16, 77, 40], value: [16, 77, 40], where h=8 is number of heads

            sim = einsum('b i d, b j d -> b i j', query, key) * self.scale # [16, 4096, 77]

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
    #         attn = self.attn_soft(sim)
            attn = sim.softmax(dim=-1) # [16, 4096, 77]
    #         attn = self.attn_soft(attn)
            out = einsum('b i j, b j d -> b i d', attn, value) # [16, 4096, 40]
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h) # [2, 4096, 320]
        
        else:
            assert context.shape[0] == x.shape[0]
            assert context is not None
            assert type(prompt) is list
            # import pdb; pdb.set_trace()

            """
            x: [b, hw, dx] i.e., [1, 64, 1280] where hw is the input size, dx is input dimension/depth
            context: [b, m, dc] i.e., [1, 77, 768] where m is the sequence length, dc is the context dimension
            prompt: [1, m, dc] i.e., [1, 77, 768] where m is the sequence length, dc is the context dimension
            """

            assert prompt[0].shape[0] == 1
            assert prompt[1].shape[0] == 1
            # assert x.shape[0] == 1 # currently support only batch size 1

            # repeat prompt to match the batch size
            if x.shape[0] > 1:
                prompt_k = torch.repeat_interleave(prompt[0], x.shape[0], dim=0) # [1, 77, 768] -> [2, 77, 768]
                prompt_v = torch.repeat_interleave(prompt[1], x.shape[0], dim=0) # [1, 77, 768] -> [2, 77, 768]
            else:
                prompt_k = prompt[0]
                prompt_v = prompt[1]

            new_context_k = torch.cat([context, prompt_k], dim=1) # [1, 154, 1280]
            new_context_v = torch.cat([context, prompt_v], dim=1) # [1, 154, 1280]

            query = self.to_q(x)  # [1, 64, 1280]
            key = self.to_k(new_context_k) # [1, 154, 1280]
            value = self.to_v(new_context_v) # [1, 154, 1280]

            query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (query, key, value))
            # after rearrange: query: [8, 64, 160], key: [8, 154, 160], value: [8, 154, 160], where h=8 is number of heads

            sim = einsum('b i d, b j d -> b i j', query, key) * self.scale # [8, 64, 154]

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
            
            attn = sim.softmax(dim=-1) # [8, 64, 154]

            out = einsum('b i j, b j d -> b i d', attn, value) # [8, 64, 160]
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h) # [1, 64, 1280]

            assert out.shape == x.shape

        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, prompt=None):
        return checkpoint(self._forward, (x, context, prompt), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, prompt=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context, prompt=prompt) + x  # CHANGE HERE
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=False, # CHANGE HERE
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )

        # self.transformer_blocks = nn.ModuleList(
        #     [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
        #                            disable_self_attn=disable_self_attn)
        #         for d in range(depth-1)
        #     ] + [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=False,
        #                            disable_self_attn=disable_self_attn)]
        # )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, prompt=None):
        # note: if no context is given, cross-attention defaults to self-attention

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context, prompt=prompt)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in