import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import *
from torch.nn.functional import linear, softmax, dropout


class TransformerEncoderRPR(Module):
    """PyTorch TransformerEncoder copied verbatim for RPR compatibility (Shaw et al. 2018)."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        return output


class TransformerEncoderLayerRPR(Module):
    """TransformerEncoderLayer modified to use MultiheadAttentionRPR."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, er_len=None):
        super().__init__()
        self.self_attn = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class MultiheadAttentionRPR(Module):
    """MultiheadAttention extended with a relative position embedding matrix Er."""

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, er_len=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.Er = Parameter(torch.rand((er_len, self.head_dim), dtype=torch.float32)) if er_len else None
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        return multi_head_attention_forward_rpr(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, rpr_mat=self.Er)


def multi_head_attention_forward_rpr(
        query, key, value, embed_dim_to_check, num_heads,
        in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn,
        dropout_p, out_proj_weight, out_proj_bias,
        training=True, key_padding_mask=None, need_weights=True,
        attn_mask=None, use_separate_proj_weight=False,
        q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
        static_k=None, static_v=None, rpr_mat=None):
    """Standard MHA forward extended with skew-optimised RPR (Huang et al. 2018)."""

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim
    scaling = float(head_dim) ** -0.5

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    if not use_separate_proj_weight:
        if qkv_same:
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif kv_same:
            _w, _b = in_proj_weight[:embed_dim], (in_proj_bias[:embed_dim] if in_proj_bias is not None else None)
            q = linear(query, _w, _b)
            _w, _b = in_proj_weight[embed_dim:], (in_proj_bias[embed_dim:] if in_proj_bias is not None else None)
            k, v = linear(key, _w, _b).chunk(2, dim=-1)
        else:
            q = linear(query, in_proj_weight[:embed_dim], in_proj_bias[:embed_dim] if in_proj_bias is not None else None)
            k = linear(key,   in_proj_weight[embed_dim:embed_dim*2], in_proj_bias[embed_dim:embed_dim*2] if in_proj_bias is not None else None)
            v = linear(value, in_proj_weight[embed_dim*2:], in_proj_bias[embed_dim*2:] if in_proj_bias is not None else None)
    else:
        q = linear(query, q_proj_weight, in_proj_bias[:embed_dim] if in_proj_bias is not None else None)
        k = linear(key,   k_proj_weight, in_proj_bias[embed_dim:embed_dim*2] if in_proj_bias is not None else None)
        v = linear(value, v_proj_weight, in_proj_bias[embed_dim*2:] if in_proj_bias is not None else None)

    q = q * scaling

    if bias_k is not None and bias_v is not None and static_k is None and static_v is None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) if k is not None else None
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) if v is not None else None

    if static_k is not None:
        k = static_k
    if static_v is not None:
        v = static_v

    src_len = k.size(1)

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))

    if rpr_mat is not None:
        rpr_mat = _get_valid_embedding(rpr_mat, q.shape[1], k.shape[1])
        qe = torch.einsum("hld,md->hlm", q, rpr_mat)
        attn_output_weights = attn_output_weights + _skew(qe)

    if attn_mask is not None:
        attn_output_weights = attn_output_weights + attn_mask.unsqueeze(0)

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    return attn_output, None


def _get_valid_embedding(Er, len_q, len_k):
    start = max(0, Er.shape[0] - len_q)
    return Er[start:, :]


def _skew(qe):
    sz = qe.shape[1]
    mask = (torch.triu(torch.ones(sz, sz).to(qe.device)) == 1).float().flip(0)
    qe = mask * qe
    qe = F.pad(qe, (1, 0, 0, 0, 0, 0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))
    return qe[:, 1:, :]
