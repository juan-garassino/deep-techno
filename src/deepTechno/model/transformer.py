import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from deepTechno.model.constants import VOCAB_SIZE, TOKEN_END, TOKEN_PAD, TORCH_LABEL_TYPE
from deepTechno.utils.device import get_device
from deepTechno.model.positional_encoding import PositionalEncoding
from deepTechno.model.rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


class MusicTransformer(nn.Module):
    """
    Decoder-only Music Transformer (Huang et al. 2018, https://arxiv.org/abs/1809.04281).
    Implements causal self-attention via a masked encoder stack with a DummyDecoder.
    Supports Relative Position Representations (Shaw et al. 2018) via rpr=True.
    """

    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False):
        super().__init__()
        self.dummy = DummyDecoder()
        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = max_sequence
        self.rpr = rpr

        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        if not self.rpr:
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy,
            )
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(
                self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder,
            )

        self.Wout = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=-1)

        # Optional cross-attention for vision/audio conditioning (Phase 2)
        self.cross_attn = nn.MultiheadAttention(self.d_model, self.nhead,
                                                dropout=self.dropout, batch_first=True)
        self.cross_norm  = LayerNorm(self.d_model)

    def forward(self, x, mask=True, encoder_memory=None):
        if mask:
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)
        x_out = x_out.permute(1, 0, 2)

        # Cross-attention to encoder memory when conditioning on vision/audio
        if encoder_memory is not None:
            attn_out, _ = self.cross_attn(x_out, encoder_memory, encoder_memory)
            x_out = self.cross_norm(x_out + attn_out)

        y = self.Wout(x_out)
        del mask
        return y

    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0):
        assert not self.training, "Cannot generate while in training mode"

        gen_seq = torch.full((1, target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while cur_i < target_seq_length:
            y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END]
            token_probs = y[:, cur_i - 1, :]

            if beam == 0:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token
                if next_token == TOKEN_END:
                    break
            else:
                beam_ran = random.uniform(0, 1)
                if beam_ran <= beam_chance:
                    top_res, top_i = torch.topk(token_probs.flatten(), beam)
                    beam_rows = top_i // VOCAB_SIZE
                    beam_cols = top_i % VOCAB_SIZE
                    gen_seq = gen_seq[beam_rows, :]
                    gen_seq[..., cur_i] = beam_cols
                else:
                    distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                    gen_seq[:, cur_i] = distrib.sample()

            cur_i += 1

        return gen_seq[:, :cur_i]


class DummyDecoder(nn.Module):
    """Pass-through decoder that returns encoder memory — makes nn.Transformer decoder-only."""

    def forward(self, tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask):
        return memory
