# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Hongyu Xiang,
#         Keyu An,
#         Huahuan Zheng (maxwellzh@outlook.com)

"""basic nn layers impl"""

import math
import pickle
from collections import OrderedDict
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StackDelta(nn.Module):
    """Stack the features from 120 into 40 x 3

    in: [batch, len, 120]
    out: [batch, 3, len, 40]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3
        assert x.size(2) % 3 == 0

        x = x.view(x.size(0), x.size(1), 3, x.size(2)//3)
        if x.requires_grad:
            out = x.transpose(1, 2).contiguous()
        else:
            out = x.transpose_(1, 2).contiguous()
        return out


class UnStackDelta(nn.Module):
    """Reverse of StackDelta"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4

        if x.requires_grad:
            out = x.transpose(1, 2).contiguous()
        else:
            out = x.transpose_(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), out.size(2) * out.size(3))
        return out


'''
Conv2dSubsampling: From Google TensorflowASR
https://github.com/TensorSpeech/TensorFlowASR/blob/5fbd6a89b93b703888662f5c47d05bae256e98b0/tensorflow_asr/models/layers/subsampling.py
Originally wrote with Tensorflow, translated into PyTorch by Huahuan.
'''


class Unsqueeze(nn.Module):
    """For pickle model"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


class Conv2dSubdampling(nn.Module):
    def __init__(self, multiplier: int, norm: Literal['batch', 'layer', 'none'] = 'none', stacksup: bool = False):
        super().__init__()
        self._lens_in_args_ = None

        if stacksup:
            self.stack = StackDelta()
            idim = 3
        else:
            self.stack = Unsqueeze(1)
            idim = 1

        if norm == 'none':
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
                nn.Conv2d(idim, multiplier, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
                nn.Conv2d(multiplier, multiplier, kernel_size=3, stride=2),
                nn.ReLU()
            )
            return
        elif norm == 'batch':
            Norm = nn.BatchNorm2d
        elif norm == 'layer':
            Norm = nn.LayerNorm
        else:
            raise RuntimeError(
                f"Unknown normalization method: {norm}, expect one of ['batch', 'layer', 'none']")
        self.conv = nn.Sequential(
            nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
            nn.Conv2d(idim, multiplier, kernel_size=3, stride=2),
            Norm(multiplier),
            nn.ReLU(),
            nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
            nn.Conv2d(multiplier, multiplier, kernel_size=3, stride=2),
            Norm(multiplier),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4 (not strictly).
        """
        # [B, T, D] -> [B, C, T, D]
        x = self.stack(x)
        # [B, C, T, D] -> [B, OD, T//4, D//4]
        out = self.conv(x)
        B, OD, NT, ND = out.size()
        # [B, OD, T//4, D//4] -> [B, T//4, OD, D//4]
        out = out.permute(0, 2, 1, 3)
        # [B, T//4, OD, D//4] -> [B, T//4, OD * D//4]
        out = out.contiguous().view(B, NT, OD*ND)
        lens_out = torch.div(lens, 2, rounding_mode='floor')
        lens_out = torch.div(lens_out, 2, rounding_mode='floor').to(lens.dtype)
        return out, lens_out


'''NOTE (Huahuan):
VGG2L with 1/4 subsample of time dimension.
In ESPNET impl, there is no normalization, so I just follow it.
Reference:
https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/rnn/encoders.html#VGG2L
'''


class VGG2LSubsampling(nn.Module):
    def __init__(self, in_channel: int = 1):
        '''VGG 1/4 subsample layer

        Input: shape (B, T, D)
        Ouput: shape (B, T//4, 128*(D//C//4)),
            C: in_channel, ensure `D % C == 0`

        '''
        super().__init__()

        self.vgg_conv = nn.Sequential(
            # first block
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            # second block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.in_channel = in_channel

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # [B, T, D] -> [B, C, T, D//C]
        x = x.view(
            x.size(0),
            x.size(1),
            self.in_channel,
            x.size(2) // self.in_channel,
        ).transpose(1, 2)

        # [B, C, T, D//C] -> [B, 128, T//4, D//C//4]
        conv_out = self.vgg_conv(x)  # type:torch.Tensor
        # [B, 128, T//4, D//C//4] -> [B, T//4, 128, D//C//4]
        transposed_out = conv_out.transpose(1, 2)
        # [B, T//4, 128, D//C//4] -> [B, T//4, 128 * (D//C//4)]
        contiguous_out = transposed_out.contiguous().view(
            transposed_out.size(0), transposed_out.size(1), -1)

        lens_out = torch.ceil(torch.ceil(lens/2).to(lens.dtype)/2)
        return contiguous_out, lens_out


'''
PositionalEncoding: From PyTorch implementation
The origin impl return pos_enc + x. Our impl only return pos_enc
'''


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=4096):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return self.pe[:x.size(1), :]


'''
Relative positional multi-head attention implementation of Transformer-XL
from huggingface
https://github.com/huggingface/transformers/blob/1bdf42409c452a767ac8e2119bceb8f5c704c8f1/src/transformers/models/transfo_xl/modeling_transfo_xl.py
And part of the codes are modified according to TensorflowASR impl
https://github.com/TensorSpeech/TensorFlowASR/blob/5fbd6a89b93b703888662f5c47d05bae256e98b0/tensorflow_asr/models/layers/multihead_attention.py
'''


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(
        self,
        idim: int,
        n_head: int,
        d_head: int,
        pe: PositionalEncoding,
        dropatt: float = 0.0
    ):

        super().__init__()

        self.pe = pe

        self.n_head = n_head
        self.d_model = idim
        self.d_head = d_head

        self.call_qkv = nn.Linear(idim, 3 * n_head * d_head, bias=False)

        self.dropoutatt = nn.Dropout(dropatt)
        self.linearout = nn.Linear(n_head * d_head, idim, bias=False)

        self.scale = 1 / (d_head ** 0.5)

        self.r_r_bias = nn.Parameter(
            torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(
            torch.FloatTensor(self.n_head, self.d_head))

        torch.nn.init.xavier_uniform_(self.r_r_bias)
        torch.nn.init.xavier_uniform_(self.r_w_bias)

        self.linearpos = nn.Linear(
            self.d_model, self.n_head * self.d_head, bias=False)

    @staticmethod
    def _rel_shift(x: torch.Tensor):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = x.new_zeros(zero_pad_shape)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, word_embedding: torch.Tensor, lens: torch.Tensor, mems=None):
        """Relative positional embedding multihead attention

        B: batch size
        T: (maximum) sequence length
        E: word embedding size (also the postional embedding size)
        C: memory context length, K=C+T

        Args:
            word_embedding (torch.Tensor): input segment (word) embedding, size [B, T, E]
            lens (torch.Tensor): input sequence lengths, used to generate attention mask, size [B]
            mems (torch.Tensor, None): if not None, memory tensors, size [B, C, E]
        
        Returns:
            torch.Tensor: attention out

        """
        '''
        word_embedding: [B, T, E] -> [T, B, E]
        mems: [B, C, E] -> [C, B, E] (if not None)
        '''
        word_embedding = word_embedding.transpose(0, 1).contiguous()
        len_seq, batchsize = word_embedding.size()[:2]
        len_k = len_seq
        if all(le == lens[0] for le in lens):
            # set mask to None
            attn_mask = None
            if mems is not None:
                mems = mems.transpose(0, 1).contiguous()
                len_k += mems.size(0)
        else:
            if mems is not None:
                mems = mems.transpose(0, 1).contiguous()
                lens = lens + mems.size(0)
                len_k += mems.size(0)
            # attn_mask: [K, B]_{1,0}
            attn_mask = torch.arange(len_k, device=word_embedding.device)[
                :, None] >= lens[None, :].to(word_embedding.device)

        # pos_embedding: [K, E]
        pos_enc = self.pe(torch.empty((1, len_k)))

        if mems is not None:
            # embed_with_mem: cat([C, B, E], [T, B, E]) -> [C+T, B, E] = [K, B, E]
            embed_with_mem = torch.cat([mems, word_embedding], dim=0)
            '''
            W_heads: f([K, B, E]) -> [K, B, 3*H*D]
                H: n_heads
                D: d_heads
            '''
            W_heads = self.call_qkv(embed_with_mem)

            # R_head_k: f([K, E]) -> [K, H*D]
            R_head_k = self.linearpos(pos_enc)

            # W_head_q/W_head_k/W_head_v: f([K, B, 3HD]) -> [K, B, HD]
            W_head_q, W_head_k, W_head_v = torch.chunk(W_heads, 3, dim=-1)

            # W_head_q: f([K, B, HD]) -> [T, B, HD]
            W_head_q = W_head_q[-len_seq:]

        else:
            # W_heads: f([T, B, E]) -> [T, B, 3*H*D]
            W_heads = self.call_qkv(word_embedding)

            # R_head_k: f([T, E]) -> [T, H*D]
            R_head_k = self.linearpos(pos_enc)

            # W_head_q/W_head_k/W_head_v: f([T, B, 3HD]) -> [T, B, HD]
            W_head_q, W_head_k, W_head_v = torch.chunk(W_heads, 3, dim=-1)

        # W_head_q: [T, B, HD] -> [T, B, H, D]
        W_head_q = W_head_q.view(len_seq, batchsize, self.n_head, self.d_head)

        # W_head_k/W_head_v: [K, B, HD] -> [K, B, H, D]
        W_head_k = W_head_k.view(len_k, batchsize, self.n_head, self.d_head)
        W_head_v = W_head_v.view_as(W_head_k)

        # R_head_k: [K, H*D] -> [K, H, D]
        R_head_k = R_head_k.view(len_k, self.n_head, self.d_head)

        # compute attention score

        # RW_head_q/RR_head_q: [T, B, H, D]
        RW_head_q = W_head_q + self.r_w_bias
        RR_head_q = W_head_q + self.r_r_bias

        # FIXME: torch.einsum is not optimized, which might cause slow computation
        # AC: f([T, B, H, D], [K, B, H, D]) -> [T, K, B, H]
        AC = torch.einsum("ibnd,jbnd->ijbn", (RW_head_q, W_head_k))

        # BD: f([T, B, H, D], [K, H, D]) -> [T, K, B, H]
        BD = torch.einsum("ibnd,jnd->ijbn", (RR_head_q, R_head_k))

        # BD: [T, K, B, H] -> [T, K, B, H]
        BD = self._rel_shift(BD)

        # attn_score: [T, K, B, H]
        attn_score = AC + BD
        attn_score *= self.scale

        # compute attention probability
        if attn_mask is not None:
            # use in-plcae fill
            attn_score = attn_score.masked_fill_(
                attn_mask[None, :, :, None], -float('inf'))

        # attn_prob: f([T, K, B, H]) -> [T, K, B, H]
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.dropoutatt(attn_prob)

        # compute attention vector

        # attn_vec: f([T, K, B, H], [K, B, H, D]) -> [T, B, H, D]
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, W_head_v))

        # attn_vec: [T, B, H, D] -> [T, B, HD]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1),
            self.n_head * self.d_head)

        # attn_out: f([T, B, HD]) -> [T, B, E]
        attn_out = self.linearout(attn_vec)

        # attn_out: [T, B, E] -> [B, T, E]
        attn_out = attn_out.transpose(0, 1).contiguous()

        return attn_out


class StandardRelPositionalMultiHeadAttention(RelPositionMultiHeadAttention):
    def __init__(self, idim: int, n_head: int, pe: PositionalEncoding, dropatt: float = 0):
        super().__init__(idim, n_head, idim//n_head, pe, dropatt=dropatt)


class FFModule(nn.Module):
    """Feed-forward module

    default output dimension = idim
    x0 -> LayerNorm -> FC -> Swish -> Dropout -> FC -> Dropout -> x1
    x0 + res_factor * x1 -> output
    """

    def __init__(self, idim: int, res_factor: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        assert res_factor > 0. and dropout >= 0.
        self._res_factor = res_factor

        self.ln = nn.LayerNorm([idim])
        self.fc0 = nn.Linear(idim, idim*4)
        self.swish = nn.SiLU()
        self.dropout0 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(idim*4, idim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        output = self.ln(x)
        output = self.fc0(output)
        output = self.swish(output)
        output = self.dropout0(output)
        output = self.fc1(output)
        output = self.dropout1(output)
        output = x + self._res_factor * output

        return output


class ConvModule(nn.Module):
    def __init__(self, idim: int, kernel_size: int = 32, dropout: float = 0.0, multiplier: int = 1) -> None:
        super().__init__()

        self.ln = nn.LayerNorm([idim])
        self.pointwise_conv0 = nn.Conv1d(
            idim, 2 * idim, kernel_size=1, stride=1)
        self.glu = nn.GLU(dim=1)
        cdim = idim
        padding = (kernel_size-1)//2
        self.padding = nn.ConstantPad2d(
            (padding, kernel_size-1-padding, 0, 0), 0.)
        self.depthwise_conv = nn.Conv1d(
            cdim, multiplier*cdim, kernel_size=kernel_size, stride=1, groups=cdim, padding=0)
        cdim = multiplier * cdim
        self.bn = nn.BatchNorm1d(cdim)
        self.swish = nn.SiLU()
        self.pointwise_conv1 = nn.Conv1d(cdim, idim, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # [B, T, D]
        output = self.ln(x)
        # [B, T, D] -> [B, D, T]
        output = output.transpose(1, 2)
        # [B, D, T] -> [B, 2*D, T]
        output = self.pointwise_conv0(output)
        # [B, 2D, T] -> [B, D, T]
        output = self.glu(output)
        # [B, D, T] -> [B, multiplier*D, T]
        output = self.padding(output)
        output = self.depthwise_conv(output)
        if output.size(0) > 1 or output.size(2) > 1:
            # Doing batchnorm with [1, D, 1] tensor raise error.
            output = self.bn(output)
        output = self.swish(output)
        # [B, multiplier*D, T] -> [B, D, T]
        output = self.pointwise_conv1(output)
        output = self.dropout(output)
        # [B, D, T] -> [B, T, D]
        output = output.transpose(1, 2)

        return x + output


class MHSAModule(nn.Module):
    def __init__(self, idim, d_head: int, num_heads: int, pe: PositionalEncoding, dropout: float = 0.0, dropout_attn: float = 0.0):
        super().__init__()

        self.ln = nn.LayerNorm(idim)
        if d_head == -1:
            # a "standard" multi-head attention
            self.mha = StandardRelPositionalMultiHeadAttention(
                idim, num_heads, pe, dropout_attn)
        else:
            self.mha = RelPositionMultiHeadAttention(
                idim, num_heads, d_head, pe, dropout_attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lens: torch.Tensor, mems=None):
        x_norm = self.ln(x)
        attn_out = self.mha(x_norm, lens, mems)
        attn_out = self.dropout(attn_out)
        return x + attn_out, lens


class ConformerCell(nn.Module):
    def __init__(
            self,
            idim: int,
            pe: PositionalEncoding,
            res_factor: float = 0.5,
            d_head: int = 36,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1,
            dropout_attn: float = 0.0):
        super().__init__()

        self.ffm0 = FFModule(idim, res_factor, dropout)
        self.mhsam = MHSAModule(idim, d_head, num_heads,
                                pe, dropout, dropout_attn)
        self.convm = ConvModule(
            idim, kernel_size, dropout, multiplier)
        self.ffm1 = FFModule(idim, res_factor, dropout)
        self.ln = nn.LayerNorm(idim)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):

        ffm0_out = self.ffm0(x)
        attn_out, attn_ls = self.mhsam(ffm0_out, lens)
        conv_out = self.convm(attn_out)
        ffm1_out = self.ffm1(conv_out)
        out = self.ln(ffm1_out)
        return out, attn_ls


class VGG2L(nn.Module):
    def __init__(self, in_channel=4):
        super(VGG2L, self).__init__()
        kernel_size = 3
        padding = 1
        self.conv1_1 = torch.nn.Conv2d(
            in_channel, 64, kernel_size, stride=1, padding=padding)
        self.conv1_2 = torch.nn.Conv2d(
            64, 64, kernel_size, stride=1, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2_1 = torch.nn.Conv2d(
            64, 128, kernel_size, stride=1, padding=padding)
        self.conv2_2 = torch.nn.Conv2d(
            128, 128, kernel_size, stride=1, padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = self.bn1(xs_pad)
        xs_pad = F.max_pool2d(xs_pad, [1, 2], stride=[1, 2], ceil_mode=True)
        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = self.bn2(xs_pad)
        xs_pad = F.max_pool2d(xs_pad, [1, 2], stride=[1, 2], ceil_mode=True)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens


class Lookahead(nn.Module):
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        return x


class TDNN(nn.Module):
    def __init__(self, idim: int, odim: int, half_context: int = 1, dilation: int = 1, stride: int = 1):
        super().__init__()

        self.stride = stride
        if stride > 1:
            dilation = 1
            padding = 0
        else:
            padding = half_context * dilation

        self.conv = torch.nn.Conv1d(
            idim, odim, 2*half_context+1, stride=stride, padding=padding, dilation=dilation)
        # NOTE (Huahuan): I think we should use layernorm instead of batchnorm
        # self.bn = MaskedBatchNorm1d(odim, eps=1e-5, affine=True)
        self.ln = nn.LayerNorm(odim)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tdnn_in = x.transpose(1, 2)
        if self.stride > 1:
            ilens = ilens // self.stride

        tdnn_out = self.conv(tdnn_in)

        output = F.relu(tdnn_out)
        output = output.transpose(1, 2)
        output = self.ln(output)
        return output, ilens

class _LSTM(nn.Module):
    def __init__(self, idim: int, hdim: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False, proj_size: int = 0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=idim,
            hidden_size=hdim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
            proj_size=proj_size
        )

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        self.lstm.flatten_parameters()

        packed_input = pack_padded_sequence(
            x, ilens.to("cpu"), batch_first=True)
        packed_output, _ = self.lstm(packed_input, hidden)
        out, olens = pad_packed_sequence(packed_output, batch_first=True)

        return out, olens


class TimeReduction(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        assert N >= 1 and isinstance(N, int)
        if N == 1:
            print("WARNING: TimeReduction factor was set to 1.")
        self.N = N

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        return x[:, ::self.N, :].clone(), torch.div(x_lens-1, self.N, rounding_mode='floor') + 1


class CausalConv2d(nn.Module):
    """
    Causal 2d-conv. Applied to (N, C, T, U) dim tensors.

    Args:
        in_channels  (int): the input dimension (namely K)
        out_channels (int): the output dimension
        kernel_size  (int): the kernel size (kernel is square)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], islast: bool = False):
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"Invalid initialization for CausalConv2d: {in_channels}, {out_channels}, {kernel_size}")

        if islast:
            self.causal_conv = nn.Sequential(OrderedDict([
                # seperate convlution
                ('depth_conv', nn.Conv2d(in_channels, in_channels,
                 kernel_size=kernel_size, groups=in_channels)),
                ('point_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1))
                # 'conv': nn.Conv2d(in_channels, out_channels, kernel_size)
            ]))
        else:
            # NOTE (Huahuan): I think a normalization is helpful so that the padding won't change the distribution of features.
            self.causal_conv = nn.Sequential(OrderedDict([
                # seperate convlution
                ('depth_conv', nn.Conv2d(in_channels, in_channels,
                 kernel_size=kernel_size, groups=in_channels)),
                ('point_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                ('relu', nn.ReLU(inplace=True)),
                ('bn', nn.BatchNorm2d(in_channels)),
                # 'conv': nn.Conv2d(in_channels, out_channels, kernel_size)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.causal_conv(x)


def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.
    
    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
    
    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
        .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's BatchNorm1d implementation for argument details.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(num_features, eps, momentum,
                                                affine, track_running_stats)

    def forward(self, inp, lengths):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0

        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = mask.sum()
        mask = mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp**2).sum([0, 2]) - mean**2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] +
                                                        self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp


class SampledSoftmax(nn.Module):
    """
    Conduct softmax on given target labels (to reduce memory consumption).

    Args:
        blank (int): CTC/RNN-T blank index, default -1.
        uniform_ratio (float): uniformly draw indices apart from target labels via given ratio.
    """

    def __init__(self, blank: int = -1, uniform_ratio: Optional[float] = 0.0) -> None:
        super().__init__()
        assert isinstance(blank, int)

        if blank != -1:
            # CTC / RNN-T keep the first place for <blk>, currently only support <blk>=0
            assert blank == 0
            self.pad = nn.ConstantPad1d((1, 0), 0)

        self.blank_id = blank

        assert isinstance(uniform_ratio, (int, float))
        assert uniform_ratio >= 0 and uniform_ratio < 1
        self.uniform_ratio = float(uniform_ratio)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):

        orin_shape = labels.shape
        if labels.dim() > 1:
            labels = labels.flatten()

        dtype = labels.dtype
        if self.uniform_ratio == 0.0:
            indices_sampled, labels = torch.unique(
                labels, sorted=True, return_inverse=True)
        else:
            n_uniform = int(x.size(-1) * self.uniform_ratio)
            assert n_uniform > 0
            concat_labels = torch.cat(
                [
                    labels,
                    torch.randperm(x.size(-1), device=labels.device,
                                   dtype=dtype)[:n_uniform]
                ],
                dim=0
            )
            indices_sampled, concat_labels = torch.unique(
                concat_labels, sorted=True, return_inverse=True)
            labels = concat_labels[:labels.size(0)]

        indices_sampled = indices_sampled.to(torch.long)
        labels = labels.to(dtype)

        if self.blank_id != -1:
            if indices_sampled[0] != self.blank_id:
                indices_sampled = self.pad(indices_sampled)
                labels += 1

        return x[..., indices_sampled].log_softmax(dim=-1), labels.reshape(orin_shape)


class SyllableEmbedding(nn.Module):
    """The char to syllable mapping is generated by pypinyin module."""

    def __init__(self, num_classes: int, dim_emb: int, syllable_data: str) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(num_classes, dim_emb)

        with open(syllable_data, 'rb') as fib:
            data = pickle.load(fib)
        self.register_buffer('converter', torch.from_numpy(
            data['converter']), persistent=False)
        self.syllable_embedding = nn.Embedding(
            data['num_syllables'], dim_emb)
        del data

    def forward(self, x: torch.Tensor):
        xshape = x.shape
        x_syllable = self.converter[x.view(-1)].reshape(xshape)
        return self.char_embedding(x) + self.syllable_embedding(x_syllable)
