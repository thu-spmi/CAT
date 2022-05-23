"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan
"""

import math
from collections import OrderedDict
from maskedbatchnorm1d import MaskedBatchNorm1d

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


class Conv2dSubdampling(nn.Module):
    def __init__(self, multiplier: int, stacksup: bool = True):
        super().__init__()
        self._lens_in_args_ = None
        def _unsqueeze(x): return x.unsqueeze(1)

        if stacksup:
            self.stack = StackDelta()
            idim = 3
        else:
            self.stack = _unsqueeze
            idim = 1

        self.conv = nn.Sequential(
            nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
            nn.Conv2d(idim, multiplier, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
            nn.Conv2d(multiplier, multiplier, kernel_size=3, stride=2),
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
        # NOTE (Huahuan): use torch.div() instead '//'
        lens_out = torch.div(lens, 2, rounding_mode='floor')
        lens_out = torch.div(lens_out, 2, rounding_mode='floor')
        return out, lens_out


'''
PositionalEncoding: From PyTorch implementation
The origin impl return pos_enc + x. Our impl only return pos_enc
'''


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

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
        dropatt: float = 0.
    ):

        super().__init__()

        # Define positional encoding in SuperNet level for efficient memory
        self.pe = None

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
                attn_mask[None, :, :, None], -1e30)

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
    def __init__(self, idim, d_head: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.ln = nn.LayerNorm(idim)
        self.mha = RelPositionMultiHeadAttention(
            idim, num_heads, d_head)
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
            res_factor: float = 0.5,
            d_head: int = 36,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1):
        super().__init__()

        self.ffm0 = FFModule(idim, res_factor, dropout)
        self.mhsam = MHSAModule(idim, d_head, num_heads, dropout)
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


class VGG2L(torch.nn.Module):
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


class TDNN(torch.nn.Module):
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
        # FIXME: (Huahuan) I think we should use layernorm instead of batchnorm
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
    def __init__(self, idim, hdim, n_layers, dropout=0.0, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(idim, hdim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        self.lstm.flatten_parameters()

        packed_input = pack_padded_sequence(
            x, ilens.to("cpu"), batch_first=True)
        packed_output, _ = self.lstm(packed_input, hidden)
        out, olens = pad_packed_sequence(packed_output, batch_first=True)

        return out, olens


class DeformTDNNlayer(nn.Module):
    def __init__(self, idim=120, hdim=640, dropout=0.5, kernel_size=5, dilation=1, padding=2, stride=1, bias=None, modulation=False, low_latency=False):
        """
        Keyu An, Yi Zhang, Zhijian Ou, "Deformable TDNN with adaptive receptive fields for speech recognition", INTERSPEECH 2021.
        
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformTDNNlayer, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv1d(
            idim, hdim, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv1d(
            idim, kernel_size, kernel_size=5, padding=2, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        self.low_latency = low_latency
        if low_latency:
            print("low latency")
        setattr(self, "dropout", torch.nn.Dropout(dropout))
        if modulation:
            print("use modulation !")
            self.m_conv = nn.Conv1d(
                idim, kernel_size, kernel_size=5, padding=2, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        l = x.size(1)
        c = x.size(2)
        x = x.transpose(1, 2)
        offset = self.p_conv(x)

        if self.low_latency:
            zero = torch.zeros_like(offset)
            offset = torch.where(offset > 0, zero, offset)

        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1)

        p = self._get_p(ks, l, c, offset, dtype, self.dilation)
        p = p.contiguous().permute(0, 2, 1)
        p = torch.clamp(p, 0, x.size(2)-1)
        q_l = p.detach().floor().long()
        q_r = q_l + 1
        q_l = torch.clamp(q_l, 0, x.size(2)-1)
        q_r = torch.clamp(q_r, 0, x.size(2)-1)
        g_l = (1 + (q_l.type_as(p) - p))
        g_r = (1 - (q_r.type_as(p) - p))
        x_q_l = self._get_x_q(x, q_l, N)
        x_q_r = self._get_x_q(x, q_r, N)
        x_offset = g_l.unsqueeze(dim=1) * x_q_l + \
            g_r.unsqueeze(dim=1) * x_q_r
        if self.modulation:
            m = m.contiguous().permute(0, 2, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        out = F.relu(out, inplace=True)
        out = out.transpose(1, 2)
        out = F.layer_norm(out, [out.size()[-1]])
        dropout = getattr(self, 'dropout')
        out = dropout(out)
        return out

    def _get_p_n(self, N, dtype, dilation):
        p_n = torch.arange(-(self.kernel_size-1)*dilation//2,
                           (self.kernel_size-1)*dilation//2+1, dilation)
        p_n = p_n.view(1, N, 1).type(dtype)

        return p_n

    def _get_p_0(self, l, c, N, dtype):
        p_0 = torch.arange(0, l*self.stride, self.stride)
        p_0 = torch.flatten(p_0).view(1, 1, l).repeat(1, N, 1)

        return p_0.type(dtype)

    def _get_p(self, ks, l, c, offset, dtype, dilation):
        p_n = self._get_p_n(ks, dtype, dilation)
        p_0 = self._get_p_0(l, c, ks, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, l,  _ = q.size()
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q
        index = index.contiguous().unsqueeze(
            dim=1).expand(-1, c, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, l, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, l, N = x_offset.size()
        x_offset = torch.cat(
            [x_offset[..., s:s+ks].contiguous().view(b, c, l*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, l*ks)
        return x_offset
