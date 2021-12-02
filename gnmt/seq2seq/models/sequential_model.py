import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mlperf_compliance import mlperf_log

import math
import torch.nn.functional as F
import seq2seq.data.config as config
from seq2seq.models.decoder import ResidualRecurrentDecoder
from seq2seq.models.decoder import RecurrentAttention, Classifier
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.seq2seq_base import Seq2Seq
from seq2seq.utils import gnmt_print
from seq2seq.utils import init_lstm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter
import itertools


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)


class embed1(nn.Module):
    def __init__(self):
        super(embed1, self).__init__()
        self.embed = nn.Embedding(32320, 1024, padding_idx=0)
        self.dropout = torch.nn.Dropout(p=0.2)
        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)

    def forward(self, inputs):
        src, src_length, tgt_input = inputs
        out = self.embed(src)
        out = self.dropout(out)
        return (out, src_length, tgt_input)


class embed2(nn.Module):
    def __init__(self):
        super(embed2, self).__init__()
        self.embed = nn.Embedding(32320, 1024, padding_idx=0)
        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)

    def forward(self, inputs):
        inp, src_length, tgt_input = inputs
        tgt_out = self.embed(tgt_input)
        return (inp, src_length, tgt_out)


class dropout1(nn.Module):
    def __init__(self):
        super(dropout1, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, inputs):
        inp, src_length, tgt_input = inputs
        inp_dropout = self.dropout(inp)
        return (inp, inp_dropout, src_length, tgt_input)


class dropout2(nn.Module):
    def __init__(self):
        super(dropout2, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, inputs):
        inp, src_length, tgt_input = inputs
        tgt_input = self.dropout(tgt_input)
        return (inp, src_length, tgt_input)


class dropout3(nn.Module):
    def __init__(self):
        super(dropout3, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, inputs):
        inp, attn, inp_cat = inputs
        inp_cat = self.dropout(inp_cat)
        return (inp, attn, inp_cat)


class lstm1(nn.Module):
    def __init__(self, bi=False, residual=True, size=1024):
        super(lstm1, self).__init__()
        self.pad = bi
        self.residual = residual
        self.layer = nn.LSTM(size, 1024, bidirectional=bi)
        init_lstm_(self.layer, 0.1)

    def forward(self, inputs):
        if self.pad:
            inp, src_length, tgt_input = inputs
            out = pack_padded_sequence(inp, src_length.cpu().numpy(), batch_first=False)
            out, _ = self.layer(out)
            out, _ = pad_packed_sequence(out, batch_first=False)
        else:
            inp, inp_dropout, src_length, tgt_input = inputs
            out, _ = self.layer(inp_dropout)
        if self.residual:
            out = out + inp
        return (out, src_length, tgt_input)


class lstm2(nn.Module):
    def __init__(self, bi=False, size=1024):
        super(lstm2, self).__init__()
        self.layer = nn.LSTM(size, 1024, bidirectional=bi, batch_first=False)
        init_lstm_(self.layer, 0.1)

    def forward(self, inputs):
        out, src_length, tgt_input = inputs
        tgt_out, _ = self.layer(tgt_input)
        return (tgt_out, out, src_length)


class lstm3(nn.Module):
    def __init__(self, bi=False, residual=True, last=False, size=2048):
        super(lstm3, self).__init__()
        self.residual = residual
        self.last = last
        self.layer = nn.LSTM(size, 1024, bidirectional=bi)
        init_lstm_(self.layer, 0.1)

    def forward(self, inputs):
        inp, attn, inp_cat = inputs
        out, _ = self.layer(inp_cat)
        if self.residual:
            out = out + inp
        if self.last:
            return tuple([out])
        else:
            out_cat = torch.cat((out, attn), dim=2)
            return (out, attn, out_cat)


class Attention_(nn.Module):
    """
    Bahdanau Attention (https://arxiv.org/abs/1409.0473)
    Implementation is very similar to tf.contrib.seq2seq.BahdanauAttention
    """
    def __init__(self, query_size=1024, key_size=1024, num_units=1024, normalize=True,
                 batch_first=False, init_weight=0.1):
        """
        Constructor for the BahdanauAttention.

        :param query_size: feature dimension for query
        :param key_size: feature dimension for keys
        :param num_units: internal feature dimension
        :param normalize: whether to normalize energy term
        :param batch_first: if True batch size is the 1st dimension, if False
            the sequence is first and batch size is second
        :param init_weight: range for uniform initializer used to initialize
            Linear key and query transform layers and linear_att vector
        """
        super(Attention_, self).__init__()

        self.normalize = normalize
        self.batch_first = batch_first
        self.num_units = num_units

        self.linear_q = nn.Linear(query_size, num_units, bias=False)
        self.linear_k = nn.Linear(key_size, num_units, bias=False)
        nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)

        self.linear_att = Parameter(torch.Tensor(num_units))

        self.mask = None

        if self.normalize:
            self.normalize_scalar = Parameter(torch.Tensor(1))
            self.normalize_bias = Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter('normalize_scalar', None)
            self.register_parameter('normalize_bias', None)

        self.reset_parameters(init_weight)

    def reset_parameters(self, init_weight):
        """
        Sets initial random values for trainable parameters.
        """
        stdv = 1. / math.sqrt(self.num_units)
        self.linear_att.data.uniform_(-init_weight, init_weight)

        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        """
        sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields

        :param context_len: b
        :param context: if batch_first: (b x t_k x n) else: (t_k x b x n)

        self.mask: (b x t_k)
        """

        if self.batch_first:
            max_len = context.size(1)
        else:
            max_len = context.size(0)

        indices = torch.arange(0, max_len, dtype=torch.int64,
                               device=context.device)
        self.mask = indices >= (context_len.unsqueeze(1))

    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n

        returns: b x t_q x t_k scores
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys

        if self.normalize:
            sum_qk = sum_qk + self.normalize_bias
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
        else:
            linear_att = self.linear_att

        out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    # def forward(self, inputs):
    def forward(self, query, keys):
        """

        :param query: if batch_first: (b x t_q x n) else: (t_q x b x n)
        :param keys: if batch_first: (b x t_k x n) else (t_k x b x n)

        :returns: (context, scores_normalized)
        context: if batch_first: (b x t_q x n) else (t_q x b x n)
        scores_normalized: if batch_first (b x t_q x t_k) else (t_q x b x t_k)
        """

        # query, keys, context_len = inputs
        query0 = query
        # self.set_mask(context_len, keys)
        # first dim of keys and query has to be 'batch', it's needed for bmm
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)

        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False

        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)

        # FC layers to transform query and key
        processed_query = self.linear_q(query)
        processed_key = self.linear_k(keys)

        # scores: (b x t_q x t_k)
        scores = self.calc_score(processed_query, processed_key)

        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            # I can't use -INF because of overflow check in pytorch
            scores.data.masked_fill_(mask, -65504.0)

        # Normalize the scores, softmax over t_k
        scores_normalized = F.softmax(scores, dim=-1)

        # Calculate the weighted average of the attention inputs according to
        # the scores
        # context: (b x t_q x n)
        context = torch.bmm(scores_normalized, keys)

        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)
        return (query0, context, torch.cat((query0, context), dim=2))


class Attention(nn.Module):
    def __init__(self, recompute=False):
        super(Attention, self).__init__()
        self.recompute = recompute
        self.attention = Attention_()

    def att_cal(self, query, keys, context_len):
        self.attention.set_mask(context_len, keys)
        out = self.attention(query, keys)
        return out

    def forward(self, inputs):
        query, keys, context_len = inputs
        if self.recompute:
            out = checkpoint(self.att_cal, query, keys, context_len)
        else:
            self.attention.set_mask(context_len, keys)
            out = self.attention(query, keys)
        return out


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, init_weight=0.1):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, inputs):
        x = inputs[0]
        out = self.classifier(x)
        return out


class Conv2d(nn.Module):
    def __init__(self, size1, size2, recompute=False):
        super(Conv2d, self).__init__()
        self.recompute = recompute
        self.layer = nn.Conv2d(size1, size2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        _initialize_weights(self.layer)

    def com_forward(self, inputs):
        re = self.relu(self.layer(inputs))
        # re = self.layer(inputs)
        return re

    def forward(self, inputs):
        if self.recompute:
            re = checkpoint(self.com_forward, inputs)
        else:
            re = self.com_forward(inputs)
        return re


class Pool2d(nn.Module):
    def __init__(self, recompute=False):
        super(Pool2d, self).__init__()
        self.recompute = recompute
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def com_forward(self, inputs):
        re = self.layer(inputs)
        return re

    def forward(self, inputs):
        if self.recompute:
            re = checkpoint(self.com_forward, inputs)
        else:
            re = self.com_forward(inputs)
        return re


class Linear(nn.Module):
    def __init__(self, size1, size2, recompute=False, R=True):
        super(Linear, self).__init__()
        self.recompute = recompute
        self.layer = nn.Linear(size1, size2)
        self.relu = nn.ReLU(inplace=True)
        self.R = R
        _initialize_weights(self.layer)

    def com_forward(self, inputs):
        if self.R:
            re = self.relu(self.layer(inputs))
        else:
            re = self.layer(inputs)
        return re

    def forward(self, inputs):
        if self.recompute:
            re = checkpoint(self.com_forward, inputs)
        else:
            re = self.com_forward(inputs)
        return re


class Flatten(nn.Module):
    def forward(self, inputs):
        return torch.flatten(inputs, start_dim=1)
