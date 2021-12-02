import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mlperf_compliance import mlperf_log
from torch.nn.functional import log_softmax

import seq2seq.data.config as config
from seq2seq.models.decoder import ResidualRecurrentDecoder
from seq2seq.models.decoder import RecurrentAttention, Classifier
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.attention import BahdanauAttention
from seq2seq.models.seq2seq_base import Seq2Seq
from seq2seq.utils import gnmt_print
from seq2seq.utils import init_lstm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from collections import OrderedDict
import itertools


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.embed = torch.nn.Embedding(32317, 1024, padding_idx=0)
        self.lstm1 = nn.LSTM(1024, 1024, bidirectional=True)

        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)
        init_lstm_(self.lstm1, 0.1)

    def forward(self, inputs):
        src, src_length = inputs
        out0 = self.embed(src)
        out0 = pack_padded_sequence(out0, src_length.cpu().numpy(),
                                    batch_first=False)
        # lstm1
        out1, _ = self.lstm1(out0)
        out1, _ = pad_packed_sequence(out1, batch_first=False)
        return out1


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.lstm2 = torch.nn.LSTM(2048, 1024)
        self.lstm3 = torch.nn.LSTM(1024, 1024)
        self.lstm4 = torch.nn.LSTM(1024, 1024)

        init_lstm_(self.lstm2, 0.1)
        init_lstm_(self.lstm3, 0.1)
        init_lstm_(self.lstm4, 0.1)

    def forward(self, inputs):
        out1 = inputs
        # lstm2
        out2, _ = self.lstm2(out1)

        # lstm3
        out3, _ = self.lstm3(out2)
        out3 = out3 + out2

        # lstm4
        out4, _ = self.lstm4(out3)
        out4 = out4 + out3

        return out4


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.embed = torch.nn.Embedding(32317, 1024, padding_idx=0)
        self.lstm5 = torch.nn.LSTM(1024, 1024)
        self.lstm6 = torch.nn.LSTM(1024, 1024)

        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)
        init_lstm_(self.lstm5, 0.1)
        init_lstm_(self.lstm6, 0.1)

    def forward(self, inputs):
        out4 = inputs
        # lstm5
        out5, _ = self.lstm5(out4)
        out5 = out5 + out4

        # lstm6
        out6, _ = self.lstm6(out5)
        out6 = out6 + out5

        # out0 = self.embed(tgt_input)
        return out6


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.lstm7 = torch.nn.LSTM(1024, 1024)
        self.lstm8 = torch.nn.LSTM(1024, 1024)
        self.lstm1 = torch.nn.LSTM(1024, 1024)

        init_lstm_(self.lstm7, 0.1)
        init_lstm_(self.lstm8, 0.1)
        init_lstm_(self.lstm1, 0.1)

    def forward(self, inputs):
        out6 = inputs
        # lstm7
        out7, _ = self.lstm7(out6)
        out7 = out7 + out6

        # lstm8
        out8, _ = self.lstm8(out7)
        out8 = out8 + out7

        # lstm1
        # out1, hidden = self.lstm1(tgt_embed, None)
        return out8


class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.attn = BahdanauAttention(1024, 1024, 1024, normalize=True)
        self.lstm2 = torch.nn.LSTM(2048, 1024)
        init_lstm_(self.lstm2, 0.1)

    def forward(self, inputs):
        out1, out8, src_length, hidden, next_hidden = inputs

        self.attn.set_mask(src_length, out8)
        attn, scores = self.attn(out1, out8)

        out1 = torch.cat((out1, attn), dim=2)
        # lstm2
        out2, h = self.lstm2(out1, hidden[1])
        next_hidden.append(h)
        out = torch.cat((out2, attn), dim=2)
        return (out, hidden, scores, next_hidden)


class Stage5(torch.nn.Module):
    def __init__(self):
        super(Stage5, self).__init__()
        self.lstm3 = torch.nn.LSTM(2048, 1024)
        self.lstm4 = torch.nn.LSTM(2048, 1024)
        self.lstm5 = torch.nn.LSTM(2048, 1024)
        init_lstm_(self.lstm3, 0.1)
        init_lstm_(self.lstm4, 0.1)
        init_lstm_(self.lstm5, 0.1)

    def forward(self, inp):
        inputs, hidden, scores, next_hidden = inp
        out2, attn = inputs.chunk(2, dim=2)
        # lstm3
        out3, h = self.lstm3(inputs, hidden[2])
        next_hidden.append(h)
        out3 = out3 + out2
        out3_cat = torch.cat((out3, attn), dim=2)
        # lstm4
        out4, h = self.lstm4(out3_cat, hidden[3])
        next_hidden.append(h)
        out4 = out4 + out3
        out4_cat = torch.cat((out4, attn), dim=2)
        # lstm5
        out5, h = self.lstm5(out4_cat, hidden[4])
        next_hidden.append(h)
        out5 = out5 + out4

        out = torch.cat((out5, attn), dim=2)
        return (out, hidden, scores, next_hidden)


class Stage6(torch.nn.Module):
    def __init__(self):
        super(Stage6, self).__init__()
        self.lstm6 = torch.nn.LSTM(2048, 1024)
        self.lstm7 = torch.nn.LSTM(2048, 1024)
        self.lstm8 = torch.nn.LSTM(2048, 1024)
        init_lstm_(self.lstm6, 0.1)
        init_lstm_(self.lstm7, 0.1)
        init_lstm_(self.lstm8, 0.1)

    def forward(self, inp):
        inputs, hidden, scores, next_hidden = inp
        out5, attn = inputs.chunk(2, dim=2)
        # lstm6
        out6, h = self.lstm6(inputs, hidden[5])
        next_hidden.append(h)
        out6 = out6 + out5
        out6_cat = torch.cat((out6, attn), dim=2)
        # lstm7
        out7, h = self.lstm7(out6_cat, hidden[6])
        next_hidden.append(h)
        out7 = out7 + out6
        out7_cat = torch.cat((out7, attn), dim=2)
        # lstm8
        out8, h = self.lstm8(out7_cat, hidden[7])
        next_hidden.append(h)
        out8 = out8 + out7
        return (out8, hidden, scores, next_hidden)


class Stage7(torch.nn.Module):
    def __init__(self):
        super(Stage7, self).__init__()
        self.layer1 = Classifier(1024, 32317)

    def forward(self, inp):
        input0, hidden, scores, next_hidden = inp
        out0 = input0
        out1 = self.layer1(out0)
        return out1, hidden, scores, next_hidden


class GNMT_v2(nn.Module):
    """
    GNMT 2 stages model
    """
    def __init__(self):
        super(GNMT_v2, self).__init__()
        self.infer0 = torch.nn.Sequential(OrderedDict([
                ('Stage0', Stage0()),
                ('Stage1', Stage1()),
                ('Stage2', Stage2()),
                ('Stage3', Stage3()),
            ]))
        self.infer1 = torch.nn.Sequential(OrderedDict([
                ('Stage4', Stage4()),
                ('Stage5', Stage5()),
                ('Stage6', Stage6()),
                ('Stage7', Stage7()),
            ]))
        self.batch_first = False

    def init_hidden(self, hidden):
        """
        Converts flattened hidden state (from sequence generator) into a tuple
        of hidden states.

        :param hidden: None or flattened hidden state for decoder RNN layers
        """
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(8)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * 8
        self.next_hidden = []
        return hidden


    def encode(self, src, src_length):
        return self.infer0((src, src_length))

    def generate(self, inputs, context, beam_size):
        out8, src_length, hidden = context
        hidden = self.init_hidden(hidden)

        out0 = self.infer0.Stage2.embed(inputs)
        out1, h = self.infer0.Stage3.lstm1(out0, hidden[0])
        self.next_hidden.append(h)
        logits, hidden, scores, next_hidden = self.infer1((out1, out8, src_length,
                                                           hidden, self.next_hidden))
        hidden = torch.cat(tuple(itertools.chain(*next_hidden)))

        new_context = [out8, src_length, hidden]
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context


class GNMT_v4(nn.Module):
    """
    GNMT 2 stages model
    """
    def __init__(self):
        super(GNMT_v4, self).__init__()
        self.infer0 = torch.nn.Sequential(OrderedDict([
                ('Stage0', Stage0()),
                ('Stage1', Stage1()),
            ]))
        self.infer1 = torch.nn.Sequential(OrderedDict([
                ('Stage2', Stage2()),
                ('Stage3', Stage3()),
            ]))
        self.infer2 = torch.nn.Sequential(OrderedDict([
                ('Stage4', Stage4()),
                ('Stage5', Stage5()),
            ]))
        self.infer3 = torch.nn.Sequential(OrderedDict([
                ('Stage6', Stage6()),
                ('Stage7', Stage7()),
            ]))
        self.batch_first = False

    def init_hidden(self, hidden):
        """
        Converts flattened hidden state (from sequence generator) into a tuple
        of hidden states.

        :param hidden: None or flattened hidden state for decoder RNN layers
        """
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(8)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * 8
        self.next_hidden = []
        return hidden

    def encode(self, src, src_length):
        return self.infer1(self.infer0((src, src_length)))

    def generate(self, inputs, context, beam_size):
        out8, src_length, hidden = context
        hidden = self.init_hidden(hidden)

        out0 = self.infer1.Stage2.embed(inputs)
        out1, h = self.infer1.Stage3.lstm1(out0, hidden[0])
        self.next_hidden.append(h)
        logits, hidden, scores, next_hidden = self.infer3(
            self.infer2((out1, out8, src_length, hidden, self.next_hidden)))
        hidden = torch.cat(tuple(itertools.chain(*next_hidden)))

        new_context = [out8, src_length, hidden]
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context


class GNMT_v8(nn.Module):
    """
    GNMT 2 stages model
    """
    def __init__(self):
        super(GNMT_v8, self).__init__()
        self.infer0 = Stage0()
        self.infer1 = Stage1()
        self.infer2 = Stage2()
        self.infer3 = Stage3()
        self.infer4 = Stage4()
        self.infer5 = Stage5()
        self.infer6 = Stage6()
        self.infer7 = Stage7()
        self.batch_first = False
        self.encoder = torch.nn.Sequential(OrderedDict([
                ('infer0', self.infer0),
                ('infer1', self.infer1),
                ('infer2', self.infer2),
                ('infer3', self.infer3),
            ]))
        self.decoder = torch.nn.Sequential(OrderedDict([
                ('infer4', self.infer4),
                ('infer5', self.infer5),
                ('infer6', self.infer6),
                ('infer7', self.infer7),
            ]))

    def init_hidden(self, hidden):
        """
        Converts flattened hidden state (from sequence generator) into a tuple
        of hidden states.

        :param hidden: None or flattened hidden state for decoder RNN layers
        """
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(8)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * 8
        self.next_hidden = []
        return hidden

    def encode(self, src, src_length):
        return self.encoder((src, src_length))

    def generate(self, inputs, context, beam_size):
        out8, src_length, hidden = context
        hidden = self.init_hidden(hidden)

        out0 = self.infer2.embed(inputs)
        out1, h = self.infer3.lstm1(out0, hidden[0])
        self.next_hidden.append(h)
        logits, hidden, scores, next_hidden = self.decoder((out1, out8, src_length,
                                                           hidden, self.next_hidden))
        hidden = torch.cat(tuple(itertools.chain(*next_hidden)))

        new_context = [out8, src_length, hidden]
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context
