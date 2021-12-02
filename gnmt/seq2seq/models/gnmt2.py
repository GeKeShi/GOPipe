import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mlperf_compliance import mlperf_log

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
import itertools


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.embed = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.lstm1 = nn.LSTM(1024, 1024, bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)

        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)
        init_lstm_(self.lstm1, 0.1)

    def forward(self, input):
        src, src_length, tgt_input = input
        out0 = self.embed(src)
        out0 = pack_padded_sequence(self.dropout(out0), src_length.cpu().numpy(),
                                    batch_first=False)
        # lstm1
        out1, _ = self.lstm1(out0)
        out1, _ = pad_packed_sequence(out1, batch_first=False)
        out1_drop = self.dropout(out1)

        shape = torch.zeros(1, out1_drop.shape[1], 2048).cuda()
        shape[0][0][0] = 1 + out1_drop.shape[0] + 1 + tgt_input.shape[0]
        shape[0][0][1] = out1_drop.shape[0] + 1
        shape[0][0][2] = 1 + shape[0][0][1]
        shape[0][0][3] = tgt_input.shape[0] + shape[0][0][2]
        src_length = torch.transpose(src_length.expand
                                     (1, 2048, src_length.shape[0]), 2, 1)
        tgt_input = tgt_input.expand(2048, tgt_input.shape[0], tgt_input.shape[1])
        tgt_input = torch.transpose(torch.transpose(tgt_input, 1, 0), 2, 1)
        out = torch.cat((shape, out1_drop, src_length, tgt_input), dim=0)
        return out


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.lstm2 = torch.nn.LSTM(2048, 1024)
        self.lstm3 = torch.nn.LSTM(1024, 1024)
        self.lstm4 = torch.nn.LSTM(1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)

        init_lstm_(self.lstm2, 0.1)
        init_lstm_(self.lstm3, 0.1)
        init_lstm_(self.lstm4, 0.1)

    def forward(self, input):
        shape = input[0:1, :, :].int()
        out1_drop = input[1:shape[0][0][1], :, :]
        src_tgt = input[shape[0][0][1]:shape[0][0][3], :, 0:1024]
        # lstm2
        out2, _ = self.lstm2(out1_drop)
        out2_drop = self.dropout(out2)

        # lstm3
        out3, _ = self.lstm3(out2_drop)
        out3 = out3 + out2

        # lstm4
        out3_drop = self.dropout(out3)
        out4, _ = self.lstm4(out3_drop)
        out4 = out4 + out3

        shape = torch.zeros(1, out4.shape[1], 1024).cuda()
        shape[0][0][0] = 1 + out4.shape[0] + src_tgt.shape[0]
        shape[0][0][1] = out4.shape[0] + 1
        shape[0][0][2] = 1 + shape[0][0][1]
        shape[0][0][3] = src_tgt.shape[0] + shape[0][0][2] - 1
        out = torch.cat((shape, out4, src_tgt), dim=0)
        return out


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.embed = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.lstm5 = torch.nn.LSTM(1024, 1024)
        self.lstm6 = torch.nn.LSTM(1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        init_lstm_(self.lstm5, 0.1)
        init_lstm_(self.lstm6, 0.1)

    def forward(self, input):
        shape = input[0:1, :, :].int()
        out4 = input[1:shape[0][0][1], :, :]
        src_length = input[shape[0][0][1]:shape[0][0][2], :, :]
        tgt_input = input[shape[0][0][2]:shape[0][0][3], :, :][:, :, 0].long()

        out4_drop = self.dropout(out4)

        # lstm5
        out5, _ = self.lstm5(out4_drop)
        out5 = out5 + out4
        out5_drop = self.dropout(out5)

        # lstm6
        out6, _ = self.lstm6(out5_drop)
        out6 = out6 + out5

        out0 = self.embed(tgt_input)

        shape = torch.zeros(1, out6.shape[1], 1024).cuda()
        shape[0][0][0] = 1 + out6.shape[0] + out0.shape[0] + 1
        shape[0][0][1] = out6.shape[0] + 1
        shape[0][0][2] = out0.shape[0] + shape[0][0][1]
        shape[0][0][3] = 1 + shape[0][0][2]
        out = torch.cat((shape, out6, out0, src_length), dim=0)
        return out


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.lstm7 = torch.nn.LSTM(1024, 1024)
        self.lstm8 = torch.nn.LSTM(1024, 1024)
        self.lstm1 = torch.nn.LSTM(1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)

        init_lstm_(self.lstm7, 0.1)
        init_lstm_(self.lstm8, 0.1)
        init_lstm_(self.lstm1, 0.1)

    def forward(self, input):
        shape = input[0:1, :, :].int()
        out6 = input[1:shape[0][0][1], :, :]
        tgt_embed = input[shape[0][0][1]:shape[0][0][2], :, :]
        src_length = input[shape[0][0][2]:shape[0][0][3], :, :]

        out6_drop = self.dropout(out6)
        # lstm7
        out7, _ = self.lstm7(out6_drop)
        out7 = out7 + out6
        out7_drop = self.dropout(out7)

        # lstm8
        out8, _ = self.lstm8(out7_drop)
        out8 = out8 + out7

        # lstm1
        out1, hidden = self.lstm1(tgt_embed, None)

        shape = torch.zeros(1, out1.shape[1], 1024).cuda()
        shape[0][0][0] = 1 + out8.shape[0] + out1.shape[0] + 1
        shape[0][0][1] = out8.shape[0] + 1
        shape[0][0][2] = out1.shape[0] + shape[0][0][1]
        shape[0][0][3] = 1 + shape[0][0][2]

        out = torch.cat((shape, out8, out1, src_length), dim=0)
        return out


class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.attn = BahdanauAttention(1024, 1024, 1024, normalize=True)
        self.lstm2 = torch.nn.LSTM(2048, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        init_lstm_(self.lstm2, 0.1)

    def attn_cal(self, out1_drop, out8, src_length):
        # out1_drop, out8 = inputs
        self.attn.set_mask(src_length, out8)
        attn, scores = self.attn(out1_drop, out8)
        return attn

    def forward(self, input):
        shape = input[0:1, :, :].int()
        out8 = input[1:shape[0][0][1], :, :]
        out1 = input[shape[0][0][1]:shape[0][0][2], :, :]
        src_length = input[shape[0][0][2]:shape[0][0][3], :, :][0, :, 0].long()

        # self.attn.set_mask(src_length, out8)
        # attn, scores = self.attn(out1, out8)

        out1_drop = self.dropout(out1)
        # attn = self.attn_cal((out1_drop, out8))
        attn = checkpoint(self.attn_cal, out1_drop, out8, src_length)

        out1 = torch.cat((out1_drop, attn), dim=2)
        out1_drop = self.dropout(out1)

        # lstm2
        out2, _ = self.lstm2(out1_drop, None)
        out = torch.cat((out2, attn), dim=2)
        return out


class Stage5(torch.nn.Module):
    def __init__(self):
        super(Stage5, self).__init__()
        self.lstm3 = torch.nn.LSTM(2048, 1024)
        self.lstm4 = torch.nn.LSTM(2048, 1024)
        self.lstm5 = torch.nn.LSTM(2048, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        init_lstm_(self.lstm3, 0.1)
        init_lstm_(self.lstm4, 0.1)
        init_lstm_(self.lstm5, 0.1)

    def forward(self, input):
        out2, attn = input.chunk(2, dim=2)
        out2_drop = self.dropout(input)
        # lstm3
        out3, _ = self.lstm3(out2_drop, None)
        out3 = out3 + out2
        out3_drop = self.dropout(torch.cat((out3, attn), dim=2))
        # lstm4
        out4, _ = self.lstm4(out3_drop, None)
        out4 = out4 + out3
        out4_drop = self.dropout(torch.cat((out4, attn), dim=2))
        # lstm5
        out5, _ = self.lstm5(out4_drop, None)
        out5 = out5 + out4

        out = torch.cat((out5, attn), dim=2)
        return out

        # out2, attn = input.chunk(2, dim=2)
        # out2_drop = self.dropout(out2)
        # # lstm3
        # out3, _ = self.lstm3(torch.cat((out2_drop, attn), dim=2), None)
        # out3 = out3 + out2
        # out3_drop = self.dropout(out3)
        #
        # # lstm4
        # out4, _ = self.lstm4(torch.cat((out3_drop, attn), dim=2), None)
        # out4 = out4 + out3
        # out4_drop = self.dropout(out4)
        #
        # # lstm5
        # out5, _ = self.lstm5(torch.cat((out4_drop, attn), dim=2), None)
        # out5 = out5 + out4
        #
        # out = torch.cat((out5, attn), dim=2)
        # return out


class Stage6(torch.nn.Module):
    def __init__(self):
        super(Stage6, self).__init__()
        self.lstm6 = torch.nn.LSTM(2048, 1024)
        self.lstm7 = torch.nn.LSTM(2048, 1024)
        self.lstm8 = torch.nn.LSTM(2048, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        init_lstm_(self.lstm6, 0.1)
        init_lstm_(self.lstm7, 0.1)
        init_lstm_(self.lstm8, 0.1)

    def forward(self, input):
        out5, attn = input.chunk(2, dim=2)
        out5_drop = self.dropout(input)
        # lstm6
        out6, _ = self.lstm6(out5_drop, None)
        out6 = out6 + out5
        out6_drop = self.dropout(torch.cat((out6, attn), dim=2))
        # lstm7
        out7, _ = self.lstm7(out6_drop, None)
        out7 = out7 + out6
        out7_drop = self.dropout(torch.cat((out7, attn), dim=2))
        # lstm8
        out8, _ = self.lstm8(out7_drop, None)
        out8 = out8 + out7
        return out8

        # out5, attn = input.chunk(2, dim=2)
        # out5_drop = self.dropout(out5)
        # # lstm6
        # out6, _ = self.lstm6(torch.cat((out5_drop, attn), dim=2), None)
        # out6 = out6 + out5
        # out6_drop = self.dropout(out6)
        #
        # # lstm7
        # out7, _ = self.lstm7(torch.cat((out6_drop, attn), dim=2), None)
        # out7 = out7 + out6
        # out7_drop = self.dropout(out7)
        #
        # # lstm8
        # out8, _ = self.lstm8(torch.cat((out7_drop, attn), dim=2), None)
        # out8 = out8 + out7
        # return out8


class Stage7(torch.nn.Module):
    def __init__(self):
        super(Stage7, self).__init__()
        self.layer1 = Classifier(1024, 32320)

    def forward(self, input0):
        out0 = input0
        out1 = self.layer1(out0)
        return out1
