import torch
import torch.nn as nn
from mlperf_compliance import mlperf_log

import seq2seq.data.config as config
from seq2seq.models.decoder import ResidualRecurrentDecoder
from seq2seq.models.decoder import RecurrentAttention, Classifier
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.seq2seq_base import Seq2Seq
from seq2seq.utils import gnmt_print
from seq2seq.utils import init_lstm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import itertools


class GNMT(Seq2Seq):
    """
    GNMT v2 model
    """
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
                 batch_first=False, share_embedding=True):
        """
        Constructor for the GNMT v2 model.

        :param vocab_size: size of vocabulary (number of tokens)
        :param hidden_size: internal hidden size of the model
        :param num_layers: number of layers, applies to both encoder and
            decoder
        :param dropout: probability of dropout (in encoder and decoder)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param share_embedding: if True embeddings are shared between encoder
            and decoder
        """

        super(GNMT, self).__init__(batch_first=batch_first)

        gnmt_print(key=mlperf_log.MODEL_HP_NUM_LAYERS,
                   value=num_layers, sync=False)
        gnmt_print(key=mlperf_log.MODEL_HP_HIDDEN_SIZE,
                   value=hidden_size, sync=False)
        gnmt_print(key=mlperf_log.MODEL_HP_DROPOUT,
                   value=dropout, sync=False)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size,
                                    padding_idx=config.PAD)
            nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.embed = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.lstm1 = nn.LSTM(1024, 1024, bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.lstm2 = torch.nn.LSTM(2048, 1024)

        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)
        init_lstm_(self.lstm1, 0.1)
        init_lstm_(self.lstm2, 0.1)

    def forward(self, input):
        src, src_length, tgt_input = input
        out0 = self.embed(src)
        out0 = pack_padded_sequence(self.dropout(out0), src_length.cpu().numpy(),
                                    batch_first=False)
        # lstm1
        out1, _ = self.lstm1(out0)
        out1, _ = pad_packed_sequence(out1, batch_first=False)

        # lstm2
        out1_drop = self.dropout(out1)
        out2, _ = self.lstm2(out1_drop)
        out2_drop = self.dropout(out2)

        shape = torch.zeros(1, out2.shape[1], 1024).cuda()
        shape[0][0][0] = 1 + out2_drop.shape[0] + out2.shape[0] + 1 + tgt_input.shape[0]
        shape[0][0][1] = out2_drop.shape[0] + 1
        shape[0][0][2] = out2.shape[0] + shape[0][0][1]
        shape[0][0][3] = 1 + shape[0][0][2]
        shape[0][0][4] = tgt_input.shape[0] + shape[0][0][3]
        src_length = torch.transpose(src_length.expand
                                     (1, 1024, src_length.shape[0]), 2, 1)
        tgt_input = tgt_input.expand(1024, tgt_input.shape[0], tgt_input.shape[1])
        tgt_input = torch.transpose(torch.transpose(tgt_input, 1, 0), 2, 1)
        out = torch.cat((shape, out2_drop, out2, src_length, tgt_input), dim=0)
        return out


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.lstm3 = torch.nn.LSTM(1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.lstm4 = torch.nn.LSTM(1024, 1024)
        self.lstm5 = torch.nn.LSTM(1024, 1024)

        init_lstm_(self.lstm3, 0.1)
        init_lstm_(self.lstm4, 0.1)
        init_lstm_(self.lstm5, 0.1)

    def forward(self, input):
        shape = input[0:1, :, :].int()
        out2_drop = input[1:shape[0][0][1], :, :]
        out2 = input[shape[0][0][1]:shape[0][0][2], :, :]
        src_tgt = input[shape[0][0][2]:shape[0][0][0], :, :]
        # lstm3
        out3, _ = self.lstm3(out2_drop)
        out3 = out3 + out2

        # lstm4
        out3_drop = self.dropout(out3)
        out4, _ = self.lstm4(out3_drop)
        out4 = out4 + out3

        # lstm5
        out4_dropout = self.dropout(out4)
        out5, _ = self.lstm5(out4_dropout)
        out5 = out5 + out4

        out5_drop = self.dropout(out5)

        shape = torch.zeros(1, out5.shape[1], 1024).cuda()
        shape[0][0][0] = 1 + out5_drop.shape[0] + out5.shape[0] + src_tgt.shape[0]
        shape[0][0][1] = out5_drop.shape[0] + 1
        shape[0][0][2] = out5.shape[0] + shape[0][0][1]
        shape[0][0][3] = 1 + shape[0][0][2]
        shape[0][0][4] = src_tgt.shape[0] + shape[0][0][2]
        out = torch.cat((shape, out5_drop, out5, src_tgt), dim=0)
        return out


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.embed = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.lstm6 = torch.nn.LSTM(1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.lstm7 = torch.nn.LSTM(1024, 1024)
        self.lstm8 = torch.nn.LSTM(1024, 1024)

        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)
        init_lstm_(self.lstm6, 0.1)
        init_lstm_(self.lstm7, 0.1)
        init_lstm_(self.lstm8, 0.1)

    def forward(self, input):
        shape = input[0:1, :, :].int()
        out5_drop = input[1:shape[0][0][1], :, :]
        out5 = input[shape[0][0][1]:shape[0][0][2], :, :]
        src_length = input[shape[0][0][2]:shape[0][0][3], :, :]
        tgt_input = input[shape[0][0][3]:shape[0][0][4], :, :][:, :, 0].long()

        # lstm6
        out6, _ = self.lstm6(out5_drop)
        out6 = out6 + out5

        # lstm7
        out6_drop = self.dropout(out6)
        out7, _ = self.lstm7(out6_drop)
        out7 = out7 + out6

        # lstm8
        out7_drop = self.dropout(out7)
        out8, _ = self.lstm8(out7_drop)
        out8 = out8 + out7

        out0 = self.embed(tgt_input)
        # out = torch.cat((out8, out0), dim=0)

        # shape: out.shape[0] - tgt_input, tgt_input
        shape = torch.zeros(1, out8.shape[1], 1024).cuda()
        shape[0][0][0] = 1 + out8.shape[0] + out0.shape[0] + 1
        shape[0][0][1] = out8.shape[0] + 1
        shape[0][0][2] = out0.shape[0] + shape[0][0][1]
        shape[0][0][3] = 1 + shape[0][0][2]
        out = torch.cat((shape, out8, out0, src_length), dim=0)
        return out


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.attn_model = RecurrentAttention(1024, 1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.lstm2 = torch.nn.LSTM(2048, 1024)
        init_lstm_(self.lstm2, 0.1)

    def forward(self, input):
        shape = input[0:1, :, :].int()
        out8 = input[1:shape[0][0][1], :, :]
        tgt_embed = input[shape[0][0][1]:shape[0][0][2], :, :]
        src_length = input[shape[0][0][2]:shape[0][0][3], :, :][0, :, 0].long()

        # attention
        out1, h, attn, scores = self.attn_model(tgt_embed, None, out8, src_length)
        out1_drop = self.dropout(out1)
        out1 = torch.cat((out1_drop, attn), dim=2)

        # lstm2
        out2, _ = self.lstm2(out1, None)
        out = torch.cat((out2, attn), dim=2)
        return out


class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.lstm3 = torch.nn.LSTM(2048, 1024)
        self.lstm4 = torch.nn.LSTM(2048, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        init_lstm_(self.lstm3, 0.1)
        init_lstm_(self.lstm4, 0.1)

    def forward(self, input):
        out2, attn = input.chunk(2, dim=2)
        # out2: out2
        # lstm3
        out2_drop = self.dropout(out2)
        out3, _ = self.lstm3(torch.cat((out2_drop, attn), dim=2), None)
        out3 = out3 + out2

        # lstm4
        out3_drop = self.dropout(out3)
        out4, _ = self.lstm4(torch.cat((out3_drop, attn), dim=2), None)
        out4 = out4 + out3

        out = torch.cat((out4, attn), dim=2)
        return out


class Stage5(torch.nn.Module):
    def __init__(self):
        super(Stage5, self).__init__()
        self.lstm5 = torch.nn.LSTM(2048, 1024)
        self.lstm6 = torch.nn.LSTM(2048, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        init_lstm_(self.lstm5, 0.1)
        init_lstm_(self.lstm6, 0.1)

    def forward(self, input):
        out4, attn = input.chunk(2, dim=2)
        # lstm5
        out4_drop = self.dropout(out4)
        out5, _ = self.lstm5(torch.cat((out4_drop, attn), dim=2), None)
        out5 = out5 + out4

        # lstm6
        out5_drop = self.dropout(out5)
        out6, _ = self.lstm6(torch.cat((out5_drop, attn), dim=2), None)
        out6 = out6 + out5

        out = torch.cat((out6, attn), dim=2)
        return out


class Stage6(torch.nn.Module):
    def __init__(self):
        super(Stage6, self).__init__()
        self.lstm7 = torch.nn.LSTM(2048, 1024)
        self.lstm8 = torch.nn.LSTM(2048, 1024)
        self.dropout = torch.nn.Dropout(p=0.2)
        init_lstm_(self.lstm7, 0.1)
        init_lstm_(self.lstm8, 0.1)

    def forward(self, input):
        out6, attn = input.chunk(2, dim=2)
        # lstm7
        out6_drop = self.dropout(out6)
        out7, _ = self.lstm7(torch.cat((out6_drop, attn), dim=2), None)
        out7 = out7 + out6

        # lstm8
        out7_drop = self.dropout(out7)
        out8, _ = self.lstm8(torch.cat((out7_drop, attn), dim=2), None)
        out8 = out8 + out7
        return out8


class Stage7(torch.nn.Module):
    def __init__(self):
        super(Stage7, self).__init__()
        self.layer1 = Classifier(1024, 32320)

    def forward(self, input0):
        out0 = input0
        out1 = self.layer1(out0)
        return out1
