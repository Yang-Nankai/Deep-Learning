# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch
import math


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = (self.linear(out[-1]))
        out = self.softmax(out)
        return out


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myLSTM, self).__init__()
        self.lstm = CustomLSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = (self.linear(out[-1]))
        out = self.softmax(out)
        return out


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        """input gate"""
        self.linear_i_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_i_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_i_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.i_sigmod = nn.Sigmoid()

        """forget gate"""
        self.linear_f_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_f_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_f_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.f_sigmod = nn.Sigmoid()

        """cell memeory"""
        self.linear_c_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_c_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_tanh = nn.Tanh()

        """output gate"""
        self.linear_o_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_o_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_o_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o_sigmod = nn.Sigmoid()

        """hidden memory"""
        self.h_tanh = nn.Tanh()

    def input_gate(self, x, h, c):
        return self.i_sigmod(self.linear_i_x(x) + self.linear_i_h(h) + self.linear_i_c(c))

    def forget_gate(self, x, h, c):
        return self.f_sigmod(self.linear_f_x(x) + self.linear_f_h(h) + self.linear_f_c(c))

    def cell_memory(self, i, f, x, h, c):
        return f * c + i * self.c_tanh(self.linear_c_x(x) + self.linear_c_h(h))

    def output_gate(self, x, h, c_next):
        o = self.o_sigmod(self.linear_o_x(
            x) + self.linear_o_h(h) + self.linear_o_c(c_next))
        return o * self.h_tanh(c_next)

    def hidden_memory(self, c_next, o):
        return o * self.h_tanh(c_next)

    def init_hidden_cell(self, x):
        """initial hidden and cell"""
        h_0 = x.data.new(x.size(0), self.hidden_dim).zero_()
        c_0 = x.data.new(x.size(0), self.hidden_dim).zero_()
        return (h_0, c_0)

    def forward(self, x, memory=None):
        _, seq_sz, _ = x.size()
        hidden_seq = []
        if memory is not None:
            h, c = memory
        else:
            h, c = self.init_hidden_cell(x)
        for t in range(seq_sz):
            x_t = x[:, t, :]
            i = self.input_gate(x_t, h, c)  # (x.size(0), hidden_dim)
            f = self.forget_gate(x_t, h, c)
            c = self.cell_memory(i, f, x_t, h, c)
            o = self.output_gate(x_t, h, c)
            h = self.hidden_memory(c, o)
            hidden_seq.append(h.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h, c)
