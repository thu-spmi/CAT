# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Keyu An

"""CUSIDE simu net impl
"""
import torch
from torch import nn


class Prenet(nn.Module):
    """Prenet is a multi-layer fully-connected network with ReLU activations.
    During training and testing (i.e., feature extraction), each input frame is
    passed into the Prenet, and the Prenet output is then fed to the RNN. If
    Prenet configuration is None, the input frames will be directly fed to the
    RNN without any transformation.
    """

    def __init__(self, input_size, num_layers, hidden_size, dropout):
        super(Prenet, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.Linear(in_features=in_size, out_features=out_size)
             for (in_size, out_size) in zip(input_sizes, output_sizes)])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, mel_dim)
        for layer in self.layers:
            inputs = self.dropout(self.relu(layer(inputs)))

        return inputs
        # inputs: (batch_size, seq_len, out_dim)


class Postnet(nn.Module):
    """Postnet is a simple linear layer for predicting the target frames given
    the RNN context during training. We don't need the Postnet for feature
    extraction.
    """

    def __init__(self, input_size, out_len, output_size=80):
        super(Postnet, self).__init__()
        self.layer = nn.Linear(in_features=input_size,
                               out_features=out_len*output_size)
        self.out_len = out_len
        self.output_size = output_size

    def forward(self, inputs):
        # inputs: (batch_size, hidden, 1) -- for conv1d operation
        return self.layer(inputs).reshape(-1, self.out_len, self.output_size)
        # (batch_size, out_len, output_size) -- back to the original shape


class SimuNet(nn.Module):
    """This class defines Autoregressive Predictive Coding (APC), a model that
    learns to extract general speech features from unlabeled speech data. These
    features are shown to contain rich speaker and phone information, and are
    useful for a wide range of downstream tasks such as speaker verification
    and phone classification.
    An APC model consists of a Prenet (optional), a multi-layer GRU network,
    and a Postnet. For each time step during training, the Prenet transforms
    the input frame into a latent representation, which is then consumed by
    the GRU network for generating internal representations across the layers.
    Finally, the Postnet takes the output of the last GRU layer and attempts to
    predict the target frame.
    After training, to extract features from the data of your interest, which
    do not have to be i.i.d. with the training data, simply feed-forward the
    the data through the APC model, and take the the internal representations
    (i.e., the GRU hidden states) as the extracted features and use them in
    your tasks.
    """

    def __init__(self, mel_dim: int, out_len: int, hdim: int, rnn_num_layers: int, rnn_dropout: float = 0.1, rnn_residual: bool = True):
        super(SimuNet, self).__init__()
        self.mel_dim = mel_dim
        # Make sure the dimensionalities are correct
        in_sizes = [mel_dim] + [hdim] * (rnn_num_layers-1)
        out_sizes = [hdim] * rnn_num_layers
        self.rnns = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.rnn_residual = rnn_residual

        self.postnet = Postnet(
            input_size=hdim,
            out_len=out_len,
            output_size=self.mel_dim)

    def forward(self, inputs, chunk_size):
        """Forward function for both training and testing (feature extraction).
        input:
        inputs: (batch_size, seq_len, mel_dim)
        return:
        predicted_mel: (batch_size, seq_len, mel_dim)
        """

        rnn_inputs = inputs
        for i, layer in enumerate(self.rnns):
            rnn_outputs, h = layer(rnn_inputs)

            if i + 1 < len(self.rnns):
                # apply dropout except the last rnn layer
                rnn_outputs = self.rnn_dropout(rnn_outputs)

            if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                # Residual connections
                rnn_outputs = rnn_outputs + rnn_inputs

            rnn_inputs = rnn_outputs

        B = rnn_outputs.size(0)
        L = rnn_outputs.size(1)
        rnn_outputs = rnn_outputs.contiguous().view(B*L//chunk_size, chunk_size, -1)
        rnn_outputs = rnn_outputs[:, -1, :]
        predicted_mel = self.postnet(rnn_outputs.squeeze(1))
        return predicted_mel
