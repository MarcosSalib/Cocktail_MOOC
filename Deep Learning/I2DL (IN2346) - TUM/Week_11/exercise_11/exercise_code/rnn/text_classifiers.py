import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import pytorch_lightning as pl

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

# from .rnn_nn import Embedding, RNN, LSTM
from torch.nn import Embedding, RNN, LSTM

from exercise_code.rnn.sentiment_dataset import (
    download_data,
    load_sentiment_data,
    load_vocab,
    SentimentDataset,
    collate
)


class RNNClassifier(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, hidden_size,
                 use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        self.hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }

        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################

        self.embedding = Embedding(self.hparams['num_embeddings'], self.hparams['embedding_dim'], 0)
        self.rnn = (LSTM if use_lstm else RNN)(self.hparams['embedding_dim'], self.hparams['hidden_size'])
        self.fc = nn.Linear(self.hparams['hidden_size'], 1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lenghts=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        embeds = self.embedding(sequence) # seq_len, batch_size, embedding_dim

        if lenghts is not None:
            embeds = pack_padded_sequence(embeds, lenghts)

        h_seq, h = self.rnn(embeds)
        if isinstance(h, tuple):
            h = h[0]

        # h_seq: seq_len, batch_size, hidden_size
        # h: 1, batch_size, hidden_size
        output = self.fc(h.squeeze(0)).sigmoid().view(-1)

        # return output, h

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output


    def general_step(self, batch, batch_idx, mode):
        # batch: data, label, lengths

        data = batch['data']
        label = batch['label']
        lengths = batch['lengths']
        out = self.forward(data)

        loss_fn = nn.BCELoss(reduction='mean')
        loss = loss_fn(out, label)
        return loss

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode+'_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'train')
        return {'loss': loss}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters())
        return optim
