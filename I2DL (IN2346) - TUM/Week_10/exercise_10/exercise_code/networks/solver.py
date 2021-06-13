from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print('START TRAIN.')

        log_count = log_nth
        for epoch in range(1, num_epochs + 1):

            model.train()
            for i, batch in enumerate(train_loader):
                data, target = batch
                date, target = data.to(device), target.to(device)

                optim.zero_grad()
                output = model.forward(data)

                # calculate the batch loss
                loss = 0
                for s in range(data.size()[0]):
                    loss += self.loss_func(output[s].view(23, -1).transpose(1, 0),
                                           target[s].view(-1))
                # loss = self.loss_func(output, target)
                loss.backward()
                optim.step()

                train_loss = loss.item()
                self.train_loss_history.append(train_loss)

                if log_count == 0:
                    print("[Iteration %d/%d] TRAIN loss: %.3f" % (i, iter_per_epoch, train_loss))
                    log_count = log_nth

            # Train acc / loss (last batch)
            pred = torch.argmax(output, dim=1)
            train_acc = torch.sum(pred == target).item() / train_loader.batch_size
            self.train_acc_history.append(train_acc)

            print("[Epoch %d/%d] TRAIN acc/loss: %.6f/%.6f" % (epoch, num_epochs, train_acc, train_loss))

        print('FINISH.')