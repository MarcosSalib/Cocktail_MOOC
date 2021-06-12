"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5), #32,92,92
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #32,46,46
            nn.Dropout(p=0.1),

            nn.Conv2d(32, 64, 4), #64,43,43
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #64,21,21
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, 3), #128,19,19
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #128,9,9
            nn.Dropout(p=0.3),

            nn.Conv2d(128, 256, 2), #256,8,8
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #256,4,4
            nn.Dropout(p=0.4),

            nn.Flatten(),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256,30),

        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        image, keypoints = batch['image'], batch['keypoints']
        predicted_keypoints = self.forward(image).view(-1, 15, 2)

        loss = F.mse_loss(keypoints, predicted_keypoints)
        return loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        return optim

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
