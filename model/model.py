import torch
from torch import nn
from typing import Tuple


class VAE(nn.Module):

    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.encoder_body = nn.ModuleList([
            nn.Conv2d(3, 32, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),


            nn.Conv2d(128, 256, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),

            nn.Flatten(),

            nn.Linear(7*7*256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 258),
            nn.ReLU(),
            nn.BatchNorm1d(258),
            nn.Dropout(0.2),
            
            nn.Linear(258, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        ])

        self.encoder_std = nn.Linear(128, 3)
        self.encoder_mean = nn.Linear(128, 3)

    def encoder_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, layer in enumerate(self.encoder_body):
            x = layer(x)

        std = self.encoder_std(x)
        mean = self.encoder_mean(x)

        return (mean, std)

    def decoder_forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        pass