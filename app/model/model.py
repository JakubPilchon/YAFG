import torch
from torch import nn
from typing import Tuple


class VAE(nn.Module):

    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.latent_dim = 16

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
            nn.Tanh(),
            #nn.BatchNorm1d(128),
            #nn.Dropout(0.2)
        ])
        self.encoder_std = nn.Linear(128, self.latent_dim)
        self.encoder_mean = nn.Linear(128, self.latent_dim)

        self.decoder = nn.ModuleList([
            nn.Linear(self.latent_dim, 8*8*256),
            nn.LeakyReLU(),
            nn.Unflatten(1, (256,8,8)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(16),
            #nn.Dropout2d(0.2),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(8),
            #nn.Dropout2d(0.2),

            nn.ConvTranspose2d(8, 3, 3, 1, 1),
            nn.Sigmoid()
            ])

    def encoder_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, layer in enumerate(self.encoder_body):
            x = layer(x)

        std = self.encoder_std(x)
        mean = self.encoder_mean(x)

        return (mean, std)

    def sample(self, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
        eps = torch.distributions.normal.Normal(0., 1.).sample(mean.shape).to(device)
        return eps * torch.exp(std * 0.5) + mean
    
    def decoder_forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)

        return x
    
    def kl_loss(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return - 0.5 * torch.mean(1. + std - torch.square(mean) - torch.exp(std))
    
    def generate_images(self, num: int = 5, device = torch.device("cpu")) -> torch.Tensor:
        mean = torch.empty((num, self.latent_dim)).normal_().to(device) #torch.rand((num, self.latent_dim), dtype=torch.float).to(device)
        std  = torch.empty((num, self.latent_dim)).normal_().to(device) #torch.rand((num, self.latent_dim), dtype=torch.float).to(device)

        eps = self.sample(mean, std, device)
        images = self.decoder_forward(eps)

        return images

    
                