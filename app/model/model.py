import torch
from torch import nn
from typing import Tuple


class VAE(nn.Module):

    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.latent_dim = 32

        self.encoder_body = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 4, 2),
            #nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 4, 2),
            #nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),


            nn.Conv2d(128, 256, 4, 2),
            #nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),

            nn.Flatten(),

            nn.Linear(6*6*256, 512),
            nn.LeakyReLU(),
            ##nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            ##nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU()
            #nn.BatchNorm1d(128),
            #nn.Dropout(0.2)
        )
        self.encoder_std = nn.Linear(128, self.latent_dim)
        self.encoder_mean = nn.Linear(128, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 8*8*512),
            nn.LeakyReLU(),
            nn.Unflatten(1, (512,8,8)),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(16),
            #nn.Dropout2d(0.2),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(8),
            #nn.Dropout2d(0.2),

            nn.ConvTranspose2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
            )

    def encoder_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder_body(x)

        std = self.encoder_std(x)
        mean = self.encoder_mean(x)

        return (mean, std)

    def sample(self, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
        eps = torch.distributions.normal.Normal(0., 1.).sample(mean.shape).to(device)
        return eps * torch.exp(std * 0.5) + mean
    
    def decoder_forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
    def kl_loss(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return - 0.5 * torch.mean(1. + std - torch.square(mean) - torch.exp(std))
        #return -0.5 * torch.sum(1 + torch.log(torch.square(std)) - torch.square(mean) - torch.square(std), dim=1).mean()
    
    def generate_images(self, num: int = 5, device = torch.device("cpu")) -> torch.Tensor:
        mean = torch.empty((num, self.latent_dim)).normal_().to(device) #torch.rand((num, self.latent_dim), dtype=torch.float).to(device)
        std  = torch.empty((num, self.latent_dim)).normal_().to(device) #torch.rand((num, self.latent_dim), dtype=torch.float).to(device)

        eps = self.sample(mean, std, device)
        images = self.decoder_forward(eps)
        #images = torch.nn.functional.sigmoid(images)
        images = images.to(torch.device("cpu"))
        return images

    
                