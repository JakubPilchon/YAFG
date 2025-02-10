import torch
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
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
    
    def __generate_images(self, num: int = 5, device = torch.device("cpu")) -> torch.Tensor:
        mean = torch.empty((num, self.latent_dim)).normal_().to(device) #torch.rand((num, self.latent_dim), dtype=torch.float).to(device)
        std  = torch.empty((num, self.latent_dim)).normal_().to(device) #torch.rand((num, self.latent_dim), dtype=torch.float).to(device)

        eps = self.sample(mean, std, device)
        images = self.decoder_forward(eps)

        return images

    def fit(self,
            train_dataloader : torch.utils.data.DataLoader,
            valid_dataloader : torch.utils.data.DataLoader,
            epochs: int = 1 ,
            logs_dir : str = "model/logs",
            cuda_available: bool = False,
            lr: float  = 0.001 
            ) -> None:
        
        #KL_LOSS = torch.nn.KLDivLoss(reduction="batchmean")
        L2_LOSS = torch.nn.MSELoss()
        OPTIMIZER = torch.optim.Adam(self.parameters(), lr = lr)
        WRITER = SummaryWriter(logs_dir)
        BETA = 0.04

        if cuda_available:
            device = torch.device("cuda")
            print("Loading model into gpu")
        else:
            device = torch.device("cpu")

        self.to(device)

        for e in range(1, 1+epochs):
            
            print(f"Epoch: {e}")

            train_loss = 0
            valid_loss = 0

            tqdm_dataloader = tqdm(train_dataloader)
            for i, train_data in enumerate(tqdm_dataloader):

                train_data  = train_data.to(device)

                OPTIMIZER.zero_grad()

                mean, std = self.encoder_forward(train_data)
                eps = self.sample(mean, std, device)
                decoded = self.decoder_forward(eps)

                kl_loss = self.kl_loss(mean, std)
                l2_loss = L2_LOSS(train_data, decoded)
                loss = BETA *  kl_loss + l2_loss
                loss.backward()

                OPTIMIZER.step()

                train_loss += loss.item()

                tqdm_dataloader.set_postfix({"LOSS": loss.item(), "KL:": kl_loss.item(), "L2:": l2_loss.item()})
                WRITER.add_scalars("losses", {"LOSS": loss.item(), "KL:": kl_loss.item(), "L2:": l2_loss.item()}, global_step=e*len(train_dataloader)+i)
            
            with torch.no_grad():
                for valid_data in valid_dataloader:

                    valid_data = valid_data.to(device)

                    mean, std = self.encoder_forward(valid_data)
                    eps = self.sample(mean, std, device)
                    decoded = self.decoder_forward(eps)

                    kl_loss = self.kl_loss(mean, std)
                    l2_loss = L2_LOSS(valid_data, decoded)
                    loss = kl_loss + l2_loss

                    valid_loss += loss.item()

                images = self.__generate_images(device=device)

                WRITER.add_images("Generated images", images, global_step=e)

                WRITER.add_scalar("VAL_LOSS", valid_loss/len(valid_data), e*len(train_dataloader))

        
            print(f"    Avg Loss: {train_loss/len(train_dataloader)} Valid Loss {valid_loss/len(valid_dataloader)}")

                