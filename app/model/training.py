import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage, Resize, ToPILImage
import PIL
from torchvision.io import read_image
from model import VAE
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

class MyDataset(VisionDataset):

    def __init__(self,
                 root: os.PathLike | str,
                 transforms: Compose
                ) -> None:
        
        super().__init__(root, transforms = transforms)

        self.dataset_list = os.listdir(root)
        self.root = root
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset_list)
    
    def __getitem__(self, index: int) -> torch.Tensor:

        image = read_image(os.path.join(self.root, self.dataset_list[index]))

        image = self.transforms(image)

        return image

transforms = Compose([

    ToImage(),
    Resize((128, 128)),
    RandomHorizontalFlip(),
    ToDtype(torch.float32, scale=True)
])
        
def fit(model : torch.nn.Module,
        train_dataloader : torch.utils.data.DataLoader,
        valid_dataloader : torch.utils.data.DataLoader,
        epochs: int = 1 ,
        logs_dir : str = "model/logs",
        cuda_available: bool = False,
        lr: float  = 1e-4 
        ) -> None:
    
    #KL_LOSS = torch.nn.KLDivLoss(reduction="batchmean")
    L2_LOSS = torch.nn.MSELoss()
    #L2_LOSS = torch.nn.BCELoss(reduction="mean")
    #L2_LOSS = torch.nn.functional.binary_cross_entropy
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr = lr)
    WRITER = SummaryWriter(logs_dir)
    BETA = 0.01

    torch.autograd.set_detect_anomaly(True)

    if cuda_available:
        device = torch.device("cuda")
        print("Loading model into gpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    for e in range(1, 1+epochs):
        
        print(f"Epoch: {e}")
        train_loss = 0
        valid_loss = 0
        tqdm_dataloader = tqdm(train_dataloader)
        for i, train_data in enumerate(tqdm_dataloader):
            train_data  = train_data.to(device)
            train_data = torch.clamp(train_data, 0 + 1e-7, 1. - 1e-7)
            OPTIMIZER.zero_grad()

            #forward pass
            mean, std = model.encoder_forward(train_data)
            eps = model.sample(mean, std, device)
            decoded = model.decoder_forward(eps)

            kl_loss = model.kl_loss(mean, std)
            l2_loss = L2_LOSS(train_data.view(-1, 128*128*3), decoded.view(-1, 128*128*3))
            loss =   l2_loss  + kl_loss * BETA
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            OPTIMIZER.step()
            train_loss += loss.item()
            tqdm_dataloader.set_postfix({"LOSS": loss.item(), "KL:": kl_loss.item(), "L2:": l2_loss.item()})
            WRITER.add_scalars("losses", {"LOSS": loss.item(), "KL:": kl_loss.item(), "L2:": l2_loss.item()}, global_step=e*len(train_dataloader)+i)
        
        with torch.no_grad():
            for valid_data in valid_dataloader:
                valid_data = valid_data.to(device)

                # forward oass
                mean, std = model.encoder_forward(valid_data)
                
                decoded = model.decoder_forward(model.sample(mean, std, device))

                kl_loss = model.kl_loss(mean, std)
                l2_loss = L2_LOSS(valid_data, decoded)
                loss = BETA * kl_loss + l2_loss
                valid_loss += loss.item()

            images = model.generate_images(device=device)
            fig, ax = plt.subplots(1, 5, figsize=(7.5, 2.5))
            fig.suptitle(f"Epoch {e}")
            for n in range(5):
                ax[n].imshow(torch.reshape(images[n], (128,128,3)).numpy())
                ax[n].axis("off")
                fig.savefig(f"plots/plot_{e}.png")
            
            WRITER.add_images("Generated images", images, global_step=e)
            WRITER.add_scalar("VAL_LOSS", valid_loss/len(valid_data), e*len(train_dataloader))
    
        print(f"    Avg Loss: {train_loss/len(train_dataloader)} Valid Loss {valid_loss/len(valid_dataloader)}")

if __name__ == "__main__":
    data = MyDataset("images", transforms=transforms)
    train_data, valid_data = random_split(data, (0.8, 0.2))
    train_data = DataLoader(train_data, 64, True)
    valid_data = DataLoader(valid_data, 64, True)


    #lookup https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder
    v = VAE()
    fit(model=v,train_dataloader=train_data,
        valid_dataloader=valid_data,
        epochs=4,
        logs_dir = "logs/log_final",
        cuda_available = torch.cuda.is_available())
    

    torch.save(v.state_dict(), "saved_model.pt")
