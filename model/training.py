import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage, Resize, ToPILImage
import PIL
from torchvision.io import read_image
from model import VAE

class MyDataset(VisionDataset):

    def __init__(self,
                 root: os.PathLike,
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
        

if __name__ == "__main__":
    data = MyDataset("images", transforms=transforms)
    train_data, valid_data = random_split(data, (0.8, 0.2))
    train_data = DataLoader(train_data, 32, True)
    valid_data = DataLoader(valid_data, 32, True)

    v = VAE()
    v.fit(train_data, valid_data, 5)


