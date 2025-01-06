import os
import torch
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
    im = data[0]
    print(im)
    print(im.shape)
    im = im.reshape((1, 3, 128, 128))
    #im = ToPILImage()(im)
    v = VAE()
    v.eval()
    im = v.encoder_forward(im)
    print(im)
    #im.show()

