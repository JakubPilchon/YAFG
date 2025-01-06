import torch 
from typing import Tuple


class VAE(torch.nn.Module):

    def __init__(self) -> None:
        super(self).__init__()

    def encoder_forward(self, torch_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def decoder_forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        pass