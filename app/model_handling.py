import torch
from torchvision.transforms.functional import to_pil_image
from typing import Dict, Tuple
import random


def preprocess(parameters: Dict[str, str], latent_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (1, latent_dim)
    means_vector = torch.distributions.normal.Normal(0.0, 1.0).sample(shape)
    #means_vector = torch.zeros(shape)
    std_vector = torch.zeros(shape)

    std = float(parameters["t"])
    std = std / 50 + 1e-2
    std_vector += std

    for i in range(latent_dim):

        mean = float(parameters[f"p{i}"])
        mean = mean /25 - 2

        means_vector[0, i-1] = mean

    return (means_vector, std_vector)

def generate_image(model: torch.nn.Module, 
                   mean_vec: torch.Tensor, 
                   std_vec: torch.Tensor
                   ) -> None:
    with torch.no_grad():
        eps = model.sample(mean_vec, std_vec, torch.device("cpu"))
        img = model.decoder_forward(eps)
        img = to_pil_image(img[0])
        img.save('app/static/image.png')

def alt_generate_image(model: torch.nn.Module,
                       mean_vec: torch.Tensor
                       ) -> None:
    with torch.no_grad():
        img = model.decoder_forward(mean_vec)
        img = to_pil_image(img[0])
        img.save('app/static/image.png')

def generate_name() -> str:
    name_gen = ["ma", "mo", "ka", "ko", "ke", "lo", "ro", "ni", "mi", "vi", "wa", "na", "go"]
    return "".join(random.choices(name_gen, k=3)).capitalize()