import torch
from torchvision.transforms.functional import to_pil_image
from typing import Dict, Tuple

SHAPE = (1,16)

def preprocess(parameters: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
    means_vector = torch.distributions.normal.Normal(0.0, 1.0).sample(SHAPE)
    #means_vector = torch.zeros(SHAPE)
    std_vector = torch.zeros(SHAPE)

    std = float(parameters["t"])
    std = std / 50 + 1e-2
    std_vector += std

    for i in range(1, 5):

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