import torch

from models import GAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GAN(device=device)

model.train()
