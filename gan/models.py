import torch
import torch.nn as nn
import torch.optim as optim

from typing import List, Optional, Union

from utils import training_data, noise


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        image_dim: int,
    ):
        super(Generator, self).__init__()

        self.layer1 = nn.Linear(noise_dim, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, image_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.layer1(x)
        x = self.leaky_relu(x)

        x = self.layer2(x)
        x = self.leaky_relu(x)

        x = self.layer3(x)
        x = self.leaky_relu(x)

        x = self.layer4(x)
        x = self.tanh(x)
        x = x.view(x.shape[0], 1, 28, 28)

        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        image_dim: int,
    ):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Linear(image_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        y = x.view(x.shape[0], x.shape[2] * x.shape[3])
        y = self.layer1(y)
        y = self.leaky_relu(y)

        y = self.layer2(y)
        y = self.leaky_relu(y)

        y = self.layer3(y)
        y = self.leaky_relu(y)

        return y


class GAN:
    def __init__(
        self,
        batch_size: int = 8,
        learning_rate: int = 1e-3,
        noise_dim: int = 100,
        image_dim: int = 784,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.device = device

        self.g = Generator(noise_dim=self.noise_dim, image_dim=self.image_dim).to(
            device
        )
        self.d = Discriminator(image_dim=self.image_dim).to(device)

        self.g_opt = optim.Adam(self.g.parameters())
        self.d_opt = optim.Adam(self.d.parameters())

        self.real_targets = (
            torch.FloatTensor(self.batch_size, 1).uniform_(0.8, 1.0)
        ).to(device)
        self.fake_targets = (
            torch.FloatTensor(self.batch_size, 1).uniform_(0.0, 0.2)
        ).to(device)

        self.loss_func = nn.BCEWithLogitsLoss()
        self.train_data = training_data(self.batch_size)

    def d_loss(self, real_logits, fake_logits):
        logits = torch.cat((real_logits, fake_logits))
        targets = torch.cat((self.real_targets, self.fake_targets))
        loss = self.loss_func(logits, targets)

        return loss

    def g_loss(self, fake_logits):
        return self.loss_func(fake_logits, self.real_targets)

    def train(self, epochs: int = 10):
        overall_g_losses = []
        overall_d_losses = []
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            for imgs, _ in self.train_data:
                imgs = imgs.to(self.device)

                self.d_opt.zero_grad()

                real_logits = self.d(imgs)
                fake_imgs = self.g(
                    noise(
                        dim=self.noise_dim,
                        batch_size=self.batch_size,
                        device=self.device,
                    )
                ).detach()
                fake_logits = self.d(fake_imgs)

                discr_loss = self.d_loss(real_logits, fake_logits)
                discr_loss.backward()

                self.d_opt.step()

                self.g_opt.zero_grad()

                fake_imgs = self.g(noise(self.noise_dim, self.batch_size, self.device))
                fake_logits = self.d(fake_imgs)

                gen_loss = self.g_loss(fake_logits)
                gen_loss.backward()

                self.g_opt.step()

            if epoch % 2 == 0:
                pass

    def generate(self):
        pass
