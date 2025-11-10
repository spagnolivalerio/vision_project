import torch
import torch.nn as nn
from utils import gradient_penalty


ngpu = 1
z_dim = 128
ngf = 128
ndf = 128
lr = 1e-4
lambda_gp = 10
critic_iterations = 5
nc = 1 # number of channels of our images (1 in the case of xrays)

# ----------
# Generator 
# ----------

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf//4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
# -----------------
# Critic
# -----------------

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 16, 1, 4, 2, 0, bias=False)
            
        )

    def forward(self, input):
        return self.main(input).view(-1)

    def train(self, iterations, z_dim, batch_size, real, gen, opt_critic, device):

        losses = []
        for i in range(iterations):
            z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake = gen(z)
            
            cscore_real = self.forward(real)
            cscore_fake = self.forward(fake.detach())

            gp = gradient_penalty(self, real, fake, device=device)
            loss_critic = (cscore_fake.mean() - cscore_real.mean()) + lambda_gp * gp
            losses.append(loss_critic.item())
            
            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()
        
        return sum(losses) / iterations
