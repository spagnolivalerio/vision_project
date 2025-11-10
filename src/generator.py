import torch
from gan import Generator
import torchvision.utils as vutils
import os

z_dim = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "syn_dataset"
os.makedirs(output_dir, exist_ok=True)

gen = Generator().to(device)
gen.load_state_dict(torch.load("weights/generator_weights_v1.pth"))
gen.eval()

num_images = 50

for i in range(num_images):
    z = torch.randn(1, z_dim, 1, 1, device=device)
    with torch.no_grad():
        fake = gen(z).detach().cpu()
    vutils.save_image(fake, f"{output_dir}/synthetic_{i}.png", normalize=True)