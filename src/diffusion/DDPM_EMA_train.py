import os
import sys
from glob import glob
from PIL import Image
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils import crop_and_normalize, IMAGE_SIZE, TIME_STEPS
from models.diffusion import Diffusion

class DentalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.paths = sorted(glob(os.path.join(root_dir, "*")))
        if len(self.paths) == 0:
            print(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = crop_and_normalize(img)
        return img  


DATA_PATH = "../../data/dentex/training_data/quadrant/xrays/"
OUTPUT_DIR = "outputs/running"
BATCH_SIZE = 16
ACCUM_STEPS = 1
TOTAL_STEPS = 20000
LR = 1e-4
SAVE_EVERY = 250
SHOW_EVERY = 100
PRINT_EVERY = 1
RESUME_CKPT = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DentalDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
print(f"Dataset loaded: {DATA_PATH}")

diffusion = Diffusion(image_size=IMAGE_SIZE, timesteps=TIME_STEPS, device=DEVICE)
optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LR)

ema_model = Diffusion(image_size=IMAGE_SIZE, timesteps=TIME_STEPS, device=DEVICE).model
ema_decay = 0.9999

def update_ema(ema, model, decay):
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.data.mul_(decay).add_(p.data, alpha=1 - decay)

os.makedirs("checkpoints/DDPM", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

start_step = 0
if RESUME_CKPT and os.path.exists(RESUME_CKPT):
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
    diffusion.model.load_state_dict(ckpt["model_state_dict"])
    ema_model.load_state_dict(ckpt["ema_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"]
else:
    print("No checkpoint found. Starting training from scratch.")

step = start_step
diffusion.model.train()
optimizer.zero_grad()

for epoch in range(999999):
    for i, imgs in enumerate(dataloader):

        imgs = imgs.to(DEVICE)
        t = torch.randint(0, TIME_STEPS, (imgs.shape[0],), device=DEVICE)

        loss = diffusion.p_losses(imgs, t)
        loss = loss / ACCUM_STEPS
        loss.backward()

        if (i + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            update_ema(ema_model, diffusion.model, ema_decay)

            if step % PRINT_EVERY == 0:
                print(f"Step {step:05d} | Loss: {loss.item() * ACCUM_STEPS:.6f}")

            if step % SHOW_EVERY == 0:
                ema_model.eval()
                old_model = diffusion.model
                diffusion.model = ema_model
                sample = diffusion.sample(n_samples=1)
                diffusion.model = old_model
                sample = (sample.clamp(-1, 1) + 1) / 2

                for i in range(sample.shape[0]):
                    img = sample[i, 0].cpu().numpy()
                    save_path = os.path.join(OUTPUT_DIR, f"{step}.png")
                    plt.imsave(save_path, img, cmap="gray")
                    print(f"Saved: {save_path}")

            if step % SAVE_EVERY == 0:
                ckpt_path = f"checkpoints/DDPM_EMA/{IMAGE_SIZE}_diffusion_step_{step}_{TIME_STEPS}_timesteps.pt"
                torch.save({
                    "model_state_dict": diffusion.model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if step >= TOTAL_STEPS:
                print(f"Training complete ({TOTAL_STEPS} steps reached)")
                break

    if step >= TOTAL_STEPS:
        break
