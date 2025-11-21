import os
import sys
from glob import glob
from PIL import Image
import torch
from torch.utils.data import DataLoader

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

if IMAGE_SIZE == 128:
    DATA_PATH = "diff_data/real_57"
    BATCH_SIZE = 8           
    ACCUM_STEPS = 1    
else:
    DATA_PATH = "diff_data/real"
    BATCH_SIZE = 8            
    ACCUM_STEPS = 1   

N_SAMPLE = 1
TOTAL_STEPS = 30000
LR = 1e-4
SAVE_EVERY = 250
PRINT_EVERY = 1
RESUME_CKPT = "checkpoints/eps_greedy/eps_greedy_256_diffusion_step_500_600_timesteps.pt"           
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS_START = 0.05
EPS_END = 0.65
EPS_STEPS = (2 / 3) * TOTAL_STEPS
EDGE_START = 30
EDGE_END = 200

dataset = DentalDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
print(f"Dataset loaded: {DATA_PATH}")

diffusion = Diffusion(image_size=IMAGE_SIZE, timesteps=TIME_STEPS, device=DEVICE)
optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LR)

os.makedirs("checkpoints", exist_ok=True)

# ------------------------------------------------------------
# Resume checkpoint (if needed)
# ------------------------------------------------------------
start_step = 0
if RESUME_CKPT and os.path.exists(RESUME_CKPT):
    print(f"Checkpoint found: {RESUME_CKPT}")
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
    diffusion.model.load_state_dict(ckpt["model_state_dict"])
    print("Model loaded.")
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("Optimizer loaded.")
    start_step = ckpt.get("step", 0)
    print(f"Resumed from step {start_step}")
else:
    print("No checkpoint found. Starting training from scratch.")

print("Work in progress, trust the process ;)")
step = start_step
optimizer.zero_grad()

for epoch in range(999999):
    for i, imgs in enumerate(dataloader):

        progress = step / EPS_STEPS
        eps = min(EPS_END, EPS_START + progress * (EPS_END - EPS_START))
        edge = min(EDGE_END, EDGE_START + progress * (EDGE_END - EDGE_START))

        imgs = imgs.to(DEVICE)
        if torch.rand(1).item() < eps:
            t = torch.randint(0, int(edge), (imgs.shape[0],), device=DEVICE)
        else:
            t = torch.randint(0, TIME_STEPS, (imgs.shape[0],), device=DEVICE)


        loss = diffusion.p_losses(imgs, t)
        loss = loss / ACCUM_STEPS # loss normalization
        loss.backward()

        if (i + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % PRINT_EVERY == 0:
                print(f"Step {step:05d} | Loss: {loss.item() * ACCUM_STEPS:.6f} \t eps: {eps:.5f} \t t: {t}")

            if step % SAVE_EVERY == 0:
                ckpt_path = f"checkpoints/eps_greedy/eps_greedy_{IMAGE_SIZE}_diffusion_step_{step}_{TIME_STEPS}_timesteps.pt"
                torch.save({
                    "model_state_dict": diffusion.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if step >= TOTAL_STEPS:
                print(f"Training complete ({TOTAL_STEPS} steps reached)")
                break

    if step >= TOTAL_STEPS:
        break
