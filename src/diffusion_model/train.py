import os
import sys
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# aggiunge il path superiore per importare utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# import delle trasformazioni e del modello
from utils import crop_and_normalize
from models.diffusion import Diffusion


# ------------------------------------------------------------
# 1️⃣ Dataset per radiografie dentali (usa le tue trasformazioni)
# ------------------------------------------------------------
class DentalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.paths = sorted(glob(os.path.join(root_dir, "*")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"Nessuna immagine trovata in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = crop_and_normalize(img)
        return img


# ------------------------------------------------------------
# 2️⃣ Parametri
# ------------------------------------------------------------
DATA_PATH = "diff_data/real"
BATCH_SIZE = 8
TOTAL_STEPS = 30000      # numero totale di step (non epoche)
LR = 1e-4
SAVE_EVERY = 1000
RESUME_CKPT = "checkpoints/diffusion_step_2il 2000.pt"  # checkpoint da cui ripartire
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# 3️⃣ Setup dataset e dataloader
# ------------------------------------------------------------
dataset = DentalDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

# ------------------------------------------------------------
# 4️⃣ Crea modello di diffusione
# ------------------------------------------------------------
diffusion = Diffusion(image_size=128, timesteps=300, device=DEVICE)
optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LR)

os.makedirs("checkpoints", exist_ok=True)

# ------------------------------------------------------------
# 5️⃣ Ripristina checkpoint se esiste
# ------------------------------------------------------------
start_step = 0
if os.path.exists(RESUME_CKPT):
    print(f"📂 Checkpoint trovato: {RESUME_CKPT}")
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
    diffusion.model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt.get("step", 0)
    print(f"✅ Ripristinato checkpoint (step = {start_step})")
else:
    print("🚀 Nessun checkpoint trovato, inizio da zero.")

# ------------------------------------------------------------
# 6️⃣ Ciclo di training
# ------------------------------------------------------------
print("🎯 Starting/resuming training...")
step = start_step

for epoch in range(999999):  # usiamo un loop lungo, controllato da step
    for imgs in dataloader:
        imgs = imgs.to(DEVICE)
        t = torch.randint(0, diffusion.timesteps, (imgs.shape[0],), device=DEVICE).long()

        loss = diffusion.p_losses(imgs, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 50 == 0:
            print(f"Step {step:05d} | Loss: {loss.item():.6f}")

        # salvataggio periodico del modello
        if step % SAVE_EVERY == 0:
            ckpt_path = f"checkpoints/diffusion_step_{step}.pt"
            torch.save({
                "model_state_dict": diffusion.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            }, ckpt_path)
            print(f"💾 Salvato checkpoint: {ckpt_path}")

        # fine allenamento
        if step >= TOTAL_STEPS:
            print(f"✅ Training completo ({TOTAL_STEPS} step raggiunti)")
            break

    if step >= TOTAL_STEPS:
        break
