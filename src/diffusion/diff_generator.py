import os
import torch
import matplotlib.pyplot as plt
from models.diffusion import Diffusion 
from utils import IMAGE_SIZE
from utils import TIME_STEPS

N_SAMPLE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = f"checkpoints/eps_greedy/eps_greedy_256_diffusion_step_1500_600_timesteps.pt"
OUTPUT_DIR = f"outputs/samples_{N_SAMPLE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

diffusion = Diffusion(
    image_size=IMAGE_SIZE,
    timesteps=TIME_STEPS,
    beta_start=1e-4,
    beta_end=0.02,
    device=DEVICE
)

print(f"Load from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
diffusion.model.load_state_dict(checkpoint["model_state_dict"])
step = checkpoint.get("step")
print(f"Checkpoint loaded (step = {step})")

print("Generating new images...")
diffusion.model.eval()

with torch.no_grad():
    samples = diffusion.sample(n_samples=1)


samples = (samples.clamp(-1, 1) + 1) / 2

print("Saved in:", OUTPUT_DIR)

for i in range(samples.shape[0]):
    img = samples[i, 0].cpu().numpy()
    save_path = os.path.join(OUTPUT_DIR, f"generated_{i}_step{step}_{IMAGE_SIZE}px.png")
    plt.imsave(save_path, img, cmap="gray")
    print(f"Saved: {save_path}")

fig, axes = plt.subplots(1, samples.shape[0], figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(samples[i, 0].cpu().numpy(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()
