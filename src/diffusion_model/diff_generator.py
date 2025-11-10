import os
import torch
import matplotlib.pyplot as plt
from models.diffusion import Diffusion  # importa la classe Diffusion che hai già nel progetto


# ------------------------------------------------------------
# 1️⃣ CONFIGURAZIONE
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/diffusion_step_2000.pt"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2️⃣ CREA IL MODELLO (stessa architettura del training)
# ------------------------------------------------------------
diffusion = Diffusion(
    image_size=256,
    timesteps=300,
    beta_start=1e-4,
    beta_end=0.02,
    device=DEVICE
)

# ------------------------------------------------------------
# 3️⃣ CARICA IL CHECKPOINT
# ------------------------------------------------------------
print(f"📂 Carico checkpoint da: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
diffusion.model.load_state_dict(checkpoint["model_state_dict"])
step = checkpoint.get("step", "sconosciuto")
print(f"✅ Checkpoint caricato (step = {step})")

# ------------------------------------------------------------
# 4️⃣ GENERA NUOVE IMMAGINI
# ------------------------------------------------------------
print("🎨 Genero immagini sintetiche dal modello...")
diffusion.model.eval()

with torch.no_grad():
    samples = diffusion.sample(n_samples=1)  # genera 4 immagini

# rimappa da [-1, 1] → [0, 1] per visualizzazione
samples = (samples.clamp(-1, 1) + 1) / 2

# ------------------------------------------------------------
# 5️⃣ SALVA E MOSTRA I RISULTATI
# ------------------------------------------------------------
print("💾 Salvo immagini generate in:", OUTPUT_DIR)

for i in range(samples.shape[0]):
    img = samples[i, 0].cpu().numpy()
    save_path = os.path.join(OUTPUT_DIR, f"generated_{i}_step{step}.png")
    plt.imsave(save_path, img, cmap="gray")
    print(f"🖼️  Salvata: {save_path}")

# opzionale: mostra a video
fig, axes = plt.subplots(1, samples.shape[0], figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(samples[i, 0].cpu().numpy(), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()
