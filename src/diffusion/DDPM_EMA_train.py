import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DentalDataset
from utils import IMAGE_SIZE
from models.diffusion import Diffusion

class EMA:
    def __init__(self, model, decay):
        """
        Initialize EMA class to manage exponential moving average of model parameters.
        
        Args:
            model (torch.nn.Module): The model for which EMA will track parameters.
            decay (float): Decay rate, typically a value close to 1, e.g., 0.999.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """
        Update shadow parameters with exponential decay.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    @torch.no_grad()
    def apply_shadow(self):
        """
        Apply shadow (EMA) parameters to model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    @torch.no_grad()
    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

DATA_PATH = "../../data/dentex/training_data/quadrant/xrays/"
OUTPUT_DIR = "outputs/running"
BATCH_SIZE = 16            
TIME_STEPS = 1000
TOTAL_STEPS = 30000
LR = 2e-5
SAVE_EVERY = 250
SHOW_EVERY = 100
PRINT_EVERY = 1
RESUME_CKPT = "checkpoints/DDPM/128_diffusion_step_9250_300_timesteps.pt"  
EMA_DECAY = 0.999         
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__": 
    
    # Dataset and dataloader initialization
    dataset = DentalDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    print(f"Dataset loaded: {DATA_PATH}")
    
    # Model instantiation
    diffusion = Diffusion(image_size=IMAGE_SIZE, timesteps=TIME_STEPS, device=DEVICE)
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LR)

    ema = EMA(diffusion.model, EMA_DECAY)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Resume checkpoint (if needed)
    start_step = 0
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        print(f"Checkpoint found: {RESUME_CKPT}")

        # Checkpoint location and loading device
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        diffusion.model.load_state_dict(ckpt["model_state_dict"])
        print("Model loaded.")

        # Optimizer state loading
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Optimizer loaded.")
        
        if "ema" in ckpt:
            ema.shadow = ckpt["ema"]
            print("EMA loaded.")

        # Taking the start step
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("Work in progress, trust the process ;)")

    # Start training
    step = start_step
    diffusion.model.train()
    optimizer.zero_grad()

    for epoch in range(999999):
        for i, imgs in enumerate(dataloader):

            imgs = imgs.to(DEVICE)

            # Create a t array to contain timestep for each image in the batch
            t = torch.randint(0, TIME_STEPS, (imgs.shape[0],), device=DEVICE) 

            # Loss computing and backpropagation
            loss = diffusion.p_losses(imgs, t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Step increment
            step += 1

            # Ema weights update
            ema.update()

            # Log print
            if step % PRINT_EVERY == 0:
                print(f"Step {step:05d} | Loss: {loss.item():.6f}")

            # Save a photo every SHOW_EVERY steps
            if step % SHOW_EVERY == 0:

                diffusion.model.eval()
                ema.apply_shadow()
                sample = diffusion.sample(n_samples=1)
                ema.restore()
                diffusion.model.train()
                # Sample normalization to visualize it
                sample = (sample.clamp(-1, 1) + 1) / 2

                for i in range(sample.shape[0]):
                    img = sample[i, 0].cpu().numpy()
                    save_path = os.path.join(OUTPUT_DIR, f"{step}.png")
                    plt.imsave(save_path, img, cmap="gray")
                    print(f"Saved: {save_path}")

            # Checkpoint save
            if step % SAVE_EVERY == 0:
                ckpt_path = f"checkpoints/DDPM/{IMAGE_SIZE}_diffusion_step_{step}_{TIME_STEPS}_timesteps.pt"
                torch.save({
                    "model_state_dict": diffusion.model.state_dict(),
                    "ema": ema.shadow, 
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # Stop condition
        if step >= TOTAL_STEPS:
            print(f"Training complete ({TOTAL_STEPS} steps reached)")
            break
