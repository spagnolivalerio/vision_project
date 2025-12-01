import torch.autograd as autograd
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(256)
])

crop_and_normalize = transforms.Compose([
    crop,
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

vflip = transforms.functional.vflip

rotate_right = transforms.RandomRotation(degrees=(45, 45))

rotate_left = transforms.RandomRotation(degrees=(-45, -45))

def gradient_penalty(critic, real, fake, device="cuda"):
    """
    GP del modello PANO-WGAN
    """
    batch_size, C, H, W = real.shape
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)

    real_fake_blend = epsilon * real + (1 - epsilon) * fake
    real_fake_blend = real_fake_blend.to(device)

    mixed_scores = critic.forward(real_fake_blend)

    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = autograd.grad(
        outputs=mixed_scores,
        inputs=real_fake_blend,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # shape (B, C, H, W)

    gradients = gradients.view(batch_size, -1)        # (B, C*H*W)
    grad_norm = gradients.norm(2, dim=1)              # (B,)

    gp = torch.mean((grad_norm - 1) ** 2)

    return gp 

def compute_masks(img, anns):
    """
    Calcola le maschere sulla base dei vertici dei poligoni deifniti nel json
    """
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in anns:
        for seg in ann["segmentation"]:
            polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            draw.polygon(polygon, outline=255, fill=255)
        
    mask = np.array(mask)

    return mask

def blank_mask(image_file, image_dir, mask_dir):

    mask_file = "mask_" + image_file
    save_path = os.path.join(mask_dir, mask_file)
    image_path = os.path.join(image_dir, image_file)
    
    img = Image.open(image_path)
    width, height = img.size

    mask_array = np.zeros((height, width), dtype=np.uint8)
    mask = Image.fromarray(mask_array, mode="L") 

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask.save(save_path)

    return img #return the PIL image to save it into the directory

    
