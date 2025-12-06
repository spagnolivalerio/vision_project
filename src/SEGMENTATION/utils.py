import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 128

crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE)
])

mask_crop = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE)
])

crop_and_normalize = transforms.Compose([
    crop,
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

hflip = transforms.functional.hflip

rotate_right = transforms.RandomRotation(degrees=(10, 10))

rotate_left = transforms.RandomRotation(degrees=(-10, -10))

def compute_masks(img, anns):

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in anns:
        for seg in ann["segmentation"]:
            polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            draw.polygon(polygon, outline=255, fill=255)
        
    mask = np.array(mask)

    return mask


    
