import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from globals import IMAGE_SIZE
from colordict import *
import random as rand

mask_crop = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE)
])

crop_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def class_to_color(classes): 
    
    palette = {}
    colors_list = list(ColorDict().values())
    rand.shuffle(colors_list)
    for i, cls in enumerate(classes):  
        palette[cls] = colors_list[i]
    
    return palette

def color_mask(mask, palette):

    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in palette.items():
        rgb = color[:3]
        img[mask == cls] = rgb

    return img

    
