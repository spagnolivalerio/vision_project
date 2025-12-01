import os
from glob import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from cleanfid import fid
from utils import crop_and_normalize 

CONVERT_IMAGES = True
REAL_DIR = "FID_data/true"
FAKE_DIR = "FID_data/fake"

inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5])
])

def normalize(img_paths):

    for path in img_paths:
        img = Image.open(path).convert("L")        
        tensor = crop_and_normalize(img)      

        # invert normalization
        t = inv_normalize(tensor)
        t = torch.clamp(t, 0, 1).squeeze(0)

        out = transforms.ToPILImage()(t)

        out.save(path)

if __name__ == "__main__":

    if CONVERT_IMAGES:
        img_paths = sorted(glob(os.path.join(REAL_DIR, "*")))
        print(f"Found {len(img_paths)} images")
        normalize(img_paths)
    
    # Compute FID
    fid_value = fid.compute_fid(REAL_DIR, FAKE_DIR, mode="clean")
    print(f"FID VALUE: {fid_value:.6f}")
