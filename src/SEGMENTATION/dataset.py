from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import numpy as np
import torch
import random

from utils import crop_and_normalize, mask_crop, hflip, rotate_left, rotate_right


class DentalDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, augment=True):
        self.imgs_files = sorted(glob(imgs_dir + "/*"))
        self.masks_files = sorted(glob(masks_dir + "/*"))
        self.augment = augment

        assert len(self.imgs_files) == len(self.masks_files), \
            "Number of images and masks do not match"

    def __len__(self):
        return len(self.imgs_files)

    def __getitem__(self, index):
        img = Image.open(self.imgs_files[index]).convert("L")

        mask = np.array(Image.open(self.masks_files[index]), dtype=np.uint8)

        if self.augment:
            if random.random() < 0.5:
                img = hflip(img)
                mask = np.fliplr(mask)

            if random.random() < 0.25:
                img = rotate_left(img)
                mask = np.rot90(mask, k=1)

            if random.random() < 0.25:
                img = rotate_right(img)
                mask = np.rot90(mask, k=-1)

        img = crop_and_normalize(img)   

        mask_pil = Image.fromarray(mask)
        mask_pil = mask_crop(mask_pil)             

        mask = torch.from_numpy(np.array(mask_pil)).long()

        return img, mask
