from torchvision.datasets import CocoDetection
from utils import compute_masks
from torch.utils.data import Dataset
from utils import crop_and_normalize, crop
from utils import blank_mask
import os
import glob
from PIL import Image

pct = 0.2

dataroot = "../data/dentex/training_data/quadrant_enumeration/xrays/"
annfile  = "../data/dentex/training_data/quadrant_enumeration/train_quadrant_enumeration.json"

class DentexDataset(CocoDetection):

    def __init__(self):
        super().__init__(root=dataroot, annFile=annfile)

class SegmentationDataset(Dataset):

    def __init__(self, root_dir, img_transform = None, mask_transform = None):

        self.root_dir = root_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.images_dir = os.path.join(root_dir, "xrays")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.image_files = sorted(os.listdir(self.images_dir))

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):
        
        img_name = self.image_files[index]
        mask_name = "mask_" + img_name
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform is not None:

            img = self.img_transform(img)
        
        if self.mask_transform is not None:

            mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)   

        return img, mask
 
    def augment_the_dataset(self, t_augmentation):
            
        index = len(self.image_files)
        
        for img_name in self.image_files:

            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, "mask_" + img_name)
            
            img = Image.open(img_path)
            mask = Image.open(mask_path)

            img_flip  = t_augmentation(img)
            mask_flip = t_augmentation(mask)

            img_flip.save(os.path.join(self.images_dir, f"train_{index}.png"))
            mask_flip.save(os.path.join(self.masks_dir, f"mask_train_{index}.png"))

            index += 1
        
        self.image_files = sorted(os.listdir(self.images_dir))

class UnsupervisedImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = sorted(glob.glob(os.path.join(root_dir, "*.png"))) + \
                         sorted(glob.glob(os.path.join(root_dir, "*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("L")  # radiografie in grayscale
        if self.transform:
            img = self.transform(img)
        return img
    

if __name__ == "__main__":

    """os.makedirs("masks", exist_ok=True)

    dentex = DentexDataset()
    
    for i, (img, anns) in enumerate(dentex):

        mask = compute_masks(img, anns)
        mask = Image.fromarray(mask)
        mask.save(os.path.join("masks", f"mask_train_{i}.png"))"""
    
dataroot = "../data/dentex/training_data/quadrant_enumeration/"
mask_dir = os.path.join(dataroot, "masks")
ood_dir = os.path.join(dataroot, "chest_xrays/images")
xrays_dir = os.path.join(dataroot, "xrays")
ood_xrays = os.path.join(dataroot, "ood_xrays")
ood_masks = os.path.join(dataroot, "ood_masks")


"""dataset = SegmentationDataset(
    root_dir=dataroot,
    img_transform=crop_and_normalize,
    mask_transform=crop
)

ood_degree = int(pct * len(dataset))

oods = sorted(os.listdir(ood_dir))
chosen_oods = random.sample(oods, min(ood_degree, len(oods)))

for img in chosen_oods:

    pil_img = blank_mask(img, ood_dir, mask_dir)
    save_path = os.path.join(xrays_dir, img)
    pil_img.save(save_path)
"""



    
