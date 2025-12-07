from globals import DEVICE, NUM_CLASSES, DATA_ROOT
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import DentalDataset
import torch
from utils import class_to_color, color_mask
import matplotlib.pyplot as plt
import numpy as np

VALIDATION_DIR = DATA_ROOT + "/validation_set"
VALIDATION_MASKS = VALIDATION_DIR + "/masks"
VALIDATION_IMGS = VALIDATION_DIR + "/xrays"
WEIGHTS_PATH = "weights/unet_legacy.pt"

BATCH_SIZE = 10 

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=1,
    classes=NUM_CLASSES,
)
model = model.to(DEVICE)

# Data configuration
dataset = DentalDataset(VALIDATION_MASKS, VALIDATION_IMGS, augment=False)
dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

# Palette creation
palette = class_to_color(list(range(NUM_CLASSES)))

# Loading weights
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    for idx, (imgs, masks_gt) in enumerate(dataloader):
        pass
                

            



