import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
from dataset import DentalDataset
from utils import crop_and_normalize, IMAGE_SIZE
import os

DEVICE = "cuda"
NUM_CLASSES = 33
WEIGHTS_PATH = "weights/unet.pt"

TEST_IMGS = "data/validation_set/xrays"
TEST_MASKS = "data/validation_set/masks"
OUT_DIR = "test_results"

os.makedirs(OUT_DIR, exist_ok=True)


def generate_palette(n_classes):
    np.random.seed(0)
    palette = np.random.randint(0, 255, size=(n_classes, 3))
    palette[0] = [0, 0, 0]   # background
    return palette

PALETTE = generate_palette(NUM_CLASSES)


# -------------------------
# DE-NORMALIZATION
# -------------------------
def denormalize(img_tensor):
    """
    Reverts crop_and_normalize:
    img_norm = (img/255 - 0.5) / 0.5
    img = ((img_norm * 0.5) + 0.5) * 255
    """
    img = img_tensor.squeeze().cpu().numpy()
    img = ((img * 0.5) + 0.5) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# -------------------------
# COLORIZE MASK
# -------------------------
def colorize_mask(mask):
    """ mask: HxW integer class map """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        color[mask == cls] = PALETTE[cls]
    return color


# -------------------------
# MODEL LOADING
# -------------------------
model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=1,
    classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

print("Model loaded.")


# -------------------------
# DATASET
# -------------------------
dataset = DentalDataset(TEST_IMGS, TEST_MASKS, augment=False)


# -------------------------
# TEST LOOP
# -------------------------
for idx in range(len(dataset)):
    img, gt_mask = dataset[idx]

    # Move to GPU
    img_gpu = img.unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        pred = model(img_gpu)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Prepare visualization
    img_vis = denormalize(img)                # grayscale original
    gt_color = colorize_mask(gt_mask.numpy()) # ground truth colored
    pred_color = colorize_mask(pred)          # prediction colored

    # Resize grayscale to 3 channels for overlay
    img_3c = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB)

    # Overlays
    overlay_pred = cv2.addWeighted(img_3c, 0.6, pred_color, 0.4, 0)
    overlay_gt   = cv2.addWeighted(img_3c, 0.6, gt_color, 0.4, 0)

    # Final panel
    final = np.hstack([
        img_3c,
        gt_color,
        pred_color,
        overlay_gt,
        overlay_pred
    ])

    save_path = os.path.join(OUT_DIR, f"result_{idx}.png")
    cv2.imwrite(save_path, final)

    print(f"Saved {save_path}")

print("Done.")
