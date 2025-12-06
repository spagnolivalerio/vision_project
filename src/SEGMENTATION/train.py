import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from dataset import DentalDataset
from metrics import multiclass_iou, multiclass_dice, multiclass_precision_recall, pixel_accuracy
import os
import numpy as np
from PIL import Image

DATA_ROOT       = "data"
TRAIN_IMGS_DIR  = DATA_ROOT + "/training_set/xrays"
TRAIN_MASKS_DIR = DATA_ROOT + "/training_set/masks"
VAL_IMGS_DIR    = DATA_ROOT + "/validation_set/xrays"
VAL_MASKS_DIR   = DATA_ROOT + "/validation_set/masks"

BATCH_SIZE   = 10
NUM_EPOCHS   = 1000
NUM_CLASSES  = 33
DEVICE       = "cuda"
LR           = 1e-4
SHOW_EVERY   = 10
WEIGHTS_DIR  = "weights"

os.makedirs(WEIGHTS_DIR, exist_ok=True)

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=1,
    classes=NUM_CLASSES,
)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

train_dataset = DentalDataset(TRAIN_IMGS_DIR, TRAIN_MASKS_DIR, augment=True)
val_dataset   = DentalDataset(VAL_IMGS_DIR,   VAL_MASKS_DIR,   augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print("Starting training...")
for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0

    for i, (imgs, masks) in enumerate(train_loader):
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE).long()

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Showing training process
        if i % SHOW_EVERY == 0:
            print(f"[Epoch {epoch}] Batch {i}/{len(train_loader)} - Train Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0
    val_miou = 0
    val_mdice = 0
    val_prec = 0
    val_rec  = 0
    val_acc  = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE).long()

            preds = model(imgs)
            loss = criterion(preds, masks)

            val_loss += loss.item()

            _, miou = multiclass_iou(preds, masks, NUM_CLASSES)
            _, mdice = multiclass_dice(preds, masks, NUM_CLASSES)
            precs, recs = multiclass_precision_recall(preds, masks, NUM_CLASSES)
            acc = pixel_accuracy(preds, masks)

            val_miou += miou
            val_mdice += mdice
            val_prec += np.nanmean(precs)
            val_rec  += np.nanmean(recs)
            val_acc  += acc

    val_loss  /= len(val_loader)
    val_miou  /= len(val_loader)
    val_mdice /= len(val_loader)
    val_prec  /= len(val_loader)
    val_rec   /= len(val_loader)
    val_acc   /= len(val_loader)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| mIoU: {val_miou:.4f} "
        f"| Mean Dice(F1): {val_mdice:.4f} "
        f"| Precision: {val_prec:.4f} "
        f"| Recall: {val_rec:.4f} "
        f"| PixelAcc: {val_acc:.4f}"
    )

    torch.save(model.state_dict(), WEIGHTS_DIR + "/unet.pt")
    print("Saved checkpoint.")
