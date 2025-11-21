import torch
import torch.optim as opt
import torch.nn as nn 
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import SegmentationDataset, OODSegmentationDataset 
from utils import crop_and_normalize, crop, vflip, rotate_left, rotate_right
from torch.utils.data import DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os

weights_root = "weights/"

dataroot = "../data/dentex/training_data/quadrant_enumeration"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50_model = deeplabv3_resnet50(weights=None, num_classes=4)
resnet50_model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet50_loss = nn.CrossEntropyLoss()

resnet18_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1, activation=None)
bce = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode='binary')

def bce_dice_loss(preds, targets):
    return bce(preds, targets) + dice(preds, targets)

batch_size = 16
lr = 1e-4
num_epochs = 100
display_every = 20

model = resnet18_model
criterion = bce_dice_loss
optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

seg_dataset = SegmentationDataset(dataroot, img_transform=crop_and_normalize, mask_transform=crop)
dataloader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=True)


"""ood_seg_dataset = OODSegmentationDataset(dataroot, img_transform=crop_and_normalize, mask_transform=crop)
labels = [int(is_ood) for _, _, is_ood in ood_seg_dataset]

num_id = sum(1 for x in labels if x == 0)
num_ood = sum(1 for x in labels if x == 1)

weight_for_id = 1.0 / num_id
weight_for_ood = 1.0 / num_ood

sample_weights = [weight_for_ood if lbl == 1 else weight_for_id for lbl in labels]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

dataloader = DataLoader(ood_seg_dataset, batch_size=32, sampler=sampler)"""

model = model.to(device)

def visualize_batch(imgs, masks, preds):
    imgs = imgs.cpu().squeeze(1)
    masks = masks.cpu().squeeze(1)
    preds = torch.sigmoid(preds).cpu().squeeze(1)

    plt.figure(figsize=(12, 4))
    for i in range(min(3, imgs.size(0))):  # mostra max 3 esempi
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.title("Input")
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(masks[i], cmap="gray")
        plt.title("Mask GT")
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(imgs[i], cmap="gray")
        plt.imshow((preds[i] > 0.5), cmap="Reds", alpha=0.4)
        plt.title("Predizione")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (imgs_batch, masks_batch) in enumerate(dataloader):

            imgs_batch = imgs_batch.to(device)
            masks_batch = masks_batch.to(device)

            optimizer.zero_grad()

            output = model(imgs_batch)

            loss = criterion(output, masks_batch)

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader) - 1} "
                    f"Loss: {loss.item():.4f}"
                )
            
            """if batch_idx % display_every == 0 and batch_idx > 0:
                model.eval()
                with torch.no_grad():
                    preds = model(imgs_batch)
                visualize_batch(imgs_batch, masks_batch, preds)
                model.train()"""

    torch.save(model.state_dict(), weights_root + "seg_cnn.pth")