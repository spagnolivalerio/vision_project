import torch
import torch.optim as opt
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import SegmentationDataset, OODSegmentationDataset, UnsupervisedImageDataset
from utils import crop_and_normalize, crop, vflip, rotate_left, rotate_right
from torch.utils.data import DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os

# ======================
# CONFIGURAZIONE BASE
# ======================
weights_root = "weights/"
os.makedirs(weights_root, exist_ok=True)

dataroot = "../data/dentex/training_data/quadrant_enumeration"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
lr = 1e-4
num_epochs_pretrain = 20      # Pretraining
num_epochs_finetune = 80      # Fine-tuning
display_every = 20

# ======================
# MODELLO: U-NET ResNet18
# ======================
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,   # nessun pretraining ImageNet
    in_channels=1,
    classes=1,
    activation=None,
    decoder_use_batchnorm=True
).to(device)

# ======================
# 1️⃣ PRETRAINING NON SUPERVISIONATO (AUTOENCODER)
# ======================
def pretrain_autoencoder(model, dataloader, device, epochs=10, lr=1e-4):
    """
    Pre-addestra solo l'encoder della U-Net come autoencoder
    sulle immagini GAN (senza maschere).
    """
    print("🚀 Avvio pretraining non supervisionato (autoencoder)...")

    # Congela il decoder, allena solo l'encoder
    for param in model.decoder.parameters():
        param.requires_grad = False

    optimizer = opt.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs in dataloader:  # immagini GAN, maschere ignorate
            imgs = imgs.to(device)
            optimizer.zero_grad()

            # output = ricostruzione (mask non serve)
            preds = model(imgs)
            loss = criterion(torch.sigmoid(preds), imgs)  # confronto immagine-ricostruzione

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Pretrain] Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

    print("✅ Pretraining completato.\n")

    # Sblocca il decoder per il fine-tuning successivo
    for param in model.decoder.parameters():
        param.requires_grad = True


# ======================
# 2️⃣ FINE-TUNING SUPERVISIONATO
# ======================
def finetune_segmentation(model, dataloader, device, epochs=80, lr=1e-4, display_every=20):
    # Definizione delle loss
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()

    # Funzione combinata: somma Dice + BCE
    def criterion(pred, target):
        return dice_loss(pred, target) + bce_loss(pred, target)

    optimizer = opt.Adam(model.parameters(), lr=lr)

    print("🎯 Avvio fine-tuning supervisionato sulla segmentazione...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (imgs_batch, masks_batch, _) in enumerate(dataloader):
            imgs_batch = imgs_batch.to(device)
            masks_batch = masks_batch.to(device)

            optimizer.zero_grad()
            output = model(imgs_batch)
            loss = criterion(output, masks_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(
                    f"[Fine-tune] Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)-1} "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"🧾 Epoch [{epoch+1}/{epochs}] Loss media: {avg_loss:.4f}")

    print("✅ Fine-tuning completato.\n")

    # Salvataggio del modello
    save_path = os.path.join(weights_root, "seg_cnn_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Modello salvato in: {save_path}")



# ======================
# VISUALIZZAZIONE
# ======================
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


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    # Dataset per pretraining (immagini GAN, senza maschere)
    gan_dataroot = "syn_dataset/xrays"  # ✅ percorso corretto

    if os.path.exists(gan_dataroot):
        gan_dataset = UnsupervisedImageDataset(
            gan_dataroot,
            transform=crop_and_normalize
        )
        gan_dataloader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True)
        pretrain_autoencoder(model, gan_dataloader, device, epochs=num_epochs_pretrain, lr=lr)
        torch.save(model.state_dict(), os.path.join(weights_root, "pretrained_autoencoder.pth"))
        print("💾 Pesi del pretraining salvati.\n")
    else:
        print(f"⚠️ Cartella '{gan_dataroot}' non trovata, salto pretraining.")

    # Dataset per fine-tuning supervisionato (dataset reale)
    ood_seg_dataset = OODSegmentationDataset(dataroot, img_transform=crop_and_normalize, mask_transform=crop)
    labels = [int(is_ood) for _, _, is_ood in ood_seg_dataset]

    num_id = sum(1 for x in labels if x == 0)
    num_ood = sum(1 for x in labels if x == 1)

    weight_for_id = 1.0 / num_id
    weight_for_ood = 1.0 / num_ood
    sample_weights = [weight_for_ood if lbl == 1 else weight_for_id for lbl in labels]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    dataloader = DataLoader(ood_seg_dataset, batch_size=batch_size, sampler=sampler)

    # Fine-tuning del modello
    finetune_segmentation(model, dataloader, device, epochs=num_epochs_finetune, lr=lr)
