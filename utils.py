import torch
import torchvision
from dataset import StrawberryDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tr_F
import cv2
import numpy as np
from PIL import Image


def save_checkpoint(state, filename="model.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform,
                num_workers=4, pin_memory=True):
    train_ds = StrawberryDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)

    val_ds = StrawberryDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def preprocess(img, size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = tr_F.resize(img, size)
    img = tr_F.to_tensor(img)
    t_img = tr_F.normalize(img, mean=[0.0, 0.0, 0.0],
                           std=[1.0, 1.0, 1.0])

    t_img.resize_(1, 3, size[0], size[1])
    return t_img.cuda()


def postprocess(image, preds, size):
    pred_mask = preds.cpu() * 255.
    reshaped_pred_mask = np.array(np.reshape(pred_mask, (size[0], size[1], 1)),
                                  dtype=np.uint8)
    colored_pred_mask = cv2.cvtColor(reshaped_pred_mask, cv2.COLOR_GRAY2RGB)
    final_pred_mask = cv2.resize(colored_pred_mask, (image.shape[1],
                                                     image.shape[0]), interpolation=cv2.INTER_NEAREST)
    output = np.hstack((image, final_pred_mask))
    return output, final_pred_mask
