from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
# import torch
# import torch.nn as nn
import numpy as np
from PIL import Image
import os
import glob


class FFHQDataset(Dataset):
    def __init__(self, files, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = files

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_part = masked_img[:, i : i + self.mask_size, i : i + self.mask_size]
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, masked_part, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if(self.mode=="train"):
            masked_img, aux = self.apply_random_mask(img)
            return img, masked_img, aux
        else:
            masked_img, masked_part, i = self.apply_center_mask(img)
            return img, masked_img, masked_part, i

        

    def __len__(self):
        return len(self.files)
    
if __name__ == "__main__":
    folders = sorted(list(os.listdir('StyleGAN.pytorch/ffhq')))[1:]
    allImages = []
    root = 'StyleGAN.pytorch/ffhq'
    for folder in folders:
        allImages.extend(sorted(glob.glob("%s/%s/*.png" %(root,folder))))
    
    transforms_ = [
        transforms.Resize((128, 128), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    trainDL = DataLoader(
        FFHQDataset(files=allImages, transforms_=transforms_, mode="train"),
        batch_size=12,
        shuffle=True,
        num_workers=1,
    )

    _, (img, masked_img, aux) = next(enumerate(trainDL))
    print(img.shape, masked_img.shape, aux.shape)