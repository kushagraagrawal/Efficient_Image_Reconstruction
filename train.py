# ============= imports =============
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from pix2pix import Generator, PatchGAN
from ffhqDataset import FFHQDataset
import numpy as np
from tqdm import tqdm

from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
import PIL.Image as Image
import os
import glob
from sklearn.model_selection import train_test_split


writer = SummaryWriter()

folders = sorted(list(os.listdir('StyleGAN.pytorch/ffhq')))[1:]
allImages = []
root = 'StyleGAN.pytorch/ffhq'
for folder in folders:
    allImages.extend(sorted(glob.glob("%s/%s/*.png" %(root,folder))))

allImages, _ = train_test_split(allImages, test_size=0.75, random_state=42) # training on 75% data
    
trainImage, testImage = train_test_split(allImages, test_size=0.2, random_state=42)
valImage, testImage = train_test_split(testImage, test_size=0.5, random_state=42)

trans = [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
trainData = FFHQDataset(files=trainImage, transforms_=trans, mode="train")
valData = FFHQDataset(files=valImage, transforms_=trans, mode="val")
testData = FFHQDataset(files=testImage, transforms_=trans, mode="test")

trainDL = DataLoader(trainData, batch_size=32, shuffle=True)
valDL = DataLoader(valData, batch_size=8, shuffle=True)
testDL = DataLoader(testData, batch_size=8, shuffle=False)

device = 'cpu'
if(torch.cuda.is_available()):
    device='cuda'
    
print(device)

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
def display_progress(cond, fake, real, figsize=(10,5)):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(cond)
    ax[2].imshow(fake)
    ax[1].imshow(real)
    plt.show()

gen = Generator(3, 3).to(device)
dis = PatchGAN(3 + 3).to(device)
gen = gen.apply(_weights_init)
patch_gan = dis.apply(_weights_init)
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
pixelwise_loss = torch.nn.L1Loss().to(device)

opt_G = torch.optim.Adam(gen.parameters(), lr=1e-4)
opt_D = torch.optim.Adam(dis.parameters(), lr=1e-4)

def _gen_step(real_images, conditioned_images):
    # Pix2Pix has adversarial and a reconstruction loss
    # First calculate the adversarial loss
    fake_images = gen(conditioned_images)
    disc_logits = patch_gan(fake_images, conditioned_images)
    adver_loss = adversarial_loss(disc_logits, torch.ones_like(disc_logits))

    # calculate reconstruction loss
    recon_loss = pixelwise_loss(fake_images, real_images)
    lambda_recon = 200

    return adver_loss + lambda_recon * recon_loss

def _disc_step(real_images, conditioned_images):
    fake_images = gen(conditioned_images).detach()
    fake_logits = patch_gan(fake_images, conditioned_images)

    real_logits = patch_gan(real_images, conditioned_images)

    fake_loss = adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
    real_loss = adversarial_loss(real_logits, torch.ones_like(real_logits))
    return (real_loss + fake_loss) / 2

gen_loss_epoch = []
dis_loss_epoch = []
for e in range(20):
    gen_loss = 0
    dis_loss = 0
    gen.train()
    patch_gan.train()
    for step, (data) in enumerate(trainDL):
        real, conditional, _ = data
        real = real.to(device)
        conditional = conditional.to(device)

        opt_G.zero_grad()
        loss = _gen_step(real, conditional)
        gen_loss += loss.item()
        loss.backward()
        opt_G.step()
        
        opt_D.zero_grad()
        loss = _disc_step(real, conditional)
        dis_loss += loss.item()
        loss.backward()
        opt_D.step()
               
        print("step loss: {:.4f}, dis loss: {:.4f}".format(gen_loss/(step+1), dis_loss/(step+1)))
    
    print("epoch gen loss: {:.4f}, dis loss: {:.4f}".format(gen_loss/len(trainDL), dis_loss/len(trainDL)))
    gen_loss_epoch.append(gen_loss/len(trainDL))
    dis_loss_epoch.append(dis_loss/len(trainDL))
    if(e%1 == 0):
        gen.eval()
        patch_gan.eval()
        _, (real, conditional) = next(enumerate(testDL))
        conditional = conditional.to(device)
        real = real.to(device)
        fake = gen(conditional).detach()
        display_progress(conditional[0], fake[0], real[0])