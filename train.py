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

import PIL.Image as Image
import os
import glob
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='training Params')
parser.add_argument('--checkpoint', default=None, help='restore training from checkpoint', type=str)
parser.add_argument('--epochs', default=25, help='number of training epochs', type=int)

args = parser.parse_args()


writer = SummaryWriter()

folders = sorted(list(os.listdir('inpainting_gmcnn/pytorch/ffhq')))[1:]
allImages = []
root = 'inpainting_gmcnn/pytorch/ffhq'
for folder in folders:
    allImages.extend(sorted(glob.glob("%s/%s/*.png" %(root,folder))))

# allImages, _ = train_test_split(allImages, test_size=0.25, random_state=42) # training on 75% data
    
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
        
def display_progress(cond, fake, real, epoch, figsize=(10,5)):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(cond)
    ax[2].imshow(fake)
    ax[1].imshow(real)
    plt.savefig("results/progress_" + str(epoch) + ".png")
    # plt.show()

gen = Generator(3, 3).to(device)
dis = PatchGAN(3 + 3).to(device)
gen = gen.apply(_weights_init)
patch_gan = dis.apply(_weights_init)
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
pixelwise_loss = torch.nn.MSELoss().to(device)

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
    lambda_recon = 100

    return adver_loss + lambda_recon * recon_loss

def _disc_step(real_images, conditioned_images):
    fake_images = gen(conditioned_images).detach()
    fake_logits = patch_gan(fake_images, conditioned_images)

    real_logits = patch_gan(real_images, conditioned_images)
    # print(fake_logits.shape, real_logits.shape)

    fake_loss = adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
    real_loss = adversarial_loss(real_logits, torch.ones_like(real_logits))
    return (real_loss + fake_loss) / 2

gen_loss_epoch = []
dis_loss_epoch = []
best_val_loss = np.inf

epoch = 0
trainIter = 0
valIter = 0
if(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    gen.load_state_dict(checkpoint['gen'])
    dis.load_state_dict(checkpoint['dis'])
    opt_D.load_state_dict(checkpoint['optD'])
    opt_G.load_state_dict(checkpoint['optG'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_loss']

for e in range(epoch +1, args.epochs):
    print("Epoch %d"%(e))
    gen_loss = 0
    dis_loss = 0
    gen.train()
    patch_gan.train()
    for step, (data) in enumerate(tqdm(trainDL)):
        trainIter += 1
        real, conditional, _ = data
        real = real.to(device)
        conditional = conditional.to(device)

        opt_G.zero_grad()
        loss = _gen_step(real, conditional)
        gen_loss += loss.item()
        loss.backward()
        opt_G.step()

        writer.add_scalar("Generator loss/train_iteration", loss.item(), trainIter)

        opt_D.zero_grad()
        loss = _disc_step(real, conditional)
        dis_loss += loss.item()
        loss.backward()
        opt_D.step()
        writer.add_scalar("Discriminator loss/train_iteration", loss.item(), trainIter)
               
        # print("step loss: {:.4f}, dis loss: {:.4f}".format(gen_loss/(step+1), dis_loss/(step+1)))
    
    print("epoch gen loss: {:.4f}, dis loss: {:.4f}".format(gen_loss/len(trainDL), dis_loss/len(trainDL)))
    writer.add_scalar("Epoch loss train Generator/epoch", gen_loss/len(trainDL), e)
    writer.add_scalar("Epoch loss train Discriminator/epoch", dis_loss/len(trainDL), e)

    gen_loss_epoch.append(gen_loss/len(trainDL))
    dis_loss_epoch.append(dis_loss/len(trainDL))

    with torch.no_grad():
        gen_loss = 0
        dis_loss = 0
        gen.eval()
        patch_gan.eval()
        for step, (data) in enumerate(tqdm(valDL)):
            valIter += 1
            real, conditional, _, _ = data
            real = real.to(device)
            conditional = conditional.to(device)

            loss = _gen_step(real, conditional)
            gen_loss += loss.item()

            writer.add_scalar("Generator loss/val_iteration", loss.item(), valIter)

            loss = _disc_step(real, conditional)
            dis_loss += loss.item()

            writer.add_scalar("Discriminator loss/val_iteration", loss.item(), valIter)

            # print("step val loss: {:.4f}, dis loss: {:.4f}".format(gen_loss/(step+1), dis_loss/(step+1)))
        
        print("step val loss: {:.4f}, dis loss: {:.4f}".format(gen_loss/len(valDL), dis_loss/len(valDL)))
        writer.add_scalar("Epoch loss val Generator/epoch", gen_loss/len(valDL), e)
        writer.add_scalar("Epoch loss val Discriminator/epoch", dis_loss/len(valDL), e)
        if((gen_loss/len(valDL)) < best_val_loss):
            best_val_loss = gen_loss/len(valDL)
            torch.save({
                "gen": gen.state_dict(),
                "dis": dis.state_dict(),
                "epoch": e,
                "optD": opt_D.state_dict(),
                "optG": opt_G.state_dict(),
                "best_loss": best_val_loss,
            }, "checkpoint_100.ckpt")

    if(e%1 == 0):
        _, (real, conditional, _, _) = next(enumerate(testDL))
        conditional = conditional.to(device)
        real = real.to(device)
        fake = gen(conditional).detach()
        display_progress(conditional[0], fake[0], real[0], e)