import numpy as np
import os
import glob
import argparse, random
import sewar

import torch
from torch.autograd import Variable
 
from torchvision.utils import save_image

from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from CEGAN import Generator, Discriminator
from ffhqDataset import ImageDataset
import cv2

random.seed(42)
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='training Params')
parser.add_argument('--checkpoint', default=None, help='restore training from checkpoint', type=str)
parser.add_argument('--epochs', default=25, help='number of training epochs', type=int)
parser.add_argument('--partition', default=25, help='dataset partition', type=int)

args = parser.parse_args()


folders = sorted(list(os.listdir('ffhq')))[1:]
allImages = []
root = 'ffhq'
for folder in folders:
    allImages.extend(sorted(glob.glob("%s/%s/*.png" %(root,folder))))

allImages, _ = train_test_split(allImages, test_size=args.partition/100, random_state=42) # training on 75% data
trainImage, testImage = train_test_split(allImages, test_size=0.2, random_state=42)
valImage, testImage = train_test_split(testImage, test_size=0.5, random_state=42)

device = 'cpu'
if(torch.cuda.is_available()):
    device='cuda'
    
print(device)

transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
trainDL = DataLoader(
    ImageDataset(files=trainImage, transforms_=transforms_, mode="train"),
    batch_size=12,
    shuffle=True
)
valDL = DataLoader(
    ImageDataset(files=valImage, transforms_=transforms_, mode="train"),
    batch_size=12,
    shuffle=True
)
testDL = DataLoader(
    ImageDataset(files=testImage, transforms_=transforms_, mode="test"),
    batch_size=12,
    shuffle=False
)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=3).to(device)
discriminator = Discriminator(channels=3).to(device)


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

patch_h, patch_w = int(64 / 2 ** 3), int(64 / 2 ** 3)
patch = (1, patch_h, patch_w)

e = 0
best_loss = np.inf

if(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    best_loss = checkpoint['loss']
    e = checkpoint['epoch']

Tensor = torch.FloatTensor

gen_adv_losses, gen_pixel_losses, disc_losses, counter = [], [], [], []


for epoch in range(e+1, args.epochs):
    
    ### Training ###
    generator.train()
    discriminator.train()
    gen_adv_loss, gen_pixel_loss, disc_loss = 0, 0, 0
    for i, (imgs, masked_imgs, masked_parts) in enumerate(tqdm(trainDL)):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        ## Train Generator ##
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_parts = generator(masked_imgs)

        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        # print(gen_parts.shape, masked_parts.shape)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        ## Train Discriminator ##
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
        
        gen_adv_loss, gen_pixel_loss, disc_loss
        gen_adv_losses, gen_pixel_losses, disc_losses, counter
        
        gen_adv_loss += g_adv.item()
        gen_pixel_loss += g_pixel.item()
        gen_adv_losses.append(g_adv.item())
        gen_pixel_losses.append(g_pixel.item())
        disc_loss += d_loss.item()
        disc_losses.append(d_loss.item())
        counter.append(i*12 + imgs.size(0) + epoch*len(trainDL.dataset))
        # tqdm_bar.set_postfix(gen_adv_loss=gen_adv_loss/(i+1), gen_pixel_loss=gen_pixel_loss/(i+1), disc_loss=disc_loss/(i+1))
        
        # Generate sample at sample interval
        batches_done = epoch * len(trainDL) + i
    
    with torch.no_grad():
        generator.eval()
        discriminator.eval()
        adv_val_loss, pixel_val_loss, disc_val_loss = 0, 0, 0
        for i, (imgs, masked_imgs, masked_parts) in enumerate(tqdm(valDL)):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Generate a batch of images
            gen_parts = generator(masked_imgs)

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            g_pixel = pixelwise_loss(gen_parts, masked_parts)
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

        
            adv_val_loss += g_adv.item()
            pixel_val_loss += g_pixel.item()
            disc_val_loss += d_loss.item()
        print("Epoch: %d, val adv loss: %f, val pixel loss: %f, val disc loss: %f"%(epoch, adv_val_loss, pixel_val_loss, disc_val_loss))

        if((0.001 * adv_val_loss + 0.999 * pixel_val_loss) < best_loss):
            best_loss = 0.001 * adv_val_loss + 0.999 * pixel_val_loss
            PATH = 'saved_models/checkpoint_0.75_data.ckpt'
            torch.save({
                       'generator': generator.state_dict(),
                       'discriminator': discriminator.state_dict(),
                       'optimizer_G':optimizer_G.state_dict(),
                       'optimizer_D':optimizer_D.state_dict(),
                       'epoch': epoch,
                       'loss': best_loss
                       }, PATH)

def normalize_img(img):
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

with torch.no_grad():
    checkpoint = torch.load('saved_models/checkpoint_0.75_data.ckpt')
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    generator.eval()
    discriminator.eval()

    pixel_test_loss = 0
    psnr_val = 0
    ssim_val = 0
    for step, (imgs, masked_imgs, masked_parts, i) in enumerate(tqdm(testDL)):

        samples = Variable(imgs.type(Tensor))
        masked_samples = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))
        i = i[0].item()  # Upper-left coordinate of mask
        # Generate inpainted image
        gen_mask = generator(masked_samples)
    
        g_pixel = pixelwise_loss(gen_mask, masked_parts)
        # print(gen_mask.shape, coords)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + 64, i : i + 64] = gen_mask
        # Save sample
        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        
        for i in range(imgs.shape[0]):
            pred = normalize_img(filled_samples[i].permute(1,2,0).cpu().numpy())
            gt = normalize_img(samples[i].permute(1,2,0).cpu().numpy())
            psnr_score = sewar.psnr(pred,gt)
            #ssim
            ssim_score = sewar.ssim(pred,gt)[0]
            psnr_val += psnr_score
            ssim_val += ssim_score
        
        pixel_test_loss += g_pixel.item()
        if(step % 100 == 0):
            save_image(sample, "images/%d_75.png" % step, nrow=6, normalize=True)
    print("final pixel loss: {:.4f}".format(pixel_test_loss/int(len(testDL))))
    print("mean psnr: {:.4f}, mean ssim: {:.4f}".format(psnr_val/len(testImage), ssim_val/len(testImage)))

