# ============= imports =============
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from pix2pix import Generator
from ffhqDataset import ImageDataset
import numpy as np
from tqdm import tqdm

import PIL.Image as Image
import os
import glob
from sklearn.model_selection import train_test_split
import argparse
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
import sewar
import cv2

parser = argparse.ArgumentParser(description='training Params')
parser.add_argument('--checkpoint', default=None, help='restore training from checkpoint', type=str, required=True)
parser.add_argument('--partition', default=100, help='share of training data', type=int)
parser.add_argument('--dataset', default="ffhq", help='dataset to run on', type=str, choices=["ffhq", "artbench"])
args = parser.parse_args()

if(args.dataset == "ffhq"):
    folders = sorted(list(os.listdir('ffhq')))[1:]
    allImages = []
    root = 'ffhq'
    for folder in folders:
        allImages.extend(sorted(glob.glob("%s/%s/*.png" %(root,folder))))
elif(args.dataset == "artbench"):
    folders = sorted(list(os.listdir('artbench-10-imagefolder-split')))[1:]
    paintingTypes = sorted(list(os.listdir('artbench-10-imagefolder-split/train')))[1:]
    allImages = []
    root = 'artbench-10-imagefolder-split'
    for painting in paintingTypes:
        for folder in folders:
            allImages.extend(sorted(glob.glob("%s/%s/%s/*.jpg" %(root,folder,painting))))
   
_, testImage = train_test_split(allImages, test_size=0.2, random_state=42)
_, testImage = train_test_split(testImage, test_size=0.5, random_state=42)

trans = [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
testData = ImageDataset(files=testImage, transforms_=trans, mode="test")
testDL = DataLoader(testData, batch_size=32, shuffle=False)

device = 'cpu'
if(torch.cuda.is_available()):
    device='cuda'
    
print(device)

gen = Generator(3, 3).to(device)

checkpoint = torch.load(args.checkpoint)
gen.load_state_dict(checkpoint['gen'])


# =================== FID and IS ===================
fid_metric = FID(device=device)
is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])

def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((128, 128), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

def evaluation_step(engine, batch):
    with torch.no_grad():
        gen.eval()
        real, conditional, _, _ = batch
        conditional = conditional.to(device)
        generated = gen(conditional)
        fake = interpolate(generated.cpu())
        real = interpolate(real)
        return fake, real
    
evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

evaluator.run(testDL, max_epochs=1)
metrics = evaluator.state.metrics
fid_score = metrics['fid']
is_score = metrics['is']
print("FID Score: {:.4f}".format(fid_score))
print("Inception Score: {:.4f}".format(is_score))

# =================== PSNR and SSIM ===================
def normalize_img(img):
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

with torch.no_grad():
    gen.eval()
    psnr_val = 0
    ssim_val = 0
    for step, (batch) in enumerate(tqdm(testDL)):
        real, conditional, _, _ = batch
        real = real.to(device)
        conditional = conditional.to(device)

        generatedImages = gen(conditional)
        for i in range(real.shape[0]):
            pred = normalize_img(generatedImages[i].permute(1,2,0).cpu().numpy())
            gt = normalize_img(real[i].permute(1,2,0).cpu().numpy())

            psnr_val += sewar.psnr(pred, gt)
            ssim_val += sewar.ssim(pred, gt)[0]
        if(step % 100 == 0):
            sampleImage = torch.cat((real.data, conditional.data, generatedImages.data), -2)
            save_image(sampleImage, "images/%d_%d.png"%(step, args.partition), nrow=6, normalize=True)
    print("mean psnr: {:.4f}, mean ssim: {:.4f}".format(psnr_val/len(testImage), ssim_val/len(testImage)))

