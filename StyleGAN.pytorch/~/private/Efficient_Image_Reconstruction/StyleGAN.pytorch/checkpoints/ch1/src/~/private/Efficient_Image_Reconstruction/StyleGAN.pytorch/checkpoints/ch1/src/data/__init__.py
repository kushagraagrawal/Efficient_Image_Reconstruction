"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""

from torchvision.datasets import ImageFolder
import pandas as pd
import torch
import os
from PIL import Image, ImageOps
import numpy as np
from data.datasets import FlatDirectoryImageDataset, FoldersDistributedDataset
from data.transforms import get_transform
import torchvision.transforms
from torchvision.transforms import Compose, ToTensor, Resize
import random
from torch.utils.data import Dataset
mlb = {'Aorticenlargement': np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),       
       'Atelectasis': np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),         
       'Calcification': np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),       
       'Cardiomegaly': np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),        
       'Consolidation': np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),       
       'ILD': np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),                 
       'Infiltration': np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),        
       'LungOpacity': np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),         
       'Nofinding': np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),          
       'NoduleMass': np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),         
       'Otherlesion': np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),        
       'Pleuraleffusion': np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),    
       'Pleuralthickening': np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),  
       'Pneumothorax': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),       
       'Pulmonaryfibrosis': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])} 
class AugmentationDatasetEmbedded(Dataset):
    def __init__(self, root_dir, aug_size=-1, res=(1024, 1024),):

        self.root_dir = root_dir
        self.res = res
        self.mask_path = os.listdir(f"{self.root_dir}/masks/")
        random.shuffle(self.mask_path)
        self.mask_path = self.mask_path[:aug_size]
        train_csv = pd.read_csv('/home/jessica/labelGAN/downstream_tasks/vinbig/train.csv')
        self.image_id_to_labels = train_csv.groupby(by="image_id").class_name.apply(list).apply(lambda x: np.unique([elem.replace(" ", "").replace("/", "") for elem in x]))

    def __len__(self):
        return len(self.mask_path)

    def __getitem__(self, idx):
 
        #Reading images
        imgname = self.mask_path[idx]
        imname = imgname.replace("_mask", "").replace(".jpg", ".png")
        img_name = os.path.join(self.root_dir, 'imgs', f"{imname}")
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image).convert('RGB')
        
        #Extracting disease label
        label_list = self.image_id_to_labels[imgname.replace(".png", "").replace(".jpg", "").replace("_mask", "")]
        labels = np.zeros(15).astype(int)
        for label in label_list:
            labels = labels | mlb[label]
        
        #Converting to tensors and Resizing images
        self.transform = Compose([ToTensor(), Resize(self.res)])
        
        image = self.transform(image)
        indices = [i for i, x in enumerate(labels) if x == 1]

        random_index = random.choice(indices)
        #image = torch.stack((torch.tensor(image),)*3, axis=0).squeeze()
        return image, torch.tensor(random_index)



def make_dataset(cfg, conditional=False):
    
    if conditional:
        dataset = AugmentationDatasetEmbedded(root_dir='/data1/shared/jessica/drive_data/train_images_embedded/')#ImageFolder
        return dataset
    else:
        if cfg.folder:
            Dataset = FoldersDistributedDataset 
        else:
            Dataset = FlatDirectoryImageDataset
    
    transforms = get_transform(new_size=(cfg.resolution, cfg.resolution))
    _dataset = Dataset(cfg.img_dir, transform=transforms)

    return _dataset


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => data_loader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    return dl
