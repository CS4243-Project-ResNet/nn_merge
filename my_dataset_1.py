import os
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, CenterCrop, Normalize

import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from my_utils.my_image import MyImage
import my_utils.seg as seg
import my_utils.get_pose as pose
import my_utils.get_yolov5 as weapon

class_dict = {
    "carrying": 0,
    "normal": 1,
    "threat": 2
}

class MyDataset(Dataset):
    # insert folder of a certin split
    def __init__(self, folder, transforms):
        imgs = []
        for subfolder in os.listdir(folder):
            subdir = os.path.join(folder, subfolder)
            imgs_now = os.listdir(subdir)
            imgs_labeled = list(map(lambda x: (os.path.join(subdir, x), subfolder),imgs_now))
            imgs = imgs + imgs_labeled
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        # print(label)
        print(img_path)
        my_image = MyImage(img_path, label)
        r_masked = self.transforms(my_image.seg_masked)
        r_pose = my_image.pose
        r_weapon = my_image.weapon_c
        r_id = my_image.c_id
        return r_masked, r_pose, r_weapon, r_id
        
data_transforms = Compose([
    Resize((64, 64)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((.5, .5, .5), (.5, .5, .5))
])

image_datasets = {
    'train': MyDataset("/home/t/tianqi/CS4243_proj/dataset/splitted/train", data_transforms),
    'val': MyDataset("/home/t/tianqi/CS4243_proj/dataset/splitted/val", data_transforms),
    'test': MyDataset("/home/t/tianqi/CS4243_proj/dataset/splitted/test", data_transforms)
}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True)
              for x in ['train', 'val', 'test']}
# my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True, drop_last=False)
