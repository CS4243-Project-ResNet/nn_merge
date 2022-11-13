import os
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, CenterCrop, Normalize
import numpy as np


import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from my_utils.get_pose import get_pose_merge_s


class_dict = {
    "carrying": 0,
    "normal": 1,
    "threat": 2
}


class PoseDataset(Dataset):
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
        # print(img_path)
        pose = get_pose_merge_s(img_path)
        r_pose = np.pad(pose.astype(np.float32), (0, 201-len(pose)), 'constant')
        r_id = class_dict[label]
        return r_pose, r_id
        
data_transforms = Compose([
    Resize((64, 64)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((.5, .5, .5), (.5, .5, .5))
])

image_datasets = {
    'train': PoseDataset("/home/t/tianqi/CS4243_proj/dataset/splitted/train", data_transforms),
    'val': PoseDataset("/home/t/tianqi/CS4243_proj/dataset/splitted/val", data_transforms),
    'test': PoseDataset("/home/t/tianqi/CS4243_proj/dataset/splitted/test", data_transforms)
}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
# my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True, drop_last=False)
