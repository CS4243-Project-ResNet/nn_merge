import os
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, CenterCrop, Normalize
from utils.my_image import MyImage
import utils.seg as seg
import utils.get_pose as pose
import utils.get_yolov5 as weapon

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
            imgs_labeled = map(lambda x: (x, class_dict[subfolder]),imgs_now)
            imgs = imgs + imgs_labeled
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        my_image = MyImage(img_path, label)
        r_img = self.transforms(my_image.img)
        r_masked = self.transforms(my_image.seg_masked)
        r_pose = my_image.pose
        r_weapon = my_image.weapon_c
        r_id = label
        return r_img, r_masked, r_pose, r_weapon, r_id
        
my_transforms = Compose([
    Resize((64, 64)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((.5, .5, .5), (.5, .5, .5))
])

my_dataset = "/home/t/tianqi/CS4243_proj/dataset/splitted"
my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True, drop_last=False)
