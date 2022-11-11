import torch 
import torch.nn
from torch.utils.data import Dataset, DataLoader, random_split
from utils.my_image import MyImage
import numpy as np

class MyDataset(Dataset):
    def __init__(self, imgs: list[MyImage]):
        self.imgs = imgs # type MyImage

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        r_img = self.imgs[idx].img
        r_masked = self.imgs[idx].seg_masked
        # r_pose = self.imgs[idx].pose
        r_pose = np.concatenate(self.imgs[idx].pose.values(), 1)
        r_weapon = self.imgs[idx].weapon_c
        r_id = self.imgs[idx].c_id
        return r_pose, r_id

    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.imgs) - test_size
        return random_split(self, [train_size, test_size])
        
loader = DataLoader()

