import torch 
import torch.nn
from torch.utils.data import Dataset, DataLoader
from utils.my_image import MyImage

class MyDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs # type MyImage

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        r_img = self.imgs[idx].img
        r_masked = self.imgs[idx].seg_masked
        r_pose = self.imgs[idx].pose
        r_weapon = self.imgs[idx].weapon_c
        r_id = self.imgs[idx].id
        return r_img, r_pose, r_id
        
loader = DataLoader()
