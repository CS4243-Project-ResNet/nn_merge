import torch
import torch.nn as nn
import time
import copy
from my_dataset_2 import MyDataset, data_transforms, image_datasets, dataloaders
from resnet import ResNet, BasicBlock

from torchvision import datasets, models, transforms
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics

#torch.multiprocessing.set_start_method('spawn')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}

from pose_mlp_new_1 import PoseMLP

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    mlppose_model = PoseMLP(201, 3)
