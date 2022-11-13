from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import my_dataset_1


cudnn.benchmark = True

image_datasets = my_dataset_1.image_datasets
dataloaders = my_dataset_1.dataloaders
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = dataloaders['train']
batch = next(iter(train_loader))
print(batch[1])
print(batch[1].size())
