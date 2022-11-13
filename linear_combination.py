# copy from ipynb file

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

# torch.multiprocessing.set_start_method('spawn')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}

from pose_mlp_new_2 import PoseMLP
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. MLP(Pose)
mlppose_model = PoseMLP(201, 3)
mlppose_model.load_state_dict(torch.load("/home/t/tianqi/CS4243_proj/nn_merge/pose_mlp_1.pt"))
mlppose_model = mlppose_model.to(device)
mlppose_model.eval()

# 2. SegResNet(img_masked) 
segresnet_model = models.resnet50(pretrained=True)
segresnet_model.fc = nn.Linear(segresnet_model.fc.in_features, 3)
segresnet_model.load_state_dict(torch.load("../baseline/seg_resnet50.pt"))
segresnet_model.to(device)
segresnet_model.eval()

# 3. ResNet(img_raw) 
resnet_model = models.resnet50()
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 3)
checkpoint = torch.load("../baseline/resnet50.pt")
resnet_model.load_state_dict(checkpoint, strict=False)
resnet_model.to(device)
resnet_model.eval()

# Final Decision
errors = []
for x in [0.1, 0.2, 0.3, 0.4]:
    for y in [0.1, 0.2, 0.3, 0.4]:
        square_error = 0.0
        # for phase in ['train', 'val']:
        for phase in ['train']:
            for imgs, masked, pose, label in dataloaders[phase]:
                imgs = imgs.to(device)
                masked = masked.to(device)
                pose = pose.to(device)
                # weapon = weapon.to(device)
                label = label.to(device)

                mlppose_data = mlppose_model(pose) #TODO
                segresnet_data = segresnet_model(masked)
                resnet_data = resnet_model(imgs)
                
                mm,ss,rr = torch.nn.functional.softmax(mlppose_data), torch.nn.functional.softmax(segresnet_data), torch.nn.functional.softmax(resnet_data)
                score = mm * x + ss * y + rr * (1-x-y)
                # _, preds = torch.max(score, 1)

                label = label.item()
                score = score.tolist()[0]
                score = np.array(score)
                l = np.array([0, 0, 0])
                l[label] = 1
                error = sum((score-l)**2)
                square_error += error

                
        errors.append(square_error)
        print(f'x={x}, y={y}, sum_square_error={square_error}')


# We manually set the parameter x and y in the linear combination,
# hence both 'val' and 'test' dataset can be used in
from sklearn.metrics import classification_report
y_true = []
y_pred = []
x, y = 0.1, 0.2
for phase in ['val', 'test']:
    for imgs, masked, pose, label in dataloaders[phase]:
                imgs = imgs.to(device)
                masked = masked.to(device)
                pose = pose.to(device)
                # weapon = weapon.to(device)
                label = label.to(device)

                mlppose_data = mlppose_model(pose)
                segresnet_data = segresnet_model(masked)
                resnet_data = resnet_model(imgs)
                
                mm,ss,rr = torch.nn.functional.softmax(mlppose_data), torch.nn.functional.softmax(segresnet_data), torch.nn.functional.softmax(resnet_data)
                score = mm * x + ss * y + rr * (1-x-y)

                label = label.item()
                score = score.tolist()[0]
                score = np.array(score)
                l = np.array([0, 0, 0])
                l[label] = 1
                y_pred.append(np.argmax(score))
                y_true.append(np.argmax(l))

# Print Performance
target_names = ['Carrying', 'Normal', 'Threat']
print(classification_report(y_true, y_pred, target_names=target_names))