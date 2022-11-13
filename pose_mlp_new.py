from numpy import vstack, argmax
from sklearn.metrics import accuracy_score
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from my_dataset_1 import dataloaders
import torch.nn as nn
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f = open("pose_mlp.out", "a")


class PoseMLP(nn.Module):
    def __init__(self, in_mlp, out_mlp):
        super(PoseMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_mlp, 2*in_mlp),
            nn.ELU(),
            nn.Linear(2*in_mlp, 3*in_mlp),
            nn.ELU(),
            nn.Linear(3*in_mlp, 2*out_mlp),
            nn.ELU(),
            nn.Linear(2*out_mlp, out_mlp)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.01)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        # r_masked, r_pose, r_weapon, r_id
        for _, inputs, _, targets in train_dl:
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            # print(inputs)
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        print(f'Epoch {epoch}: Loss')

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for _, inputs, _, targets in test_dl:
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# prepare the data
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}

# define the network
model = PoseMLP(201, 3)
# train the model
train_model(dataloaders['train'], model)
# evaluate the model
acc = evaluate_model(dataloaders['test'], model)
print('Accuracy: %.3f' % acc)

torch.save(model.state_dict(), "pose_mlp.pt")
