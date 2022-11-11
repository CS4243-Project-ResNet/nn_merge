import torch
import torch.nn as nn
import time
import copy
from my_dataset_1 import MyDataset, data_transforms, image_datasets, dataloaders
from resnet import ResNet, BasicBlock

class PoseMLP(nn.Module):
    def __init__(self, in_mlp, out_mlp):
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

class MergeMLP(nn.Module):
    def __init__(self, pose_mlp, seg_resnet, n_out=3):
        self.pose_mlp = pose_mlp
        self.seg_resnet = seg_resnet
        self.stack_in_dim = pose_mlp.modules[-1].out_features + seg_resnet.modules[-1].out_features + 1
        self.stack_mlp = nn.Sequential(
            nn.Linear(self.stack_in_dim, 2*self.stack_in_dim),
            nn.ELU(),
            nn.Linear(2*self.stack_in_dim, 2*n_out),
            nn.ELU(),
            nn.Linear(2*n_out, n_out),
            nn.Softmax()
        )

    def forward(self, pose, masked, weap):
        pose = self.pose_mlp(pose)
        masked = self.seg_resnet(masked)
        x4 = torch.cat((pose, masked, weap), 0)
        x5 = self.stack_mlp(x4)
        return x5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


criterion = nn.CrossEntropyLoss()
my_resnet = ResNet(BasicBlock, [5, 5, 5], 3)
my_pose_mlp = PoseMLP(201, 3)
my_stack = MergeMLP(my_pose_mlp, my_resnet)
train_model()
