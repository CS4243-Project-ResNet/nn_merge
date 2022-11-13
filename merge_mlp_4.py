import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report
import time
import copy
import wandb
import math
from my_dataset_1 import MyDataset, data_transforms, image_datasets, dataloaders
from resnet import ResNet, BasicBlock
cudnn.benchmark = True

class PoseMLP(nn.Module):
    def __init__(self, in_mlp, out_mlp):
        super(PoseMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_mlp, 2*in_mlp),
            nn.ELU(),
            nn.Linear(2*in_mlp, 3*in_mlp),
            nn.ELU(),
            nn.Linear(3*in_mlp, math.floor(1.5*in_mlp)),
            nn.ELU(),
            nn.Linear(math.floor(1.5*in_mlp), math.floor(in_mlp/2)),
            nn.ELU(),
            nn.Linear(math.floor(in_mlp/2), math.floor(in_mlp/4)),
            nn.ELU(),
            nn.Linear(math.floor(in_mlp/4), 4*in_mlp),
            nn.ELU(),
            nn.Linear(4*in_mlp, 2*out_mlp),
            nn.ELU(),
            nn.Linear(2*out_mlp, out_mlp),
            nn.ELU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class MergeMLP(nn.Module):
    def __init__(self, pose_mlp, seg_resnet, n_in, n_out=3):
        super(MergeMLP, self).__init__()
        self.pose_mlp = pose_mlp
        self.seg_resnet = seg_resnet
        self.stack_in_dim = n_in # pose_mlp.modules[-1].out_features + seg_resnet.modules[-1].out_features + 1
        self.stack_mlp = nn.Sequential(
            nn.Linear(self.stack_in_dim, math.floor(self.stack_in_dim / 2)),
            nn.ELU(),
            nn.Linear(math.floor(self.stack_in_dim/2), n_out),
        )
        self.shortcut = nn.Sequential()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pose, masked, weap):
        # print(pose.size())
        pose = self.pose_mlp(pose)
        # print(pose.size())
        masked = self.seg_resnet(masked)
        print(masked.size())
        weap = torch.unsqueeze(weap, 1)
        print(weap.size())
        x4 = torch.cat((pose, masked, weap), 1)
        x5 = self.stack_mlp(x4)
        x5 += self.shortcut(pose)
        x5 += self.shortcut(masked)
        x6 = self.softmax(x5)
        return x6


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=150):
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
            for imgs, poses, weapons, labels in dataloaders[phase]:
                imgs = imgs.to(device)
                poses = poses.to(device)
                weapons = weapons.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                print(poses.size())
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(poses, imgs, weapons)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                wandb.log({
                    'Train Loss': epoch_loss,
                    'Train Acc': epoch_acc
                })
            elif phase == 'val':
                wandb.log({
                    'Val Acc': epoch_acc
                })
            else:
                wandb.log({
                    'Test Acc': epoch_acc
                })

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "runs/merge_mlp_05_f.pt")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def print_classification_report(model_ft):
    y_pred = []
    y_true = []

    model_ft.eval()

    for poses, imgs, weapons, labels in dataloaders['test']:
        imgs = imgs.to(device)
        poses = poses.to(device)
        weapons = weapons.to(device)
        labels = labels.to(device)

        outputs = model_ft(imgs, poses, weapons)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.data.cpu())
        # print(y_pred)
        y_true.extend(labels.data.cpu())
        # print(y_true)
    print(classification_report(y_true, y_pred, labels=[0,1,2]))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    lr = 2.5e-4
    mom = 0.85
    step_size = 7
    gamma = 0.1
    config={
        'optimizer': 'SGD',
        'lr': lr,
        'momentum': mom,
        'w_decay_step': step_size,
        'gamma': gamma
    }
    wandb.init(project="cs4243_proj_merge", entity="tz02", config=config)
    criterion = nn.CrossEntropyLoss()
    my_resnet = ResNet(BasicBlock, [7, 7, 7], 3)
    my_pose_mlp = PoseMLP(201, 3)
    my_stack = MergeMLP(my_pose_mlp, my_resnet, 7)
    # my_stack.load_state_dict(torch.load('/home/t/tianqi/CS4243_proj/nn_merge/runs/merge_mlp_01.pt'))
    my_stack.to(device)
    my_optim = optim.SGD(my_stack.parameters(), lr, mom)
    exp_lr_scheduler = lr_scheduler.StepLR(my_optim, step_size, gamma)
    model_ft = train_model(my_stack, criterion, my_optim, exp_lr_scheduler,
                       num_epochs=100)
    # print_classification_report(model_ft)
