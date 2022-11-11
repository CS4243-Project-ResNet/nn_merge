import torch
import torch.nn as nn
import TODO as detect_weapon

class MergeMLP(nn.Module):
    def __init__(self, pose_mlp, seg_resnet):
        self.pose_mlp = pose_mlp
        self.seg_resnet = seg_resnet
        self.stack_mlp = nn.Sequential(
            
        )

    def forward(self, x1, x2):
        x1 = self.pose_mlp(x1)
        x2 = self.seg_resnet(x2)
        x3 = detect_weapon(x2)
        x4 = 
