import torch
import torch.nn as nn
import 

class MergeMLP(nn.Module):
    def __init__(self, pose_mlp, seg_resnet):
        self.pose_mlp = pose_mlp
        self.seg_resnet = seg_resnet
        self.stack_mlp = nn.Sequential(
            
        )
