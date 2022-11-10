import torch
import torch.nn as nn

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


class StackModel(nn.Module):
    def __init__(self, in_mlp, out_mlp, in_res, out_res, out):
        self.mlp = nn.Sequential(
            nn.Linear(in_mlp, 2*in_mlp),
            nn.ELU(),
            nn.Linear(2*in_mlp, 3*in_mlp),
            nn.ELU(),
            nn.Linear(3*in_mlp, 2*out_mlp),
            nn.ELU(),
            nn.Linear(2*out_mlp, out_mlp)
        )
        self.res =
        self.last = nn.Linear(out_mlp + out_res, out)

    def forward(self, x1, x2):
        x1 = self.fc(x1)
        x2 = self.res(x2)
        
