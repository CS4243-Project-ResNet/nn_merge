from models import PoseMLP
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PoseMLP(201, 3)
model.load_state_dict(torch.load("/home/t/tianqi/CS4243_proj/nn_merge/pose_mlp_1.pt"))
model = model.to(device)
model.eval()