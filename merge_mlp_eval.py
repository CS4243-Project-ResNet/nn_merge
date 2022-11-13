from merge_mlp_4 import MergeMLP, PoseMLP, print_classification_report, device
from resnet import ResNet, BasicBlock
import torch

torch.multiprocessing.set_start_method('spawn')
my_resnet = ResNet(BasicBlock, [7, 7, 7], 3)
my_pose_mlp = PoseMLP(201, 3)
my_stack = MergeMLP(my_pose_mlp, my_resnet, 7).to(device)
my_stack.load_state_dict(torch.load('/home/t/tianqi/CS4243_proj/nn_merge/runs/merge_mlp_05_f.pt'))
print_classification_report(my_stack)
