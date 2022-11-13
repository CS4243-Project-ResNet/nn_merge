from numpy import vstack, argmax
from my_dataset_1 import dataloaders
import numpy as np
import os
import os.path as path

def get_pose_dir(img_path):
    img_basename = path.basename(img_path)
    img_name = path.splitext(img_basename)[0]
    img_dirname = path.dirname(img_path)
    imgs_dir, class_n = path.split(img_dirname)
    dataset_dir = path.dirname(imgs_dir[:-1])
    dataset_dir = path.dirname(dataset_dir[:-1])
    pose_basename = img_name + '_keypoints.json'
    pose_dir = path.join(dataset_dir, 'pose', class_n, pose_basename)
    return pose_dir

def check(folder):
    imgs = []
    for subfolder in os.listdir(folder):
        subdir = os.path.join(folder, subfolder)
        imgs_now = os.listdir(subdir)
        imgs_labeled = list(map(lambda x: (os.path.join(subdir, x), subfolder),imgs_now))
        imgs = imgs + imgs_labeled

    img_paths = list(map(lambda x: x[0], imgs))

    pose_paths = list(map(lambda x: get_pose_dir(x), img_paths))
    print(len(pose_paths))
    for p in pose_paths:
        if not path.isfile(p):
            print(p)

check('/home/t/tianqi/CS4243_proj/dataset/splitted/train')
check('/home/t/tianqi/CS4243_proj/dataset/splitted/val')
check('/home/t/tianqi/CS4243_proj/dataset/splitted/test')


    
    