import os
import shutil
from os.path import join
import glob
import numpy as np

import torch
from torch import nn
from PIL import Image
import open3d as o3d
import clip

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def extract_clip_feature(labelset, model_name="ViT-B/32"):
    # "ViT-L/14@336px" # the big model that OpenSeg uses
    # pdb.set_trace()
    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")

    if isinstance(labelset, str):
        lines = labelset.split(',')
    elif isinstance(labelset, list):
        lines = labelset
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

def extract_text_feature(labelset):
    '''extract CLIP text features.'''

    # a bit of prompt engineering
    # labelset = [ "a " + label + " in a scene" for label in labelset]

    text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px")


    return text_features

import pdb
def main():
    labelset = ['toilet', 'bed', 'chair', 'sofa', 'dresser', 'scanner', 'fridge', 'lamp', 'desk', 'table', 'stand', 'cabinet', 'counter', 'garbage_bin', 'bookshelf', 'pillow', 'microwave', 'sink', 'stool']
    print('Use prompt engineering: a XX in a scene')
    labelset = [ "a " + label + " in a scene" for label in labelset]
    text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px")
    clip_file_name = './text_features/SUNRGBD_text_features_prompt.pt'
    torch.save(text_features, clip_file_name)

if __name__ == '__main__':
    main()

