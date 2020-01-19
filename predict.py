import helper
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import numpy as np
import time
from torch import nn
from torch import optim
import seaborn as sns
from PIL import Image
import argparse
import main

parser = argparse.ArgumentParser()
parser.add_argument('img', default='flowers/test/17/image_03911.jpg', type = str)
parser.add_argument('checkpoint', default='checkpoint.pth',type = str)
parser.add_argument('--top_k', default=5,  type=int)
parser.add_argument('--category_names',  default='cat_to_name.json')
args = parser.parse_args()

img_path = args.img
top_k = args.top_k
checkpoint_path = args.checkpoint
lables = args.category_names

model = main.load_checkpoint(checkpoint_path)

with open('lables', 'r') as f:
    cat_to_name = json.load(f)

prob, classes = main.predict(img_path, model)
prob = prob[0].detach().numpy()*100
labels = []
for each in classes:
    labels.append(cat_to_name[each])

for i in range(top_k):
    print("{}has p = {}%".format(labels[i],prob[i]))

print("The prediction is finsished")