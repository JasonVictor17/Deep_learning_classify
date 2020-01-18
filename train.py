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
parser.add_argument("data_directory", help="add a data directory", default="flowers")
parser.add_argument("--arch", default="vgg19", type=str)
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--hidden_units", default=2048)
parser.add_argument("--epochs", default=8, type=int)
parser.add_argument("--save_dir", default="checkpoint.pth")
args = parser.parse_args()


data_dir = args.data_directory
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
save_dir = args.save_dir

trainloader, validloader, testloader, train_data = main.load_data(data_dir)
model, criterion, optimizer = main.model_setup(arch, learning_rate, hidden_units)

main.train_model(model, criterion, optimizer, epochs,trainloader,validloader)

main.saving_checkpoint(model, save_dir, train_data,hidden_units, optimizer)

print("The model is trained")


