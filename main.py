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


def load_data(path):
    '''

    :param path:
    :return: the loader for train, valid and test sets

    '''
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    image_datasets = [train_data, valid_data, test_data]
    return trainloader, validloader, testloader, train_data


def model_setup (arch, learning_rate, hidden_units):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Build and train your network
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print("{} is not a valid model choose a vgg model".format(arch))
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.25),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    model.to(device)
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, epoch, trainloader,validloader):
    # train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epoch
    steps = 0
    running_loss = 0
    print_every = 50
    start = time.time()
    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Valid loss: {test_loss / len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy / len(validloader):.3f}")
                running_loss = 0
                model.train()
    time_end = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_end//60, time_end % 60))

def saving_checkpoint(model, path, train_data,hidden_units, optimizer):
    # saving the checkpoint
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.25),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'classifier': classifier,
                  'arch': 'vgg19',
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, path)

def load_checkpoint(filepath):
    checkpoints = torch.load(filepath)
    model = models.vgg19(pretrained=True)
    model.class_to_idx = checkpoints['class_to_idx']
    model.classifier = checkpoints['classifier']
    model.load_state_dict(checkpoints['state_dict'])
    # optimizer.load_state_dict(checkpoints['optimizer'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256,256))
    transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    im = transform(im)
    return im


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_index = probs.topk(topk)
    top_index = top_index[0].numpy()
    index = []
    for i in range(len(model.class_to_idx.items())):
        index.append(list(model.class_to_idx.items())[i][0])

    label = []
    for i in range(5):
        label.append(index[top_index[i]])

    return top_probs, label