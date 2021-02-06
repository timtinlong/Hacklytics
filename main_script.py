import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
import torchvision
from torch import nn, optim
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

data_dir = '../input/chest-xray-pneumonia/chest_xray'
model_path = '../model/pretrained_model.pt'
TRAIN = 'train'
TEST = 'test'
VAL = 'val'

# show batches
def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    images = make_grid(images, nrow=4, padding=15)
    imshow(images, title=["NORMAL" if x==0  else "PNEUMONIA" for x in labels])


# define augmentation

def apply_transform(mode=None):

    if mode == 'train':
        transform = T.Compose([T.Resize((256,256)),
                               T.RandomHorizontalFlip(),
                               T.RandomRotation((-20,+20)),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
                              ])

    elif mode == 'test' or mode == 'val':
        transform = T.Compose([T.Resize((256,256)),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
                              ])
        
    return transform

# Initialize dataset

# trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN),
#                                 transform = apply_transform(TRAIN))

# valset = datasets.ImageFolder(os.path.join(data_dir,VAL),
#                               transform = apply_transform(VAL))

testset = datasets.ImageFolder(os.path.join(data_dir, TEST),
                               transform = apply_transform(TEST))

print('Name of Labels:', trainset.classes)
print('Index of Labels:', trainset.class_to_idx)

# define methods for class-count visualization

def class_count(dataset):
    count = dict(Counter(dataset.targets))
    count = dict(zip(dataset.classes[::-1], list(count.values())))      # changing keys of dictionary 
    return count

def plot_class_count(dataset, name='Dataset Labels Count'):
    count = class_count(dataset)
    pd.DataFrame(count, index=['Labels']).plot(kind='bar', title=name).show()

test_loader = DataLoader(testset,
                         batch_size=8)
print('\nTest Images:')
dataiter = iter(test_loader)
images,labels = dataiter.next()
print("shape of images : {}".format(images.shape))
print("shape of labels : {}".format(labels.shape))

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(model_path))
model.eval()

''' baseline without tuning '''

test_loss = 0
test_acc = 0
for images,labels in tqdm(test_loader):

    images = images.to(device)
    labels = labels.to(device)

    preds = model(images)
    loss = criterion(preds,labels)
    test_loss += loss.item()
    test_acc += accuracy(preds,labels)

avg_test_loss = test_loss/len(test_loader)
avg_test_acc = test_acc/len(test_loader)

print("Test Loss : {:.6f} Test Acc : {:.6f}".format(avg_test_loss,avg_test_acc))