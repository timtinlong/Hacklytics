import argparse
import os
import configparser
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from misc_functions import *
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
referenced https://www.kaggle.com/vatsalmavani/pneumonia-classification-using-pre-trained-model/notebook?select=Pneumonia_model.pt 
for the original model weights and preprocessing/training steps

'''


# parser = argparse.ArgumentParser(description='Process arguments')
# parser.add_argument('-n', '--num_freeze', type=int, default=2)

# args = parser.parse_args()

# data_dir = './chest_xray'

def evaluate_performance(preds, labels):
    preds = torch.exp(preds)
    top_p,top_class = preds.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    test_acc = torch.mean(equals.type(torch.FloatTensor))
    return test_acc
    
def test_model(test_loader, model, eval_dict, purpose_stamp, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_acc = 0
    for images,labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = criterion(preds,labels)
        test_loss += loss.item()
        test_acc += evaluate_performance(preds,labels)

    avg_test_loss = test_loss/len(test_loader)
    avg_test_acc = test_acc/len(test_loader)

    print(purpose_stamp + "  - Test Loss : {:.6f} Test Acc : {:.6f}".format(avg_test_loss,avg_test_acc))

    eval_dict[purpose_stamp]['Avg_Accuracy'] = avg_test_acc
    eval_dict[purpose_stamp]['Avg_loss'] = avg_test_loss

    return eval_dict

def get_model(model_path, device, num_freeze, num_class=2, pretrained=True):

    model = torchvision.models.vgg19(pretrained=pretrained)
    # add Linear classifier layer
    in_features = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 2),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    model.load_state_dict(torch.load(model_path), strict=False)
    # load_state_dict(torch.load(PATH), strict=False)
    model.to(device)
    layer_lst = []

    # get the number of layers 
    for name, param in model.named_parameters():
        layer_lst.append(name)

    total_num_layers = len(layer_lst)

    loop_ctr = 0
    freezing_ctr = 0
    # freeze the model layer
    # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
    for name, param in model.named_parameters():
        # bias and weights included, so each layer will have 2 counts of 'param'
        if loop_ctr < total_num_layers-(num_freeze*2):
            param.requires_grad = False
        else:
            param.requires_grad = True
            freezing_ctr += 1
        loop_ctr += 1

    # input('freezing_ctr ' + str(freezing_ctr))
    print('freezing_ctr ' + str(freezing_ctr))

    return model

# define augmentation
def apply_transform(mode=None):
# same preprocessing as the original trained model for pneumonia
    if mode == 'train_noise':
        transform = T.Compose([T.Resize((256,256)),
                            T.RandomHorizontalFlip(),
                            T.RandomRotation((-20,+20)),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                            ])

    elif mode == 'test_noise' or mode == 'val_noise':
        transform = T.Compose([T.Resize((256,256)),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                            ])
        
    return transform

def fine_tune(data_dir='./chest_xray', model_path='./model/pretrained_model.pt', num_freeze=2, epochs=10):
    start = time.time()
    print("hello")
    end = time.time()
    print(end - start)
    
    # TEST = 'test'
    TEST = 'test_noise'
    TRAIN = 'train_noise'
    VAL = 'val_noise'

    # Initialize dataset

    trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN),
                                    transform = apply_transform(TRAIN))

    valset = datasets.ImageFolder(os.path.join(data_dir,VAL),
                                  transform = apply_transform(VAL))

    testset = datasets.ImageFolder(os.path.join(data_dir, TEST),
                                transform = apply_transform(TEST))

    print('Name of Labels:', testset.classes)
    print('Index of Labels:', testset.class_to_idx)

    eval_dict = { 'original': {'Avg_Accuracy': 'value_1', 'Avg_loss': 'value_1'},
                    'fine-tuned': {'Avg_Accuracy': 'value_2', 'Avg_loss': 'value_1'}}

    train_loader = DataLoader(trainset,
                            batch_size=8,
                            shuffle=True)

    test_loader = DataLoader(testset,
                            batch_size=8)

    val_loader = DataLoader(valset,
                            batch_size=8)


    print('Training Images:')
    dataiter = iter(train_loader)
    images,labels = dataiter.next()
    print("shape of images : {}".format(images.shape))
    print("shape of labels : {}".format(labels.shape))

    print('\nValidation Images:')
    dataiter = iter(val_loader)
    images,labels = dataiter.next()
    print("shape of images : {}".format(images.shape))
    print("shape of labels : {}".format(labels.shape))

    print('\nTest Images:')
    dataiter = iter(test_loader)
    images,labels = dataiter.next()
    print("shape of images : {}".format(images.shape))
    print("shape of labels : {}".format(labels.shape))

    # need to find number of unique labels
    num_class = len(testset.classes)
    print("number of classes: {}".format(str(num_class)))

    # model = TheModelClass(*args, **kwargs)
    model = get_model(model_path, device, num_freeze, num_class=num_class, pretrained=True)

    ''' baseline without tuning '''
    # low LR for fine tuning
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    purpose_stamp = 'original'
    eval_dict = test_model(test_loader, model, eval_dict, purpose_stamp, device)

    print("Baseline training and testing duration:")
    end = time.time()
    print(end - start)
    start = time.time()


    # input('done original testing')
    val_loss_min = np.Inf
    criterion = nn.CrossEntropyLoss()

    ''' fine tune model ''' 
    for epoch in range(epochs):
        print('training epoch number:', epoch)

        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        
        model.train()
        for images,labels in tqdm(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += evaluate_performance(preds, labels)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        model.eval()
        with torch.no_grad():
            for images,labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_acc += evaluate_performance(preds, labels)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)

        schedular.step(avg_val_loss)

        print("Epoch : {} \ntrain_loss : {:.6f}, \tTrain_acc : {:.6f}, \nVal_loss : {:.6f}, \tVal_acc : {:.6f}".format(epoch + 1,
                                                                                                                    avg_train_loss, avg_train_acc,
                                                                                                                    avg_val_loss, avg_val_acc))
        if avg_val_loss <= val_loss_min:
            print('Validation loss decreased from ({:.6f} --> {:.6f}).\nSaving model ...'.format(val_loss_min, avg_val_loss))
            torch.save(model.state_dict(), 'Pneumonia_model.pt')
            val_loss_min = avg_val_loss

    model.load_state_dict(torch.load('./model/finetuned_model.pt'), strict=False)
    # load_state_dict(torch.load(PATH), strict=False)
    model.to(device)
    purpose_stamp = 'fine-tuned'
    eval_dict = test_model(test_loader, model, eval_dict, purpose_stamp, device)
    print(eval_dict)

    print("fine tuning training and testing duration:")
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    fine_tune(data_dir='./chest_xray', model_path='./model/pretrained_model.pt', num_freeze=2, epochs=10)