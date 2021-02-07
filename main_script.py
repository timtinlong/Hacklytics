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
from shutil import copyfile
import zerorpc
import sys
from glob import glob
import json
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_performance(preds, labels, nb_classes, conf_matrix):
    preds = torch.exp(preds)
    top_p,top_class = preds.topk(1, dim=1)

    for t, p in zip(labels.view(*top_class.shape), top_class):
        conf_matrix[t, p] += 1

    equals = top_class == labels.view(*top_class.shape)
    test_acc = torch.mean(equals.type(torch.FloatTensor))
    return test_acc, conf_matrix

def get_acc(preds, labels, nb_classes):
    preds = torch.exp(preds)
    top_p,top_class = preds.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    test_acc = torch.mean(equals.type(torch.FloatTensor))
    return test_acc
    
def process_conf_matrix(conf_matrix, nb_classes):
    e = 0.000001
    
    if nb_classes == 2:
        # binary classificaiton
        TP = conf_matrix[0,0]
        TN = conf_matrix[1,1]
        FN = conf_matrix[0,1]
        FP = conf_matrix[1,0]

        sensitivity = round(float((TP / (TP+FN+e))),3)
        specificity = round(float((TN / (TN+FP+e))),3)

        print(conf_matrix)
        print('TP {}, TN {}, FP {}, FN {}'.format(TP, TN, FP, FN))
        print('Sensitivity = {}'.format(sensitivity))
        print('Specificity = {}'.format(specificity))
        return TP, TN, FP, FN, sensitivity, specificity

    else:

        TP = conf_matrix.diag()
        class_lst = []
        TP_lst = []
        TN_lst = []
        FP_lst = []
        FN_lst = []
        sens_lst = []
        spec_lst = []
        for c in range(nb_classes):
            # https://discuss.pytorch.org/t/how-to-check-and-read-confusion-matrix/41835 confusion matrix calculations
            idx = torch.ones(nb_classes).byte()
            idx[c] = 0
            TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
            FN = conf_matrix[c, idx].sum()
            FP = conf_matrix[idx, c].sum()

            sensitivity = round(float((TP[c] / (TP[c]+FN+e))),3)
            specificity = round(float((TN / (TN+FP+e))),3)

            print(conf_matrix)
            print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
                c, TP[c], TN, FP, FN))
            print('Sensitivity = {}'.format(sensitivity))
            print('Specificity = {}'.format(specificity))
            class_lst.append(c)
            TP_lst.append(TP)
            TN_lst.append(TN)
            FP_lst.append(FP)
            FN_lst.append(FN)
            sens_lst.append(sensitivity)
            spec_lst.append(specificity)

        return TP_lst, TN_lst, FP_lst, FN_lst, sens_lst, spec_lst
    


def test_model(test_loader, model, eval_dict, purpose_stamp, nb_classes, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_acc = 0
    conf_matrix = torch.zeros(nb_classes, nb_classes)

    gevent_ctr = 0
    for images,labels in tqdm(test_loader):
        gevent_ctr += 1
        if gevent_ctr % 5 == 0:
            gevent.sleep(0)
        images = images.to(device)
        labels = labels.to(device)
        
        preds = model(images)
        loss = criterion(preds,labels)
        test_loss += loss.item()
        test_acc_temp, conf_matrix = evaluate_performance(preds,labels, nb_classes, conf_matrix)
        test_acc += test_acc_temp


    TP, TN, FP, FN, sensitivity, specificity = process_conf_matrix(conf_matrix, nb_classes)

    avg_test_loss = test_loss/len(test_loader)
    avg_test_acc = test_acc/len(test_loader)
    gevent.sleep(0)
    print(purpose_stamp + "  - Test Loss : {:.6f} Test Acc : {:.6f}".format(avg_test_loss,avg_test_acc))
    eval_dict[purpose_stamp]['Avg_Accuracy'] = avg_test_acc
    eval_dict[purpose_stamp]['Avg_Loss'] = avg_test_loss
    eval_dict[purpose_stamp]['TP'] = TP
    eval_dict[purpose_stamp]['TN'] = TN
    eval_dict[purpose_stamp]['FP'] = FP
    eval_dict[purpose_stamp]['FN'] = FN
    eval_dict[purpose_stamp]['sensitivity'] = sensitivity
    eval_dict[purpose_stamp]['specificity'] = specificity

    return eval_dict

def get_model(model_path, device, num_freeze, num_class=2, pretrained=True):
    from initialization.initialization import get_arch

    model = get_arch()
    model.load_state_dict(torch.load(model_path), strict=False)
    # load_state_dict(torch.load(PATH), strict=False)
    model.to(device)
    # torch.save(model, './model/pneumonia_id/save_new_model.pt')
    # input('asdf')
    layer_lst = []

    # get the number of layers 
    for name, param in model.named_parameters():
        layer_lst.append(name)


    total_num_layers = len(layer_lst)
    arch_str = str(model)
    arch_str = arch_str.splitlines()
    # find the first layer
    first_layer = [s for s in arch_str if "(0)" in s]
    index_first_layer = arch_str.index(first_layer[0])
    arch_str = arch_str[index_first_layer:]

    model_dict = {'Layer':[], 'Type': [], 'NumberOfFilters': []}
    layer_ctr = 0

    # parsing the architecture outputs
    for layer in arch_str:
        # often found when combining portions of layers together
        # found as '(' or ')' 
        if len(layer) < 7:
            continue
        layer_details = layer.split(':')[1]

        layer_type = layer_details.split('(')[0]
        layer_type = layer_type[1:]
        model_dict['Layer'].append(str(layer_ctr))

        if layer_type == 'Conv2d': 
            temp = layer_details.split('(')[1]
            temp = temp.split(',')[0]
            model_dict['Type'].append(layer_type)
            model_dict['NumberOfFilters'].append(temp)

        else:
            model_dict['Type'].append(layer_type)
            model_dict['NumberOfFilters'].append('N/A')
        layer_ctr += 1

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

    return model, model_dict

class HelloRPC(object):
    '''
    referenced https://www.kaggle.com/vatsalmavani/pneumonia-classification-using-pre-trained-model/notebook?select=Pneumonia_model.pt 
    for the original model weights and preprocessing/training steps

    '''
    def fine_tune(id, root_dir, acc_threshold, model_name='pretrained_model.pt', lr=0.001, num_freeze=2, epochs=10, loss_threshold=0.8, model_path='./model'):
        '''
        input: 
            - id (either pneumonia_id or oct) 
                - used for connecting the data with the test set
            - root_dir 
                - path to the training data from the user
                - eg. /home/jason/HB/uuid/
                - should have a folder inside called train_noise and extension.py
            - model_name
                - don't need to change. point this model name to the path
            - model_path 
                - change to uuid for real scenerio, but for testing and proof of concept, models are preloaded into ./models
        '''
        copyfile(os.path.join('./data', id, 'extension.py'), './initialization/initialization.py')
        from initialization.initialization import apply_transform

        start = time.time()
        model_path_OG = os.path.join(model_path,id)
        model_path = os.path.join(model_path_OG, model_name)

        test_set_dir = os.path.join('./data',id)

        OG = 'OG'
        TEST = 'test_noise'
        TRAIN = 'train_noise'
        gevent.sleep(0)
        # Initialize dataset

        trainset = datasets.ImageFolder(os.path.join(root_dir, TRAIN),
                                        transform = apply_transform(TRAIN))
        gevent.sleep(0)
        # valset = datasets.ImageFolder(os.path.join(data_dir,VAL),
        #                               transform = apply_transform(VAL))
        gevent.sleep(0)
        testset = datasets.ImageFolder(os.path.join(test_set_dir, TEST),
                                    transform = apply_transform(TEST))
        gevent.sleep(0)
        OGset = datasets.ImageFolder(os.path.join(test_set_dir, OG),
                                    transform = apply_transform(TEST))

        # # print('Name of Labels:', testset.classes)
        # # print('Index of Labels:', testset.class_to_idx)

        eval_dict = { 'OG': {'Avg_Accuracy': 'value_1', 'Avg_Loss': 'value_1', 'TP': 'value_1', 
                        'TN': 'value_1', 'FP': 'value_1', 'FN': 'value_1', 
                        'sensitivity': 'value_1', 'specificity': 'value_1'},
                    'noisy': {'Avg_Accuracy': 'value_1', 'Avg_Loss': 'value_1', 'TP': 'value_1', 
                        'TN': 'value_1', 'FP': 'value_1', 'FN': 'value_1', 
                        'sensitivity': 'value_1', 'specificity': 'value_1'},
                    'fine-tuned': {'Avg_Accuracy': 'value_1', 'Avg_Loss': 'value_1', 'TP': 'value_1', 
                        'TN': 'value_1', 'FP': 'value_1', 'FN': 'value_1', 
                        'sensitivity': 'value_1', 'specificity': 'value_1'}}


        train_loader = DataLoader(trainset,
                                batch_size=16,
                                shuffle=True)

        OG_loader = DataLoader(OGset,
                                batch_size=8)

        test_loader = DataLoader(testset,
                                batch_size=8)

        # val_loader = DataLoader(valset,
        #                         batch_size=8)
        gevent.sleep(0)

        print('Training Images:')
        dataiter = iter(train_loader)
        images,labels = dataiter.next()
        print("shape of images : {}".format(images.shape))
        print("shape of labels : {}".format(labels.shape))

        # print('\nValidation Images:')
        # dataiter = iter(val_loader)
        # images,labels = dataiter.next()
        # print("shape of images : {}".format(images.shape))
        # print("shape of labels : {}".format(labels.shape))

        print('\nTest Images:')
        dataiter = iter(test_loader)
        images,labels = dataiter.next()
        print("shape of images : {}".format(images.shape))
        print("shape of labels : {}".format(labels.shape))

        # need to find number of unique labels
        num_class = len(testset.classes)
        nb_classes = num_class
        print("number of classes: {}".format(str(num_class)))
        gevent.sleep(0)
        # model = TheModelClass(*args, **kwargs)
        model, model_dict = get_model(model_path, device, num_freeze, num_class=num_class, pretrained=True)
        # model = torch.load('./model/pneumonia_id/save_new_model.pt')
        gevent.sleep(0)
        ''' baseline without tuning '''
        # low LR for fine tuning
        optimizer = optim.Adam(model.parameters(), lr=lr)
        schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

        purpose_stamp = 'OG'
        eval_dict = test_model(OG_loader, model, eval_dict, purpose_stamp, nb_classes, device)

        purpose_stamp = 'noisy'

        eval_dict = test_model(test_loader, model, eval_dict, purpose_stamp, nb_classes, device)

        # input('done original testing')
        train_loss_min = np.Inf

        if num_class > 2:
            criterion = nn.NLLLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        ''' fine tune model ''' 
        # for epoch in range(epochs):
        epoch = 0 
        best_train_acc = 0
        gevent.sleep(0)
        
        # keep training until accuracy threshold AND the specified number of epochs
        while best_train_acc < acc_threshold or epoch < epochs:
            if epoch > 20:
                break
            gevent.sleep(0)
            epoch += 1
            print('training epoch number:', epoch)

            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0
            val_acc = 0.0
            
            model.train()
            gevent_ctr = 0
            for images,labels in tqdm(train_loader):
                gevent_ctr += 1
                if gevent_ctr % 10 == 0:
                    gevent.sleep(0)
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += get_acc(preds, labels, nb_classes)

            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_acc / len(train_loader)

            schedular.step(avg_train_loss)
            gevent.sleep(0)
            # print("Epoch : {} \ntrain_loss : {:.6f}, \tTrain_acc : {:.6f}".format(epoch + 1, avg_train_loss, avg_train_acc))

            if avg_train_loss <= train_loss_min and avg_train_loss < loss_threshold:
                # if avg_train_acc > acc_threshold:
                # print('Training loss decreased from ({:.6f} --> {:.6f}).\nSaving model ...'.format(train_loss_min, avg_train_loss))
                gevent.sleep(0)
                torch.save(model.state_dict(), os.path.join(model_path_OG,'finetuned_model.pt'))
                gevent.sleep(0)
                train_loss_min = avg_train_loss
                best_train_acc = avg_train_acc


        model.load_state_dict(torch.load(os.path.join(model_path_OG,'finetuned_model.pt')), strict=False)
        # load_state_dict(torch.load(PATH), strict=False)
        gevent.sleep(0)
        model.to(device)
        purpose_stamp = 'fine-tuned'
        eval_dict = test_model(test_loader, model, eval_dict, purpose_stamp, nb_classes, device)
        gevent.sleep(0)
        print('######################   Outputs   #######################')
        print('evaluation metrics:')
        print(eval_dict)
        # print('\n\nmodel architecture:')
        # print(model_dict)
        os.remove('./initialization/initialization.py')
        gevent.sleep(0)
        print("Total duration:")
        end = time.time()
        print(end - start)
        # summary(model, (images.shape[1],images.shape[2],images.shape[3]))
        return_dict = {'evalMetrics': eval_dict, 'modelArch': model_dict, 'runDuration': end-start}
        # return json.dumps(str(return_dict)), model
        return 'working'




if __name__ == "__main__":
    s = zerorpc.Server(HelloRPC())
    s.bind("tcp://0.0.0.0:4242")
    s.run()
    # HelloRPC.fine_tune(id='oct', root_dir='./data/oct', acc_threshold=0.7, model_name='pretrained_model.pt', lr=0.001, num_freeze=2, epochs=10, loss_threshold=0.8, model_path='./model')

    # id='pneumonia_id', root_dir='./data/pneumonia_id', acc_threshold=0.8, model_name='pretrained_model.pt', lr=0.001, num_freeze=2, epochs=10, loss_threshold=0.8, model_path='./model'
    # id='pneumonia_id', model_name = 'pretrained_model.pt', acc_thW