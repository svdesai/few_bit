
# what we need: current label/unlabel split, current best model, stage number
# sample 20k images from currently unlabeled list and create a new csv file
# use that csv to create a testing dataset

# evaluate the current model on each image in the csv and get class prediction for each
# compare prediction with the ground truth, if correct - save : (image name, label)
# if incorrect, save (image name, [wrong label])

# create a new dataset with these pairs
# when training, if label is integer, learn using cross entropy loss
# if label is not an integer and a list, learn using L2 loss
# save the best model

# modifications: label smoothing, more guesses, margin sampling

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from models.resnet import resnet26
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os, shutil
import copy
import pdb
import pandas as pd
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(level=logging.INFO)

from utils.datasets import CIFAR100, CIFAR100_BitLabeled


def train_model(net, trainloader, optimizer, criterion, device, epoch=-1):
    logging.info('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(trainloader):
        inputs = batch['image'].to(device)
        targets = batch['label']


        guess_indices = [i for i,x in enumerate(targets) if x < 0]
        true_indices = [i for i,x in enumerate(targets) if x >= 0]

        guess_inputs = inputs[guess_indices]
        guess_targets = -1 * targets[guess_indices]

        guess_target_tensors = []
        for gt in guess_targets:
            vec = np.zeros(100)
            vec[gt] = -1.0 * 10000 # large negative value
            guess_target_tensors.append(vec)

        true_inputs = inputs[true_indices]
        true_targets = targets[true_indices]

        guess_inputs = guess_inputs.to(device)
        guess_targets = torch.Tensor(np.array(guess_target_tensors)).to(device)
        true_inputs, true_targets = true_inputs.to(device), true_targets.to(device)
        optimizer.zero_grad()

        guess_outputs = net(guess_inputs)
        
        for i in range(len(guess_outputs)):
            for j in range(len(guess_outputs[i])):
                if guess_targets[i][j] != -10000.0:
                    guess_targets[i][j] = guess_outputs[i][j]
        
        mse = nn.MSELoss()
        loss = 0.0001 * mse(guess_outputs, guess_targets)

        true_outputs = net(true_inputs)
        loss += criterion(true_outputs, true_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0 or batch_idx == len(trainloader) - 1:
            logging.info('Loss: %.3f' % (train_loss/(batch_idx+1)))
            # logging.info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return net


def test_model(net, testloader, criterion, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    logging.info("\nTest accuracy: %.3f" % acc)
    return acc


def sample_unlabeled_data(dset_list, n_sample, method='random'):
    unlab_list = np.array([i for i,x in enumerate(dset_list) if x[1] == 'unlabeled'])
    if method == 'random':
        np.random.shuffle(unlab_list)
        indices_to_label = unlab_list[:n_sample]
    
    for idx in indices_to_label:
        dset_list[idx][1] = 'guess_labeled'
    return dset_list


def guess_labels(preds, reals, bits=1):
    """
    return a list of guessed labels. in the list,
    if the item is scalar, then it's a correct guess
    else if the item is a list, then it's a list of wrong guesses, len of that list = bits
    """
    # apply softmax
    preds = F.softmax(preds).numpy()

    guessed_labels = []

    for i, pred in enumerate(preds):
        pred_class = np.argmax(pred)
        real_class = reals[i].item()
        if pred_class == real_class:
            guessed_labels.append(real_class)
        else:
            guessed_labels.append([pred_class])
    
    return guessed_labels


def predict_and_guess(net, testloader, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    guess_dataset = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            guessed_labels = guess_labels(outputs.detach().cpu(), targets.detach().cpu())
            
            for i, gl in enumerate(guessed_labels):
                guess_dataset.append({ "image_path": batch['name'][i] , "label": gl })
    return guess_dataset
    

def main():
    output_dir = 'output/trial1_stage1'
    dataset_root = 'data/cifar100'
    stage = 1
    n_sample = 20000
    seed = 42
    lr = 0.05
    epochs = 5
    curr_model_path = 'output/trial1/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Loading the prev stage model from %s"  % os.path.join(curr_model_path, 'best_model.pth'))
    # Load current model
    curr_model = resnet26(pretrained=False, num_classes=100)
    state_dict = torch.load(os.path.join(curr_model_path, 'best_model.pth'))
    curr_model.load_state_dict(state_dict['model'])
    curr_model = curr_model.to(device)

    curr_split_csv = os.path.join(curr_model_path, 'train_stage0.csv')
    curr_split_list  = pd.read_csv(curr_split_csv, header=None).to_numpy()

    logging.info("Sampling %d images from unlabeled set for guessing" % n_sample)
    new_split_list = sample_unlabeled_data(curr_split_list, n_sample)

    # save the new split list
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    stage_n_csv = os.path.join(output_dir, 'train_stage1.csv')
    np.savetxt(stage_n_csv, new_split_list, fmt="%s,%s")

    # create a test dataset with this list
    stage_n_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((224,224)),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])

    test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((224,224)),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
    
    cifar100_stage_n = CIFAR100(csv_path=stage_n_csv, root_dir='data/cifar100/train', transforms=stage_n_transforms, keyword='guess_labeled')
    cifar100_stage_n_loader = DataLoader(cifar100_stage_n, batch_size=32, shuffle=True, num_workers=0)
    
    logging.info("Using current model to make predictions (and guess)")
    # get the guess dataset
    guess_dataset = predict_and_guess(curr_model, cifar100_stage_n_loader, device)

    # create a new dataset = labeled_dataset + guessed_dataset
    labeled_train = CIFAR100(csv_path=stage_n_csv, root_dir='data/cifar100/train', transforms=stage_n_transforms, keyword='labeled')
    cifar100_bitlabeled = CIFAR100_BitLabeled(root_dir='data/cifar100/train', guess_dataset=guess_dataset, 
                                                labeled_dataset=labeled_train, transforms=stage_n_transforms)
    cifar100_test = CIFAR100(csv_path='data/cifar100/test.csv', root_dir='data/cifar100/test', transforms=test_transforms)
    
    cifar100_bitlabeled_loader = DataLoader(cifar100_bitlabeled, batch_size=32, shuffle=True, num_workers=0)
    cifar100_testloader = DataLoader(cifar100_test, batch_size=32, shuffle=False, num_workers=0)

    optimizer = optim.Adam(curr_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        curr_model = train_model(curr_model, cifar100_bitlabeled_loader, optimizer, criterion, device, epoch)
        acc = test_model(curr_model, cifar100_testloader, criterion, device)

        if acc > best_acc:
            best_acc = acc
            print("\n\n")
            logging.info("Saving current best model at epoch %d with acc %.2f percent" % (epoch, best_acc))
            torch.save({ "model": model.state_dict(), "acc": acc },
                        os.path.join(output_dir, 'best_model.pth'))

    pdb.set_trace()




if __name__  == "__main__":
    main()

