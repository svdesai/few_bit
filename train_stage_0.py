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
import time
import os, shutil
import copy
import pdb
import pandas as pd
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(level=logging.INFO)

from utils.datasets import CIFAR100


def train_model(net, trainloader, optimizer, criterion, device, epoch=-1):
    logging.info('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(trainloader):
        inputs = batch['image'].to(device)
        targets = batch['label'].to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0 or batch_idx == len(trainloader) - 1:
            logging.info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

def main():
    output_dir = 'output/trial1'
    dataset_root = 'data/cifar100'
    stage0_split = 0.06 # 6%
    seed = 42
    lr = 0.09
    epochs = 10

    np.random.seed = seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    
    orig_dataset_csv = os.path.join(dataset_root, 'train.csv')
    shutil.copy(orig_dataset_csv, output_dir)
    dataset_csv = os.path.join(output_dir, 'train.csv')

    orig_dataset_list = pd.read_csv(dataset_csv, header=None).to_numpy()
    dataset_list = orig_dataset_list

    unlabeled_idxs = [i for i,x in enumerate(dataset_list) if x[1] == 'unlabeled']
    labeled_idxs = [i for i,x in enumerate(dataset_list) if x[1] == 'labeled']
    logging.info("Number of unlabeled train images: %d" % (len(unlabeled_idxs)))
    logging.info("Number of labeled train images: %d" % (len(labeled_idxs)))

    # Stage 0
    # Randomly sample stage0_split fraction of images and label them
    np.random.shuffle(unlabeled_idxs)
    stage0_size = int(stage0_split * len(dataset_list))
    stage0_idxs = unlabeled_idxs[:stage0_size]

    for idx in stage0_idxs:
        assert dataset_list[idx][1] == 'unlabeled'
        dataset_list[idx][1] = 'labeled'
    # save the current dataset to csv
    stage0_csv = os.path.join(output_dir, 'train_stage0.csv')
    np.savetxt(stage0_csv, dataset_list, fmt="%s,%s")

    # read the current dataset
    stage0_dataset = pd.read_csv(stage0_csv, header=None).to_numpy()
    unlabeled_idxs = [i for i,x in enumerate(stage0_dataset) if x[1] == 'unlabeled']
    labeled_idxs = [i for i,x in enumerate(stage0_dataset) if x[1] == 'labeled']
    logging.info("STAGE 0")
    logging.info("-"*40)
    logging.info("Number of unlabeled train images: %d" % (len(unlabeled_idxs)))
    logging.info("Number of labeled train images: %d" % (len(labeled_idxs)))

    train_transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Resize((224,224)),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
    
    test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((224,224)),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])

    # create the pytorch dataset
    cifar100_train = CIFAR100(csv_path=stage0_csv, root_dir='data/cifar100/train', transforms=train_transforms)
    cifar100_test = CIFAR100(csv_path='data/cifar100/test.csv', root_dir='data/cifar100/test', transforms=test_transforms)

    cifar100_trainloader = DataLoader(cifar100_train, batch_size=32, shuffle=True, num_workers=0)
    cifar100_testloader = DataLoader(cifar100_test, batch_size=32, shuffle=False, num_workers=0)

    model = resnet26(pretrained=False, num_classes=100)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        model = train_model(model, cifar100_trainloader, optimizer, criterion, device, epoch)
        acc = test_model(model, cifar100_testloader, criterion, device)

        if acc > best_acc:
            best_acc = acc
            print("\n\n")
            logging.info("Saving current best model at epoch %d with acc %.2f percent" % (epoch, best_acc))
            torch.save({ "model": model.state_dict(), "acc": acc },
                        os.path.join(output_dir, 'best_model.pth'))


if __name__ == "__main__":
    main()




