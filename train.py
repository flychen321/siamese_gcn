# -*- coding: utf-8 -*-
# Author: chen
from __future__ import print_function, division
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
import matplotlib
from PIL import Image

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from model import ft_net, ft_net_dense
from random_erasing import RandomErasing
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.io import savemat
from datasets import SiameseDataset, SggDataset
from model import ft_net_dense, SiameseNet, Sggnn_siamese, Sggnn_gcn, Sggnn_end_to_end
from model import save_network, save_whole_network, load_network_easy, load_network
from losses import ContrastiveLoss, SigmoidLoss

######################################################################
# Options
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='SGG_model', type=str, help='output model name')
parser.add_argument('--data_dir', default='data/market/pytorch', type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0.8, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_soft_label', default=True, type=bool, help='use_soft_label')
parser.add_argument('--prob', default=80, type=float, help='hard label probability, in [0,1]')
parser.add_argument('--modelname', default='', type=str, help='save model name')
parser.add_argument('--max', default=80, type=float, help='max label probability, in [0,1]')
parser.add_argument('--min', default=60, type=float, help='min label probability, in [0,1]')

opt = parser.parse_args()
opt.use_dense = True
data_dir = opt.data_dir
name = opt.name
opt.prob = opt.prob / 100.0
print('prob = %.3f' % opt.prob)
print('save model name = %s' % opt.modelname)
use_gpu = torch.cuda.is_available()

######################################################################
transform_train_list = [
    transforms.Resize(144, interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}


dataset_sizes = {}
dataset_train_dir = os.path.join(data_dir, 'train_all_new')
dataset_val_dir = os.path.join(data_dir, 'val_new')
dataset_sizes['train'] = sum(len(os.listdir(os.path.join(dataset_train_dir, i))) for i in os.listdir(dataset_train_dir))
dataset_sizes['val'] = sum(len(os.listdir(os.path.join(dataset_val_dir, i))) for i in os.listdir(dataset_val_dir))

print('dataset_sizes[train] = %s' % dataset_sizes['train'])
print('dataset_sizes[val] = %s' % dataset_sizes['val'])

dataset = datasets.ImageFolder(dataset_train_dir, data_transforms['train'])
dataloaders_siamese = {}
dataloaders_gcn = {}

dataloaders_siamese['train'] = DataLoader(
    SiameseDataset(dataset_train_dir, data_transforms['train']), batch_size=opt.batchsize, shuffle=True, num_workers=8)
# dataloaders['val'] = DataLoader(SiameseDataset(datasets.ImageFolder(dataset_val_dir, data_transforms['val'])),
#                                 batch_size=opt.batchsize,
#                                 shuffle=True, num_workers=8)

dataloaders_gcn['train'] = DataLoader(
    SggDataset(dataset_train_dir, data_transforms['train']), batch_size=opt.batchsize, shuffle=True, num_workers=8)


# for data in dataloaders['train']:
#     print(data)


# fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
######################################################################
# Training the model
def train_model(train_loader, model, loss_fn, optimizer, num_epochs=25):
    global cnt
    since = time.time()
    model.train()
    # model.eval()
    losses = []
    total_loss = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if use_gpu:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            optimizer.zero_grad()
            outputs = model(*data)  # for contrastive loss
            # outputs, target = model(*data, target) # for SGGNN

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('batch_idx = %4d  loss = %f' % (batch_idx, loss))
        if (epoch + 1) % 5 == 0:
            save_network(model, str(epoch + 1))
            save_whole_network(model, str(epoch))
    time_elapsed = time.time() - since
    print('time = %f' % (time_elapsed))
    save_network(model, 'best')
    save_whole_network(model, 'best')
    return model


def train_siamese(train_loader, model, loss_fn, optimizer, num_epochs=25):
    global cnt
    since = time.time()
    model.train()
    # model.eval()
    losses = []
    total_loss = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if use_gpu:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            optimizer.zero_grad()
            outputs = model(*data)  # for contrastive loss

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('batch_idx = %4d  loss = %f' % (batch_idx, loss))

    time_elapsed = time.time() - since
    print('time = %f' % (time_elapsed))
    save_network(model, 'best')
    return model


def train_gcn(train_loader, model_siamese, loss_siamese_fn, model_gcn, loss_gcn_fn, optimizer, num_epochs=25):
    global cnt
    since = time.time()
    model_gcn.train()
    model_siamese.eval()
    losses = []
    total_loss = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if use_gpu:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model_siamese(*data, target)

            outputs, target = model_gcn(*outputs)  # for SGGNN

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_gcn_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('batch_idx = %4d  loss = %f' % (batch_idx, loss))

    time_elapsed = time.time() - since
    print('time = %f' % (time_elapsed))
    save_whole_network(model_gcn, 'best')
    return model_gcn


dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

def stage_1_params(model):
    # cnt = 0
    # for m in model.children():
    #     print('before  cnt = %d' % cnt)
    #     print(m)
    #     cnt += 1
    #     print('cnt = %d' % cnt)
    # exit()
    stage_1_id = list(map(id, model.embedding_net.parameters())) \
                 + list(map(id, model.bn.parameters())) \
                 + list(map(id, model.fc.parameters())) \
                 + list(map(id, model.classifier.parameters()))
    stage_1_base_id = list(map(id, model.embedding_net.parameters()))
    stage_1_base_params = filter(lambda p: id(p) in stage_1_base_id, model.parameters())
    stage_1_classifier_params = filter(lambda p: id(p) in stage_1_id and id(p) not in stage_1_base_id,
                                       model.parameters())

    return stage_1_base_params, stage_1_classifier_params


stage_1 = False
stage_2 = True

if stage_1:
    margin = 1.
    embedding_net = ft_net_dense()
    model = SiameseNet(embedding_net)
    # model = Sggnn(SiameseNet(embedding_net))
    if use_gpu:
        model.cuda()
    # save_whole_network(model, 'best')
    # exit()
    loss_fn = ContrastiveLoss(margin)
    # loss_fn = SigmoidLoss()
    # loss_fn = nn.CrossEntropyLoss()

    lr = 1e-3
    step = 8

    # stage_1_base_params, stage_1_classifier_params = stage_1_params(model)
    # optimizer = optim.Adam([
    #     {'params': stage_1_base_params, 'lr': 1 * lr},
    #     {'params': stage_1_classifier_params, 'lr': 1 * lr},
    # ])

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = lr_scheduler.StepLR(optimizer, step, gamma=0.1, last_epoch=-1)
    n_epochs = 1
    log_interval = 100
    model = train_model(dataloaders_siamese['train'], model, loss_fn, optimizer, num_epochs=n_epochs)


if stage_2:
    margin = 1.
    embedding_net = ft_net_dense()
    model_siamese = Sggnn_siamese(SiameseNet(embedding_net))
    model_siamese.eval()
    model_gcn = Sggnn_gcn()
    model_gcn.train()

    if use_gpu:
        model_siamese.cuda()
        model_gcn.cuda()
    # loss_fn = ContrastiveLoss(margin)
    model_siamese = load_network(model_siamese)
    loss_siamese_fn = ContrastiveLoss(margin)
    loss_gcn_fn = SigmoidLoss()
    # loss_fn = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = optim.Adam(model_gcn.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100
    model = train_gcn(dataloaders_gcn['train'], model_siamese, loss_siamese_fn, model_gcn, loss_gcn_fn, optimizer,
                      num_epochs=n_epochs)
