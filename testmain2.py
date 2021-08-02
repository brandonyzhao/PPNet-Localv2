import os
import shutil

import torch
import torch.utils.data
from tensorboardX import SummaryWriter
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

log = print
class_specific = True

from settings import base_architecture, img_size, prototype_shape, num_classes, global_neg, \
                     prototype_activation_function, add_on_layers_type, experiment_run
import model_ada_dist as model_ada

# construct the model
ppnet = model_ada.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              global_neg=global_neg,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)

#from settings import deneg_global, add_sym, pos_weight, neg_weight
deneg_global=True
add_sym=True
pos_weight = 1.0
neg_weight = -1.0

separation = torch.rand((num_classes, num_classes))
separation = separation * (1. - torch.eye(num_classes)) #zero the diagonal

pci = ppnet.get_new_pci(separation, prototype_shape[0]-ppnet.num_global_prototypes, deneg_global, add_sym, pos_weight, neg_weight)
ppnet.initialize_from_pci(pci)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size, crop

if crop: 
    log('loading cropped images')
else: 
    log('loading uncropped images')

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# train eval set
train_eval_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_eval_loader = torch.utils.data.DataLoader(
    train_eval_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# evaluate accuracy overlap

import train_and_test_diag as tnt

proto_k = -1. 
by_class = False
debug = False




log('test')
# tnt.test(ppnet_multi, ppnet_control_multi, test_loader, proto_k, by_class, log, debug)
tnt.test_protos(ppnet_multi, test_loader, proto_k, by_class, log, debug)

# log('train')
# tnt.test_protos(ppnet_multi, train_eval_loader, proto_k, by_class, log, debug)

#log('train (augmented)')
#tnt.test_protos(ppnet_multi, train_loader, proto_k, by_class, log, debug)