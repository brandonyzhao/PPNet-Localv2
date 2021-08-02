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

model_dir_control = 'control'
model_file_control = '10_106push0.7710.pth'
model_path_control = './saved_models/vgg19_bn/' + model_dir_control + '/' + model_file_control

log('control model file: ', model_file_control)

model_dir = '448_add_withnorm200global_negTruedeneg_globalTrueadd_symTruenewscalingsep_margin2.0top-1p1.0n-1.0clst_l0.5sep_l-0.05dist1.0'
model_file = 'scr10nopush0.7831.pth'
model_path = './saved_models/vgg19_bn/' + model_dir + '/' + model_file

log('model dir: ', model_dir)
log('model file: ', model_file)

# load control model
ppnet_control = torch.load(model_path_control)
ppnet_control = ppnet_control.cuda()
ppnet_control_multi = torch.nn.DataParallel(ppnet_control)
class_specific = True

img_size = ppnet_control.img_size

#load model with local+global prototypes

ppnet = torch.load(model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

assert (img_size == ppnet.img_size), 'models are trained on different image sizes'

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