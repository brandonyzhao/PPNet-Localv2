import os
import shutil

import torch
import torch.utils.data
from tensorboardX import SummaryWriter
import torchvision.utils as utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

import numpy as np

from helpers import makedir, invert
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

print('experiment run: ', experiment_run)

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)


model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
'''
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
'''

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)
unmean, unstd = invert(mean, std)
unnormalize = transforms.Normalize(mean= unmean, std = unstd)

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
    train_dataset, batch_size=5, shuffle=True,
    num_workers=4, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=1, shuffle=False,
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

base_dir = '/usr/xtmp/brandon_zhao/PPNet-Spike/saved_models/vgg19_bn/'

# construct the model
ppnet1 = torch.load('saved_models/vgg19_bn/448_2000_crs1divcs0pl22pl414sep8e-2dist0.0/10nopush0.7656.pth')
imgdir1 = base_dir + '448_2000_crs1divcs0pl22pl414sep8e-2dist0.0/img/epoch-10/'
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet1 = ppnet1.cuda()
ppnet1_multi = torch.nn.DataParallel(ppnet1)

# construct the model
ppnet2 = torch.load('saved_models/vgg19_bn/448_2000_crs1divcs0pl22pl414sep8e-2dist0.1/10nopush0.7687.pth')
imgdir2 = base_dir + '448_2000_crs1divcs0pl22pl414sep8e-2dist0.1/img/epoch-10/'
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet2 = ppnet2.cuda()
ppnet2_multi = torch.nn.DataParallel(ppnet2)

# construct the model
ppnet3 = torch.load('saved_models/vgg19_bn/distfullgrad_protomax_448_2000_crs1divcs0pl22pl414sep8e-2dist0.1/10nopush0.7737.pth')
imgdir3 = base_dir + 'distfullgrad_protomax_448_2000_crs1divcs0pl22pl414sep8e-2dist0.1/img/epoch-10/'
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet3 = ppnet3.cuda()
ppnet3_multi = torch.nn.DataParallel(ppnet3)

# construct the model
ppnet4 = torch.load('saved_models/vgg19_bn/distfullgrad_protomax_448_2000_crs1divcs0pl22pl414sep8e-2dist1.0/10nopush0.7737.pth')
imgdir4 = base_dir + 'distfullgrad_protomax_448_2000_crs1divcs0pl22pl414sep8e-2dist1.0/img/epoch-10/'
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet4 = ppnet4.cuda()
ppnet4_multi = torch.nn.DataParallel(ppnet4)

# construct the model
ppnet5 = torch.load('saved_models/vgg19_bn/distfullgrad_classmax_448_2000_crs1divcs0pl22pl414sep8e-2dist1.0/10nopush0.7677.pth')
imgdir5 = base_dir + 'distfullgrad_classmax_448_2000_crs1divcs0pl22pl414sep8e-2dist1.0/img/epoch-10/'
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet5 = ppnet5.cuda()
ppnet5_multi = torch.nn.DataParallel(ppnet5)

ppnets = [ppnet1_multi, ppnet2_multi, ppnet3_multi, ppnet4_multi, ppnet5_multi]
imgdirs = [imgdir1, imgdir2, imgdir3, imgdir4, imgdir5]

class_specific = True

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

writer = SummaryWriter('runs/test')

from helpers import class_act_maps, find_high_activation_values, get_self_act_batch

for i, (images, labels) in enumerate(train_loader): 
  images = images.cuda()
  labels = labels.cuda()
  with torch.no_grad():
    for j in range(len(ppnets)):
      inputs = images.clone()
      targets = labels.clone()
      protomax_act_maps = class_act_maps(inputs, ppnets[j], targets, 200, unnormalize, classmax=False)
      protomax_grid = utils.make_grid(protomax_act_maps, 10)
      utils.save_image(protomax_grid, str(j) + '_protomax_act_maps.jpg')
    
      classmax_act_maps = class_act_maps(inputs, ppnets[j], targets, 200, unnormalize, classmax=True)
      classmax_grid = utils.make_grid(classmax_act_maps, 10)
      utils.save_image(classmax_grid, str(j) + '_classmax_act_maps.jpg')

      self_act_maps = get_self_act_batch(ppnets[j], targets, imgdirs[j], '/usr/xtmp/brandon_zhao/PPNet-Spike')
      self_act_grid = utils.make_grid(self_act_maps, 10)
      utils.save_image(self_act_grid, str(j) + '_self_act_maps.jpg')
  break