import os
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')

from settings import train_dir, test_dir, train_eval_dir, train_batch_size, test_batch_size, img_size, num_classes, num_train_epochs, train_push_dir, train_push_batch_size
from preprocess import mean, std, preprocess_input_function
from log import create_logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-batch_size', nargs=1, type=int, default=[train_batch_size])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

train_batch_size = args.batch_size[0]

log, logclose = create_logger(log_filename = os.path.join('/usr/xtmp/brandon_zhao/PPNet-Spike', 'train.log'))

normalize = transforms.Normalize(mean=mean, std=std)

test_dataset = datasets.ImageFolder(
	test_dir,
	transforms.Compose([
		transforms.Resize(size = (img_size, img_size)),
		transforms.ToTensor(),
		normalize,
	]))

train_dataset = datasets.ImageFolder(
	train_dir, 
	transforms.Compose([
		transforms.Resize(size = (img_size, img_size)),
		transforms.ToTensor(),
		normalize,
	]))

train_eval_dataset = datasets.ImageFolder(
	train_eval_dir, 
	transforms.Compose([
		transforms.Resize(size = (img_size, img_size)),
		transforms.ToTensor(),
		normalize,
	]))

push_dataset = datasets.ImageFolder(
	train_eval_dir, 
	transforms.Compose([
		transforms.Resize(size = (img_size, img_size)),
		transforms.ToTensor(),
	]))

log('Train Batch Size: {0}'.format(train_batch_size))
log('Test/Eval Batch Size: {0}'.format(test_batch_size))

train_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=train_batch_size, shuffle = True,
	num_workers=4, pin_memory=False)

train_eval_loader = torch.utils.data.DataLoader(
	train_eval_dataset, batch_size=train_batch_size, shuffle = False,
	num_workers=4, pin_memory=False)

test_loader = torch.utils.data.DataLoader(
	test_dataset, batch_size=1, shuffle=False,
	num_workers=4, pin_memory=False)

train_push_loader = torch.utils.data.DataLoader(
	push_dataset, batch_size=train_batch_size, shuffle = False,
	num_workers=4, pin_memory=False)

import model

from settings import base_architecture, img_size, prototype_shape, num_classes, \
					 prototype_activation_function, add_on_layers_type

pp1_filename = './saved_models/vgg19_bn/448_local_clamp10neg10pos1.0negw-0.5dist0.0/10_54push0.8053.pth'

print('pp1: ', pp1_filename)

pp1 = torch.load(pp1_filename)

pp1 = pp1.cuda()

pp1 = torch.nn.DataParallel(pp1)

import train
import train_and_test as tnt

import math

ipe = math.ceil(len(train_loader.dataset) / train_batch_size)

prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'
class_specific = True

from tensorboardX import SummaryWriter
writer = SummaryWriter('trash')

cutoff = 5
iter = 0

from train import get_sparsities

sparsities = get_sparsities(pp1, test_loader)
sparsities = torch.tensor(sparsities)
print('mean sparsity', torch.mean(sparsities.float()))

import pickle

pickle.dump(sparsities, open('sparsities.p', 'wb'))