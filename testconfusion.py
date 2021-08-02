import os
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
	test_dataset, batch_size=train_batch_size, shuffle=False,
	num_workers=4, pin_memory=False)

train_push_loader = torch.utils.data.DataLoader(
	push_dataset, batch_size=train_batch_size, shuffle = False,
	num_workers=4, pin_memory=False)

import model

from settings import base_architecture, img_size, prototype_shape, num_classes, \
					 prototype_activation_function, add_on_layers_type

pp1_filename = './saved_models/vgg19_bn/448_local_clamp_withreg10neg10pos1.0negw-0.5dist0.0/10_29push0.7996.pth'

print('pp1: ', pp1_filename)

pp1 = torch.load(pp1_filename)
pp1 = pp1.cuda()
pp1 = torch.nn.DataParallel(pp1)

def classaccuracy(pp1, loader, train=True, log=print, k=-1):
	top_proto = False
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	with torch.no_grad():
		correct1 = torch.zeros(num_classes)
		total = torch.zeros(num_classes)
		confusion1 = torch.zeros((num_classes, num_classes))
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			log1, min_distances, activation_maps, _, _ = pp1(images)

			if k == -1: 
				batchcorr1, batchtotal1, confusion1 = batch_class_acc(labels, log1, confusion1)
			else: 
				batchcorr1, batchtotal1 = batch_class_acc_topk(labels, log1, k)		


			correct1 += batchcorr1
			total += batchtotal1
		
		acc1 = 100 * correct1 / total
		'''
		log('Level 1 Overall Accuracy: {0}'.format(torch.mean(acc1)))

		for i in range(num_classes): 
			log('Class ' + str(i) + ' ' + tt + 'accuracy scale 1(logit sum): {0}'.format(acc1[i]))

		print('Top 10 worst classes: ')

		sorted1, sortind1 = torch.sort(acc1, descending = False)

		print('1st level')
		for i in range(50): 
			log('Class ' + str(sortind1[i]) + ': ' + str(sorted1[i]) + ' (total examples): ' + str(total[sortind1[i]]))
			classnum = sortind1[i]
			foo, bar = torch.sort(confusion1[classnum], descending = True)
			for j in range(10): 
				print('Classified as class ' + str(bar[j]) + ' frequency: ' + str(foo[j]))
		'''
	return acc1, confusion1

acc1, confusion1 = train.classaccuracy(pp1, test_loader, False, log, k=-1)
confusion_matrix = confusion1.cpu().numpy()

import pickle

pickle.dump(confusion_matrix, open('confusion_matrix_withreg.p', 'wb'))

