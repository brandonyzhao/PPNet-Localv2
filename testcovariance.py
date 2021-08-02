import os
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from settings import val_dir, test_dir, test_batch_size, img_size, num_classes, num_train_epochs, train_push_dir, train_push_batch_size
from preprocess import mean, std, preprocess_input_function
from log import create_logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-batch_size', nargs=1, type=int, default=[test_batch_size])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

test_batch_size = args.batch_size[0]

log, logclose = create_logger(log_filename = os.path.join('/usr/xtmp/brandon_zhao/PPNet_unified', 'train.log'))

normalize = transforms.Normalize(mean=mean, std=std)

val_dataset = datasets.ImageFolder(
	val_dir,
	transforms.Compose([
		transforms.Resize(size = (img_size, img_size)),
		transforms.ToTensor(),
		normalize,
	]))
test_dataset = datasets.ImageFolder(
	test_dir,
	transforms.Compose([
		transforms.Resize(size = (img_size, img_size)),
		transforms.ToTensor(),
		normalize,
	]))

log('Test/Val Batch Size: {0}'.format(test_batch_size))

test_loader = torch.utils.data.DataLoader(
	test_dataset, batch_size=test_batch_size, shuffle=False,
	num_workers=4, pin_memory=False)
val_loader = torch.utils.data.DataLoader(
	val_dataset, batch_size=test_batch_size, shuffle=False,
	num_workers=4, pin_memory=False)

import model

from settings import base_architecture, img_size, prototype_shape, num_classes, \
					 prototype_activation_function, add_on_layers_type

pp1_filename = './saved_models/vgg19_bn/448_local_clamp_withreg10neg10pos1.0negw-0.5dist0.0/10_29push0.7996.pth'

print('pp1: ', pp1_filename)

pp1 = torch.load(pp1_filename)
pp1 = pp1.cuda()
pp1 = torch.nn.DataParallel(pp1)


def logitobservations(pp1, loader):
	pp1.eval()
	with torch.no_grad():
		logits = []
		labelsbig = []
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			log1, _, _, _, _ = pp1(images)
			logits.append(log1)
			labelsbig.append(labels)
		logits = torch.cat(logits, 0)
		labelsbig = torch.cat(labelsbig, 0)

	return logits.cpu().numpy(), labelsbig.cpu().numpy()

logits, labels = logitobservations(pp1, val_loader)
_, counts = np.unique(labels, return_counts=True)
print(counts.shape)
cov_matrix = np.cov(logits, rowvar=False, aweights=(1/counts[labels]))

print(cov_matrix)

import pickle

pickle.dump(cov_matrix, open('cov_matrix_test.p', 'wb'))

