import os
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

import pickle
import numpy as np

from settings import train_dir, test_dir, test_batch_size, img_size, num_classes, num_train_epochs, train_push_dir, train_push_batch_size
#from settings import val_dir
from preprocess import mean, std, preprocess_input_function
from log import create_logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])


log, logclose = create_logger(log_filename = os.path.join('/usr/xtmp/brandon_zhao/PPNet_unified', 'train.log'))

normalize = transforms.Normalize(mean=mean, std=std)

import model

from settings import base_architecture, img_size, prototype_shape, num_classes, \
					 prototype_activation_function, add_on_layers_type

from helpers import list_of_distances

pp1_filename = './saved_models/vgg19_bn/448_local_clamp_withreg10neg10pos1.0negw-0.5dist0.0/10_29push0.7996.pth'

print('pp1: ', pp1_filename)

pp1 = torch.load(pp1_filename)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

X_torch = torch.squeeze(pp1.prototype_vectors)
X_mean = torch.mean(X_torch.view(pp1.num_classes, -1, X_torch.size(1)), dim=1)
X_numpy = X_torch.detach().cpu().numpy()
X_mean_numpy = X_mean.detach().cpu().numpy()

#y_torch = torch.tensor([pp1.prototype_class_identity[i].nonzero()[0] for i in range(X_torch.size(0))]).cpu()
#y_numpy = y_torch.numpy()

X_mean_distances = list_of_distances(X_mean, X_mean)
pickle.dump(X_mean_distances, open('local_distances.p', 'wb'))
'''
clf = LinearDiscriminantAnalysis()
clf.fit(X_numpy, y_numpy)

means_transformed = clf.transform(X_mean_numpy)
kmeans = KMeans(n_clusters=50).fit(means_transformed)
kmeans2 = KMeans(n_clusters=50).fit(X_mean_numpy)
print(kmeans.labels_)

for i in range(np.amax(kmeans.labels_) + 1): 
	print(len(np.where(kmeans.labels_ == i)[0].tolist()))
	print(str(i) + ' : ' + str(np.where(kmeans.labels_ == i)[0].tolist()))

print(kmeans2.labels_)

for i in range(np.amax(kmeans2.labels_) + 1): 
	print(len(np.where(kmeans2.labels_ == i)[0].tolist()))
	print(str(i) + ' : ' + str(np.where(kmeans2.labels_ == i)[0].tolist()))


out = np.zeros((200, 200))

for i in range(200): 
	for j in range(200): 
		if i == j: 
			out[i][j] = 0
		out[i][j] = np.linalg.norm(means_transformed[i] - means_transformed[j])

pickle.dump(torch.tensor(kmeans.labels_).long(), open('LDA_members.p', 'wb'))
pickle.dump(torch.tensor(kmeans2.labels_).long(), open('prototype_members.p', 'wb'))
'''