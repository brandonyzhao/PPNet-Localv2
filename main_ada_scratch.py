import os
import shutil

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test_ada as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('--debug', dest='debug', action = 'store_true')
parser.add_argument('-exp', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.set_defaults(debug=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, global_neg, \
                     prototype_activation_function, add_on_layers_type

experiment_run = args.exp[0]
model_file = args.model[0]

print('experiment run: ', experiment_run)

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

print(base_architecture)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size, crop

if crop: 
    print('loading cropped images')
else: 
    print('loading uncropped images')

if args.debug: 
    train_batch_size = 5
    test_batch_size = 5

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

ppnet_donor = torch.load(model_dir + model_file)
pci = ppnet_donor.prototype_class_identity_negc.clone()
del ppnet_donor

import model_ada_dist as model_ada

# construct the model
ppnet = model_ada.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              global_neg=global_neg,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
ppnet.initialize_from_pci(pci)
'''
import pickle
pci = pickle.load(open('pci_dump_sym.p', 'rb')).cpu()
ppnet.initialize_from_pci(pci)
'''
#if prototype_activation_function == 'linear':
#   ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True



# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

from settings import num_prototypes_to_add, deneg_global

writer = SummaryWriter('./runs/' + experiment_run)
iter = 0

import resource
from helpers import sizeof_fmt
import pickle

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))
    log('max memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, iter, _  = tnt.train(model=ppnet_multi, dataloader=train_loader, writer=writer, iter=iter, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, debug=args.debug)
        #pci_new = ppnet_multi.module.get_new_pci(sep_cost_matrix, num_prototypes_to_add, deneg_global)
        #del sep_cost_matrix
    else:      
        tnt.joint(model=ppnet_multi, log=log)
        _, iter, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, writer=writer, iter=iter, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, debug=args.debug)
        #pci_new = ppnet_multi.module.get_new_pci(sep_cost_matrix, num_prototypes_to_add, deneg_global)
        #del sep_cost_matrix
        joint_lr_scheduler.step()
    log('total prototypes: \t{0}'.format(ppnet.num_global_prototypes + ppnet.num_local_prototypes))

    accu, iter = tnt.test(model=ppnet_multi, dataloader=test_loader, writer=writer, iter=iter,
                    class_specific=class_specific, log=log, debug=args.debug)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='scr' + str(epoch) + 'nopush', accu=accu,
                                target_accu=0.60, log=log)
    start = time.time()
    '''
    if ppnet.num_prototypes > ppnet.num_local_prototypes + ppnet.num_global_prototypes + num_prototypes_to_add:
        log('Adding {0} local prototypes'.format(num_prototypes_to_add))
        ppnet_multi.module.add_local_prototypes(pci_new, num_prototypes_to_add)
        ppnet_multi = ppnet_multi.cuda()
        end = time.time()
        log('\tproto add time: \t{0}'.format(end -  start))
        log('dumping prototype class identity')
        pickle.dump(ppnet.prototype_class_identity_negc, open('pci_dump.p', 'wb'))

        log('warming new prototypes')
        tnt.warm_only(model=ppnet_multi, log=log)
        _, iter, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, writer=writer, iter=iter, optimizer=warm_optimizer, new_protos = True,
                  class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, debug=args.debug)
    else:
        log('Reached prototype limit, not assigning prototypes')
    '''
    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, _ = tnt.test(model=ppnet_multi, dataloader=test_loader, iter=(epoch*22)+1,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(200):
                log('iteration: \t{0}'.format(i))
                _, _, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, writer=writer, iter=(epoch*22)+2+i, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log, epoch=epoch)
                accu, _ = tnt.test(model=ppnet_multi, dataloader=test_loader, writer=writer, iter=(epoch*22)+2+i,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
    
logclose()

