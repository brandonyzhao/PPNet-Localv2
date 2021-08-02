import os
import shutil

import torch
import torch.utils.data
#from tensorboardX import SummaryWriter
import torch.distributed as distributed
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
from datetime import timedelta

#writing this script only for 1 gpu/node, 1 process/node
#use the other main for dataparallel

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
#DDP Arguments
parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
#print(os.environ['CUDA_VISIBLE_DEVICES'])
    
print(args.rank)
#DDP Stuff
args.world_size = os.environ['WORLD_SIZE']
master = (args.rank == 0)
print(int(os.environ['WORLD_SIZE']))
print(master)
import socket
if master: 
    master_addr = socket.gethostname() 
    with open('masternode.txt', 'w') as f:
        f.write(master_addr)
    time.sleep(10)
    store = distributed.TCPStore(master_addr, 12340, int(os.environ['WORLD_SIZE']), master, timedelta(seconds=30))
else: 
    time.sleep(10)
    with open('masternode.txt', 'r') as f: 
        master_addr = f.readline()
    store = distributed.TCPStore(master_addr, 12340, int(os.environ['WORLD_SIZE']), master)
print('successfully initialized store')
distributed.init_process_group(backend=args.dist_backend, store=store,
                        world_size=int(args.world_size), rank=int(args.rank))
print('successfully initialized process group')

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

experiment_run = experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
#don't log if not on master gpu
if args.rank!=0: 
    def log_pass(*args): 
        pass
    log = log_pass
log('experiment run: ', experiment_run)
log(base_architecture)
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

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size,
    num_workers=4, pin_memory=True, sampler = train_sampler, drop_last = True)
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

import model_ada

# construct the model
ppnet = model_ada.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
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

from settings import num_prototypes_to_add

writer = None
iter = 0

# train the model
log('start training')
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))
    log('total prototypes: \t{0}'.format(ppnet.num_global_prototypes + ppnet.num_local_prototypes))
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, iter, sep_cost_matrix  = tnt.train(model=ppnet_multi, dataloader=train_loader, writer=writer, iter=iter, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, epoch=epoch)
        pci_new = ppnet_multi.module.get_new_pci(sep_cost_matrix, num_prototypes_to_add)
        del sep_cost_matrix
    else:     
        tnt.joint(model=ppnet_multi, log=log)
        _, iter, sep_cost_matrix = tnt.train(model=ppnet_multi, dataloader=train_loader, writer=writer, iter=iter, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log, epoch=epoch)
        pci_new = ppnet_multi.module.get_new_pci(sep_cost_matrix, num_prototypes_to_add)
        del sep_cost_matrix
        joint_lr_scheduler.step()

    accu, iter = tnt.test(model=ppnet_multi, dataloader=test_loader, writer=writer, iter=iter,
                    class_specific=class_specific, log=log)

    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)
    start = time.time()
    if ppnet.num_prototypes > ppnet.num_local_prototypes + ppnet.num_global_prototypes:
        log('Adding {0} local prototypes'.format(num_prototypes_to_add))
        ppnet_multi.module.add_local_prototypes(pci_new, num_prototypes_to_add)
        ppnet_multi = ppnet_multi.cuda()
        end = time.time()
        log('\tproto add time: \t{0}'.format(end -  start))
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
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, writer=writer, iter=(epoch*22)+2+i, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log, epoch=epoch)
                accu, _ = tnt.test(model=ppnet_multi, dataloader=test_loader, writer=writer, iter=(epoch*22)+2+i,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
    '''
logclose()

