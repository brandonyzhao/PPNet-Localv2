import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def find_high_activation_values(prototype_network_parallel, images, num_activations = 4): 
    batch_size = images.size(0)
    with torch.no_grad(): 
        images = images.cuda()
        _, proto_dist_torch = prototype_network_parallel.module.push_forward(images)
        #print(proto_dist_torch.size())
        proto_dist_torch = proto_dist_torch.view(batch_size, -1)
        #print(proto_dist_torch.size())
        proto_dist_sort, _ = torch.sort(proto_dist_torch, 1)
        proto_dist_sort = proto_dist_sort[:, :num_activations]
        proto_act_sort = prototype_network_parallel.module.distance_2_similarity(proto_dist_sort)
    return proto_act_sort

def class_act_maps(search_batch_input, 
                   prototype_network_parallel, 
                   search_y,
                   num_classes, 
                   unnormalize=None,
                   preprocess_input_function=None,
                   writer=None,
                   classmax = False):
    prototype_network_parallel.eval()
    if preprocess_input_function is not None: 
        search_batch = preprocess_input_function(search_batch_input)
    else: search_batch = search_batch_input
    with torch.no_grad():
        search_batch = search_batch.cuda()
        _, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    batch_size = search_batch_input.size(0)
    original_img_size = search_batch_input.size(-1)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    prototypes_per_class = int(n_prototypes / num_classes)

    proto_act_torch = prototype_network_parallel.module.distance_2_similarity(proto_dist_torch)

    act_h = proto_act_torch.size(2)
    act_w = proto_act_torch.size(3)
    proto_act_torch = proto_act_torch.view(batch_size , num_classes , prototypes_per_class , act_h , act_w)
    #proto_act_torch = proto_act_torch[:, search_y, :, :, :]
    for i in range(search_batch.size(0)): 
        search_batch[i] = unnormalize(search_batch[i])

    search_batch_ = np.copy(search_batch.detach().cpu().numpy())
    proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

    del proto_dist_torch

    class_act_maps = []

    for j in range(batch_size): 
        original_img_j = search_batch_[j]
        original_img_j = np.transpose(original_img_j, (1, 2, 0))

        target_class = proto_act_[j,search_y[j], :, :, :]
        for i in range(prototypes_per_class): 
            proto_act_img_j = target_class[i]

            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                                 interpolation=cv2.INTER_CUBIC)
            if classmax: 
                rescaled_act_img_j = upsampled_act_img_j - np.amin(target_class)
                rescaled_act_img_j = rescaled_act_img_j / (np.amax(target_class) - np.amin(target_class))
            else: 
                rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
            class_act_maps.append(overlayed_original_img_j)

    class_act_maps = np.array(class_act_maps)
    class_act_maps = np.transpose(class_act_maps, (0, 3, 1, 2))
    return torch.tensor(class_act_maps)

def get_self_act_batch(prototype_network_parallel, search_y, image_dir, original_dir): 
    prototype_class_identity = prototype_network_parallel.module.prototype_class_identity
    num_prototypes = prototype_network_parallel.module.num_prototypes
    prototypes_per_class = int(num_prototypes / prototype_network_parallel.module.num_classes)
    prototypes_of_correct_class = torch.t(prototype_class_identity[:,search_y])

    os.chdir(image_dir)

    self_act_batch = []

    for i in range(search_y.size(0)): 
        for j in range(num_prototypes):
            if prototypes_of_correct_class[i][j] == 1: 
                imgname = 'prototype-img-original_with_self_act' + str(j) + '.png'
                img = Image.open(imgname)
                img_torch = TF.to_tensor(img)
                self_act_batch.append(img_torch.unsqueeze(0)[:, :3, :, :])

    self_act_batch_torch = torch.cat(self_act_batch, 0)

    os.chdir(original_dir)
    return self_act_batch_torch

def invert(mean, std): 
    unmean = [-1 * mean[i] / std[i] for i in range(len(mean))]
    unstd = [1 / std[i] for i in range(len(std))]
    return unmean, unstd

def calculate_sparsity(image, model, label): 
    logits, min_distances,_,_,_ = model(image)
    _, predicted = torch.max(logits.data, 1)
    print('predicted: ', predicted[0])
    print('label: ', label)
    if predicted[0] != label: 
        print('incorrect')
        return False, -1
    else: 
        print('correct')
        prototype_activations = model.module.distance_2_similarity(min_distances.squeeze())
        prototype_scores = prototype_activations.cuda() * model.module.last_layer.weight[label].squeeze().cuda() * model.module.prototype_class_identity[:, label].squeeze().cuda()
        prototype_scores, _ = torch.sort(prototype_scores, descending = True)
        print('prototype scores: ', prototype_scores)
        done = False
        foo = logits.clone().squeeze()
        foo, _ = torch.sort(foo, descending = True)
        next_highest = foo[1]
        print('next highest: ', next_highest)
        scoresum = 0
        count = 0
        while not done: 
            if prototype_scores[count] == 0.0: 
                print('ran out of prototypes!')
                exit()
            scoresum += prototype_scores[count]
            print('scoresum: ', scoresum)
            if scoresum > next_highest: 
                done = True
            count += 1
        return True, count
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

