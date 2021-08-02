import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field import compute_proto_layer_rf_info_v2

from settings import true_midpoint, add_sym

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape, num_global_prototypes,
                 proto_layer_rf_info, num_classes, init_weights=True, global_neg=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        #model will now ignore first entry of prototype_shape
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.num_global_prototypes = num_global_prototypes
        self.global_prototypes_per_class = int(self.num_global_prototypes / self.num_classes)
        self.num_local_prototypes = 0
        self.global_neg = global_neg

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity_negc = torch.cat((torch.ones((self.num_global_prototypes, self.num_classes)) * -0.5, torch.zeros((self.num_prototypes - self.num_global_prototypes, self.num_classes))), dim=0)

        for j in range(self.num_global_prototypes):
            self.prototype_class_identity_negc[j, j // self.global_prototypes_per_class] = 1

        self.prototype_class_identity = torch.clamp(self.prototype_class_identity_negc, min=0, max=1)

        _, self.pci_list = torch.max(self.prototype_class_identity, dim=1)

        self.prototype_class_identity_negc_weighted = self.prototype_class_identity_negc

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        if init_weights:
            self._initialize_weights()

    def get_new_pci(self, separation, num_prototypes_to_add, deneg_global=True, add_sym=True, pos_weight = 1.0, neg_weight = -1.0):
        #update prototype_class_identity and prototype_class_identity_negc
        #if add sym there should be an even number of prototypes added
        if add_sym:
            separation += torch.t(separation)
        _, indices = torch.sort(separation.flatten(), descending=True)
        indices = indices[:num_prototypes_to_add]
        indi = indices // self.num_classes
        indj = indices % self.num_classes
        pci_new = self.prototype_class_identity_negc.clone()
        for i in range(indices.size(0)): 
            if indi[i] == indj[i]: 
                print('ERROR! FOUND MAX SEPARATION VALUE BETWEEN CLASS AND ITSELF')
            if deneg_global: 
                #this now works with the current implementation of add_sym as well
                pci_list_global = self.pci_list[:self.num_global_prototypes]
                newmask = torch.cat((pci_list_global, -1. * torch.ones(self.num_prototypes - self.num_global_prototypes)))
                pci_new[newmask==indj[i].cpu(), indi[i]] = 0.
            if num_prototypes_to_add > 0:
                if add_sym:
                    pci_new[i + self.num_global_prototypes + self.num_local_prototypes][indi[i]] = neg_weight
                    pci_new[i + self.num_global_prototypes + self.num_local_prototypes][indj[i]] = pos_weight
                else: 
                    pci_new[i + self.num_global_prototypes + self.num_local_prototypes][indi[i]] = -5.0
                    pci_new[i + self.num_global_prototypes + self.num_local_prototypes][indj[i]] = 1.0
            
        return pci_new

    def add_local_prototypes(self, pci_new, num_prototypes_to_add, normalize_pos=True, normalize_neg=True): 
        #separation is a 200x200 matrix where separation[i][j] is the total separation cost incurred for images of class i against prototypes of class j
        #for the highest separation costs (lowest distances), prototypes of class j are added  with negative connections to class i
        
        #todo: you should probably turn off midpoint init for now
        #midpoint_vectors = self.initialize_midpoint(pci_new[self.num_local_prototypes:self.num_local_prototypes+num_prototypes_to_add, :], true_midpoint)
        #self.prototype_vectors.data[self.num_local_prototypes:self.num_local_prototypes+num_prototypes_to_add, :, :, :] = midpoint_vectors

        self.num_local_prototypes += num_prototypes_to_add
        self.prototype_class_identity_negc = pci_new
        self.prototype_class_identity = torch.clamp(self.prototype_class_identity_negc, min=0, max=1)
        _, self.pci_list = torch.max(self.prototype_class_identity, dim=1)
        #if normalize_pos or normalize_neg: 
        self.normalize_weights(normalize_pos = normalize_pos, normalize_neg = normalize_neg)

    def initialize_from_pci(self, pci): 
        self.prototype_class_identity_negc = pci
        self.prototype_class_identity = torch.clamp(self.prototype_class_identity_negc, min=0, max=1)
        _, self.pci_list = torch.max(self.prototype_class_identity, dim=1)
        #assuming pci tensor is full
        self.num_local_prototypes = self.num_prototypes - self.num_global_prototypes
        self.normalize_weights(normalize_pos=False, normalize_neg = False)

    def initialize_midpoint(self, pci_new, true_midpoint = True):
        #ASSUMES LATENT PATCH SIZE IS 1X1
        with torch.no_grad():
            global_prototype_vectors = self.prototype_vectors[:self.num_global_prototypes, :, :, :].view(self.num_classes, self.global_prototypes_per_class, self.prototype_shape[1], self.prototype_shape[2], self.prototype_shape[3])
            global_prototype_means = torch.mean(global_prototype_vectors, dim=1).squeeze()
            if true_midpoint:
                proto_class_indicator = torch.clamp(torch.abs(pci_new * 2), min=0, max=1)
                midpoint_vectors = torch.matmul(proto_class_indicator.cuda(), global_prototype_means) / 2
            else: 
                proto_class_indicator = torch.abs(pci_new)
                midpoint_vectors = torch.matmul(proto_class_indicator.cuda(), global_prototype_means) / 1.5
            midpoint_vectors = midpoint_vectors.unsqueeze(2).unsqueeze(3)
            return midpoint_vectors

    def normalize_weights(self, normalize_pos=True, normalize_neg=False):
        #normalize fixed positive weights in self.pci_negc_weighted
        if normalize_pos: 
            resize_factor = (torch.sum(self.prototype_class_identity, dim=0) / self.global_prototypes_per_class).unsqueeze(0)
        else: 
            resize_factor = 1.
        pci_neg = torch.clamp(self.prototype_class_identity_negc, min=self.prototype_class_identity_negc.min(), max=0.)
        if normalize_neg: 
            resize_factor_neg = (torch.sum(pci_neg, dim=0) / (-0.5 * (self.num_global_prototypes - self.global_prototypes_per_class))).unsqueeze(0)
        else: 
            resize_factor_neg = 1.
        self.prototype_class_identity_negc_weighted = (self.prototype_class_identity / resize_factor) + (pci_neg / resize_factor_neg)
        #update last layer connection
        self.set_last_layer_incorrect_connection()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        #print(x[0][0], flush = True)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''

        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances
    '''
    def get_prototype_vectors(self): 
        if len(self.local_prototype_vectors) > 0:
            local_prototypes = torch.cat([param for param in self.local_prototype_vectors], dim=0)
            return torch.cat((self.global_prototype_vectors, local_prototypes), dim=0)
        else:
            print(len(self.local_prototype_vectors))
            return self.global_prototype_vectors
    '''
    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        only do this operation for prototypes that have been assigned already
        '''
        num_prototypes = self.num_global_prototypes + self.num_local_prototypes

        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones[:num_prototypes, :, :, :])

        p2 = self.prototype_vectors[:num_prototypes, :, :, :] ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)
        xp = F.conv2d(input=x, weight=self.prototype_vectors[:num_prototypes, :, :, :])
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        max_dist = (self.prototype_shape[1]
                            * self.prototype_shape[2]
                            * self.prototype_shape[3])
        
        dist_2 = torch.ones((distances.size(0), self.num_prototypes - num_prototypes, distances.size(2), distances.size(3))).cuda() * max_dist
        return torch.cat((distances, dist_2), dim=1)

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        batch_size = x.size(0)
        distances = self.prototype_distances(x)

        distances = distances.view(batch_size, self.num_prototypes, -1)
        min_distances, minind = torch.min(distances, 2)

        prototype_activations = self.distance_2_similarity(min_distances.view(-1, self.num_prototypes))
        logits = self.last_layer(prototype_activations)

        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)

        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]
        self.prototype_class_identity_negc = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength

        last_layer_global_weights = correct_class_connection * positive_one_weights_locations
        '''
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias 
        if self.global_neg: 
            self.last_layer.weight.data.copy_(
                torch.t(self.prototype_class_identity_negc_weighted))
        else: 
            foo = torch.cat((torch.abs(self.prototype_class_identity_negc_weighted[:self.num_global_prototypes, :]), self.prototype_class_identity_negc_weighted[self.num_global_prototypes:, :]), dim=0)
            self.last_layer.weight.data.copy_(
                torch.t(foo)
                )
    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection()



def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(4000, 128, 1, 1), num_global_prototypes=2000, num_classes=200, global_neg=True,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 num_global_prototypes=num_global_prototypes,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 global_neg = global_neg,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

