import torch

base_architecture = 'vgg19_bn'
img_size = 224
prototype_shape = (4000, 128, 1, 1)
num_global_prototypes = 2000
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
pruned = False

pos_weight = 1.0
neg_weight = -1.0

true_midpoint = True

twostep=True #whether or not to use a 2-step process
twostepk=5 #k value to use for top-k classes
global_neg=True #whether or not to include negative weights for global prototypes
deneg_global=True #whether or not to remove negative weights for global prototypes when a local prototype is added
add_sym=True #whether or not to add local prototypes symmetrically (two for each class pairing)
normalize_pos=False#whether or not to normalize positive last layer weights when adding local prototypes
normalize_neg=False#whether or not to normalize negative last layer weights when adding local prototypes
sep_crsent = False#splits the cross-entropy backprop into two parts: 1 cross entropy on global only logits, then another cross entropy considering only the local prototypes on the top k predicted classes from global logits
sep_crsent_k = -1
#normalize code still needs some work to be compatible with custom last layer weights (!= 1.0 or -0.5)

num_prototypes_to_add = 200 #number of local prototypes to add: if 0, does control run with only global prototypes
proto_k = -1
by_class=False



group_wrong_weights = False
dropout_p = 0.

sep_margin = 2.
clamp_prototypes = True
freeze_zeros = True
hard_ll = True
neg_local = True

one_vs_most = False

crop = True
if crop:
  data_path = '/usr/xtmp/brandon_zhao/datasets/cub200_cropped/'
  train_dir = data_path + 'train_cropped_augmented/'
  test_dir = data_path + 'test_cropped/'
  train_eval_dir = data_path + 'train_cropped/'
else:
  data_path = '/usr/xtmp/cfchen/datasets/cub200/'
  train_dir = data_path + 'train_augmented/'
  test_dir = data_path + 'test/'
  train_eval_dir = data_path + 'train/'

cropstr = 'crop' if crop else 'uncrop'

train_push_dir = train_eval_dir
if torch.cuda.get_device_name(0) == 'GeForce RTX 2080 Ti': 
  train_batch_size = 20
elif torch.cuda.get_device_name(0) == 'Tesla V100-PCIE-32GB': 
  train_batch_size = 30
else: 
  train_batch_size = 20
test_batch_size = 30
train_push_batch_size = 60

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

'''
coefs = {
  'crs_ent': 1,
  'clst': 0.8,
  'sep': -0.08,
  'l1': 1e-4,
  'dist': 1.0,
  'clst_l': 0.1,
  'sep_l': -0.01
}
'''

coefs = {
  'crs_ent': 0.5,
  'crs_ent_g': 0.5,
  'clst': 0.8,
  'sep': -0.08,
  'l1': 1e-4,
  'dist': 1.0,
  'tb_l': 1.0
}

padding = 2
padding2 = 14
dist_start = 0
distnormalize = True
dist_classmax = False

all_class_specific = True

maxstr = 'classmax' if dist_classmax else 'protomax'
specstr = 'allcs' if all_class_specific else 'halfcs'
stepstr = '2step' + str(twostepk) if twostep else '1step'

experiment_run = '3ce' + stepstr + '_' + str(img_size) + cropstr + 'add' + str(num_prototypes_to_add) + 'global_neg' + str(global_neg) + 'deneg_global' + str(deneg_global)  
#experiment_run = 'control_crop_' + str(img_size)
#experiment_run = 'control_deneg'
#experiment_run = str(img_size) + 'control_deneg_sym'
#experiment_run = 'scratch_random_224'

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 100
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
