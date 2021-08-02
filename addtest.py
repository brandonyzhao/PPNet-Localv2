import torch
import model_ada
from settings import base_architecture, img_size, prototype_shape, num_classes, prototype_activation_function, add_on_layers_type, experiment_run, num_global_prototypes
ppnet = model_ada.construct_PPNet(base_architecture=base_architecture,pretrained=True, img_size=img_size,prototype_shape=prototype_shape,num_global_prototypes=num_global_prototypes,num_classes=num_classes,prototype_activation_function=prototype_activation_function,add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()
print(torch.sum(ppnet.prototype_class_identity_negc[2000:, :]))
sep_cost_matrix = torch.rand((200,200)).cuda()
pci_new = ppnet.get_new_pci(sep_cost_matrix, 100)
ppnet.add_local_prototypes(pci_new, 100)
ppnet = ppnet.cuda()
print(torch.sum(ppnet.prototype_class_identity_negc[2000:, :]))
print(torch.sum(ppnet.prototype_class_identity_negc[2100:, :]))
print(torch.sum(ppnet.prototype_class_identity_negc[2200:, :]))
_ = ppnet(torch.rand((1,3,448,448)).cuda())

print('done')