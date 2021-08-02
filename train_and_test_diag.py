import time
import torch
import sys

from helpers import list_of_distances, make_one_hot, sizeof_fmt
import pci_into_matrix

def test(model, model_ctrl, dataloader, proto_k = -1., by_class = False, log=print, debug=False):
	
	model.eval()
	model_ctrl.eval()

	start = time.time()
	n_examples = 0
	n_correct = 0
	n_correct_ctrl = 0
	n_correct_both = 0

	n_inc_local_ctrl = 0
	n_inc_local = 0

	pci = model.module.prototype_class_identity_negc
	pci_array = pci_into_matrix.convert(pci, model.module.num_global_prototypes)

	assert (torch.max(pci_array) == 1.), 'invalid pci array'
	assert (torch.min(pci_array) == 0.), 'invalid pci array'
	assert (torch.trace(pci_array) == 0.), 'invalid pci array'
	 
	for i, (image, label) in enumerate(dataloader):
		if debug: 
			print(i)
			if i > 0: 
				break
		input = image.cuda()
		target = label.cuda()

		with torch.no_grad():
			output_ctrl, _ = model_ctrl(input)
			output, _, _ = model(input, k=proto_k, labels=target, dist_loss = False, by_class=by_class)

			# evaluation statistics
			_, predicted = torch.max(output.data, 1)
			_, predicted_ctrl = torch.max(output_ctrl.data, 1)
			n_examples += target.size(0)
			n_correct += (predicted == target).sum().item()
			n_correct_ctrl += (predicted_ctrl == target).sum().item()
			n_correct_both += torch.logical_and(predicted==target, predicted_ctrl==target).sum().item()

			for i in range(target.size(0)): 
				if pci_array[predicted_ctrl[i]][target[i]] == 1.: 
					n_inc_local_ctrl += 1
				if pci_array[predicted[i]][target[i]] == 1.: 
					n_inc_local += 1

		del input
		del target
		del output
		del output_ctrl
		del predicted
		del predicted_ctrl
	
	end = time.time()
	
	log('\ttime: \t{0}'.format(end -  start))
	log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
	log('\taccu (ctrl): \t\t{0}%'.format(n_correct_ctrl / n_examples * 100))
	log('\taccu (overlap): \t\t{0}%'.format(n_correct_both / n_examples * 100))
	log('\terror from local prototype class pairs: \t\t{0}%'.format(n_inc_local / n_examples * 100))
	log('\terror from local prototype class pairs (ctrl): \t\t{0}%'.format(n_inc_local_ctrl / n_examples * 100))

def test_protos(model, dataloader, proto_k = -1., by_class = False, log=print, debug=False):
	
	model.eval()

	start = time.time()
	n_correct_local = 0
	n_examples_local = 0
	n_correct_local_both = 0
	n_correct = 0
	n_correct_local_both_global = 0
	n_correct_global = 0

	pci = model.module.prototype_class_identity_negc
	ngp = model.module.num_global_prototypes
	nlp = model.module.num_local_prototypes
	pci_array = pci_into_matrix.convert(pci, ngp, nlp)

	assert (torch.max(pci_array) == 1.), 'invalid pci array'
	assert (torch.min(pci_array) == 0.), 'invalid pci array'
	assert (torch.trace(pci_array) == 0.), 'invalid pci array'
	assert (torch.equal(pci_array, torch.t(pci_array))), 'local prototypes must have been added symmetrically'

	#check that prototype pairs were added consecutively
	foo = pci[ngp:, :].view(-1, 2, model.module.num_classes).sum(1)
	assert (torch.max(foo) == 0. and torch.min(foo) == 0.), 'prototype pairs should have been added consecutively'
	
	pci_pos = model.module.prototype_class_identity[ngp:, :].cuda()
	#pci_neg = torch.clamp(model.module.prototype_class_identity_negc[ngp:, :], -1., 0) / -1.
	#assert(torch.max(pci_neg) == 1.), 'checking pci is scaled properly'


	for i, (image, label) in enumerate(dataloader):
		if debug: 
			print(i)
			if i > 0: 
				break
		input = image.cuda()
		target = label.cuda()
		batch_size = target.size(0)

		with torch.no_grad():
			output, global_output, min_distances = model(input, k=proto_k, labels=target, dist_loss = False, by_class=by_class)

			_, counts = model.module.get_local_mask(global_output, k=2)
			#print('counts from local mask', counts)
			foo = min_distances[:, ngp:].view(batch_size, -1, 2)
			values, indices = torch.sort(foo, dim=2, descending=True)
			indices = indices.view(batch_size, -1).float()
			#indices is now a one-hot indicator matrix for the prototypes in each pair with the lowest min distance
			#indices: batch_size * num_local_prototypes
			#pci_pos: num_local_prototypes * num_classes
			#proto_outputs: batch_size * num_classes
			proto_outputs = torch.matmul(indices, pci_pos)
			n_correct_batch = torch.gather(proto_outputs, 1, target.unsqueeze(1)).squeeze()
			n_correct_local += n_correct_batch.cpu().sum().item()
			n_examples_batch = torch.t(pci_pos[:, label]).sum(1)
			n_examples_local += n_examples_batch.sum().cpu().item()

			_, predicted = torch.max(output.data, 1)
			_, predicted_global = torch.max(global_output.data, 1)
			corr_mask = (predicted == target).long()
			corr_mask_global = (predicted_global == target).long()

			n_correct_local_both += (corr_mask * n_correct_batch).sum().item()
			n_correct += (corr_mask * n_examples_batch).sum().item()

			n_correct_local_both_global += (corr_mask_global * n_correct_batch).sum().item()
			n_correct_global += (corr_mask_global * n_examples_batch).sum().item()

			# evaluation statistics
			#_, predicted = torch.max(output.data, 1)
			#_, pred2 = torch.topk(output.data, 2, 1)


		del input
		del target
		del output
	
	end = time.time()
	
	log('\ttime: \t{0}'.format(end -  start))
	if not debug: 
		log('\tlocal prototype accu (overall): \t\t{0}%'.format(n_correct_local / n_examples_local * 100))
	if n_examples_local > n_correct and n_correct > 0.: 
		log('\tlocal prototype accu (network correct): \t\t{0}%'.format(n_correct_local_both / n_correct * 100))
		log('\tlocal prototype accu (network incorrect): \t\t{0}%'.format((n_correct_local - n_correct_local_both) / (n_examples_local - n_correct) * 100))
	elif n_correct == 0.: 
		log('\tnetwork has 0 accu')
	else: 
		log('\tnetwork has 100 percent accu')
	if n_examples_local > n_correct_global and n_correct_global > 0.: 
		log('\tlocal prototype accu (network correct, global only): \t\t{0}%'.format(n_correct_local_both_global / n_correct_global * 100))
		log('\tlocal prototype accu (network incorrect, global only): \t\t{0}%'.format((n_correct_local - n_correct_local_both_global) / (n_examples_local - n_correct_global) * 100))
	elif n_correct_global == 0.: 
		log('\tnetwork has 0 global accu')
	else: 
		log('\tnetwork has 100 percent accu (global only)')
def per_class_test(model, dataloader, class_specific=True, log=print): 
	log('per-class test')
	start = time.time()
	model.eval()
	with torch.no_grad():
		n_examples = torch.zeros(model.module.num_classes).cuda()
		n_correct = torch.zeros(model.module.num_classes).cuda()
		for i, (images, labels) in enumerate(dataloader): 
			images = images.cuda()
			labels = labels.cuda()
			logits, _, _ = model(images)	
			_, predicted = torch.max(logits.data, 1)
			for j in range(labels.size(0)): 
				corr_label = labels.data[j]
				n_examples[corr_label] += 1
				if predicted[j] == labels[j]: 
					n_correct[corr_label] += 1
	acc = 100 * n_correct / n_examples
	log('Mean Acc: {0}'.format(torch.mean(acc)))
	for i in range(acc.size(0)): 
		log('Class ' + str(i) + ' Acc: {0}'.format(acc[i]))
	log('Time: {0}'.format(time.time() - start))
	return torch.mean(acc)

def topk_correct_batch(logits, labels, k=10): 
    _, indices = torch.topk(logits, k, dim=1)
    labels_compare = labels.clone().unsqueeze(1)
    n_correct = torch.eq(indices, labels_compare).sum().item()
    return n_correct