import time
import torch

from settings import dist_start, padding, padding2, distnormalize, all_class_specific, num_classes, dist_classmax
from settings import pos_weight, neg_weight, clamp_prototypes, freeze_zeros, hard_ll, neg_local, dropout_p

from helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, writer=None, iter=0, optimizer=None, class_specific=True, use_l1_mask=True,
				   coefs=None, log=print, epoch=0):
	'''
	model: the multi-gpu model
	dataloader:
	optimizer: if None, will be test evaluation
	'''
	is_train = optimizer is not None
	start = time.time()
	n_examples = 0
	n_correct = 0
	n_batches = 0
	total_cross_entropy = 0
	total_cluster_cost_local = 0
	total_cluster_cost_global = 0
	# separation cost is meaningful only for class_specific
	total_separation_cost_local = 0
	total_separation_cost_global = 0
	total_avg_separation_cost_local = 0
	total_avg_separation_cost_global = 0
	total_l1 = 0

	model = model.cuda()

	#get L1 mask
	'''
	if neg_local: 
		pci_local, pci_global = local_global(model.module.prototype_class_identity_negc.clone(), 0, model.module.num_classes, model.module.local_prototypes_per_class, pci=True)
		l1_local = torch.clamp(pci_local, min=0, max=pos_weight) / pos_weight #only sparsifies positive local connections
		l1_global =  torch.clamp(pci_global, min=neg_weight, max=0) / neg_weight #only sparsifies negative global connections

		l1_mask = torch.t(torch.cat((l1_local, l1_global), dim=1).reshape(model.module.num_prototypes, model.module.num_classes)).cuda()
	else: 
	'''
	zero_mask_t = torch.clamp(model.module.prototype_class_identity_negc.clone(), min=0, max=1)
	zero_mask = torch.t(zero_mask_t).cuda()

	l1_mask = 1 - zero_mask

	for i, (image, label) in enumerate(dataloader):
		input = image.cuda()
		target = label.cuda()

		batch_size = target.size(0)

		if clamp_prototypes: 
			model.module.prototype_vectors.data = torch.clamp(model.module.prototype_vectors, min=0, max=1)

		# torch.enable_grad() has no effect outside of no_grad()
		grad_req = torch.enable_grad() if is_train else torch.no_grad()
		with grad_req:
			# nn.Module has implemented __call__() function
			# so no need to call .forward
			if is_train: 
				output, min_distances = model(input, dropout_p)
			else: 
				output, min_distances = model(input, 0.)

			min_distances_global = min_distances[:, :model.module.num_global_prototypes]
			min_distances_local = min_distances[:, model.module.num_global_prototypes:]

			prototype_class_identity = model.module.prototype_class_identity_negc
			prototype_class_identity_cluster = torch.clamp(prototype_class_identity, 0, 1)
			prototype_class_identity_sep = torch.clamp(prototype_class_identity, -0.5, 0) / -0.5
			# compute loss
			cross_entropy = torch.nn.functional.cross_entropy(output, target)

			if class_specific:
				max_dist = (model.module.prototype_shape[1]
							* model.module.prototype_shape[2]
							* model.module.prototype_shape[3])

				# prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
				# calculate cluster cost
				prototypes_of_correct_class = torch.t(prototype_class_identity_cluster[:,label]).cuda()
				#prototypes_of_correct_class_local, prototypes_of_correct_class_global = local_global(prototypes_of_correct_class, batch_size, model.module.num_classes, model.module.local_prototypes_per_class)
				prototypes_of_correct_class_global = prototypes_of_correct_class[:, :model.module.num_global_prototypes]
				prototypes_of_correct_class_local = prototypes_of_correct_class[:, model.module.num_global_prototypes:]

				clst_distances_local, _ = torch.max((max_dist - min_distances_local) * prototypes_of_correct_class_local, dim=1)
				clst_distances_global, _ = torch.max((max_dist - min_distances_global) * prototypes_of_correct_class_global, dim=1)
				clst_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

				cluster_cost_local = torch.mean(max_dist - clst_distances_local)
				cluster_cost_global = torch.mean(max_dist - clst_distances_global)
				cluster_cost = torch.mean(max_dist - clst_distances)
				# calculate separation cost
				prototypes_of_wrong_class = torch.t(prototype_class_identity_sep[:, label]).cuda()
				#prototypes_of_wrong_class_local, prototypes_of_wrong_class_global = local_global(prototypes_of_wrong_class, batch_size, model.module.num_classes, model.module.local_prototypes_per_class)
				prototypes_of_wrong_class_global = prototypes_of_wrong_class[:, :model.module.num_global_prototypes]
				prototypes_of_wrong_class_local = prototypes_of_wrong_class[:, model.module.num_global_prototypes:]

				sep_distances_local, _ = \
					torch.max((max_dist - min_distances_local) * prototypes_of_wrong_class_local, dim=1)
				sep_distances_global, _ = \
					torch.max((max_dist - min_distances_global) * prototypes_of_wrong_class_global, dim=1)
				sep_distances, _ = \
					torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
				
				separation_cost_local = torch.mean(max_dist - sep_distances_local)
				separation_cost_global = torch.mean(max_dist - sep_distances_global)
				separation_cost = torch.mean(max_dist - sep_distances)
				
				avg_separation_cost_local = \
					torch.sum(min_distances_local * prototypes_of_wrong_class_local, dim=1) / torch.sum(prototypes_of_wrong_class_local, dim=1)
				avg_separation_cost_local = torch.mean(avg_separation_cost_local)

				avg_separation_cost_global = \
					torch.sum(min_distances_global * prototypes_of_wrong_class_global, dim=1) / torch.sum(prototypes_of_wrong_class_global, dim=1)
				avg_separation_cost_global = torch.mean(avg_separation_cost_global)
				
				if use_l1_mask:
					l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
					#l1 should be 0 when hard_ll is true
				else:
					l1 = model.module.last_layer.weight.norm(p=1) 

			else:
				min_distance, _ = torch.min(min_distances, dim=1)
				cluster_cost = torch.mean(min_distance)
				l1 = model.module.last_layer.weight.norm(p=1)

			# evaluation statistics
			_, predicted = torch.max(output.data, 1)
			n_examples += target.size(0)
			n_correct += (predicted == target).sum().item()

			n_batches += 1
			total_cross_entropy += cross_entropy.item()
			total_cluster_cost_local += cluster_cost_local.item()
			total_cluster_cost_global += cluster_cost_global.item()
			total_separation_cost_local += separation_cost_local.item()
			total_separation_cost_global += separation_cost_global.item()
			total_avg_separation_cost_local += avg_separation_cost_local.item()
			total_avg_separation_cost_global += avg_separation_cost_global.item()
			total_l1 += l1.item()

			if is_train and writer is not None:
				writer.add_scalar('Train Batch Accuracy', 100 * (predicted == target).sum().item() / target.size(0), iter)
				writer.add_scalar('Train CrsEnt', cross_entropy.item(), iter)
				writer.add_scalar('Train Clst Local', cluster_cost_local.item(), iter)
				writer.add_scalar('Train Clst Global', cluster_cost_global.item(), iter)
				writer.add_scalar('Train Sep Local', separation_cost_local.item(), iter)
				writer.add_scalar('Train Sep Global', separation_cost_global.item(), iter)
				writer.add_scalar('Train Avg Sep Local', avg_separation_cost_local.item(), iter)
				writer.add_scalar('Train Avg Sep Global', avg_separation_cost_global.item(), iter)
				iter += 1

		# compute gradient and do SGD step
		if is_train:
			distc = 0 if epoch < dist_start else coefs['dist']
			if class_specific:
				if coefs is not None:
					loss = (coefs['crs_ent'] * cross_entropy
						  + coefs['clst'] * cluster_cost_global
						  + coefs['clst_l'] * cluster_cost_local
						  + coefs['sep'] * separation_cost_global
						  + coefs['sep_l'] * separation_cost_local
						  + coefs['l1'] * l1)
				else:
					loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
			else:
				if coefs is not None:
					loss = (coefs['crs_ent'] * cross_entropy
						  + coefs['clst'] * cluster_cost
						  + coefs['l1'] * l1)
				else:
					loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
			optimizer.zero_grad()
			loss.backward()
			'''
			if freeze_zeros: 
				if hard_ll: 
					model.module.last_layer.weight.grad = model.module.last_layer.weight.grad * (zero_mask)
				else: 
					foo = (torch.abs(model.module.prototype_class_identity_negc.clone()) / min(-1 * neg_weight, pos_weight)).cuda()
					zero_mask = torch.clamp(foo, min=0, max=1)
					model.module.last_layer.weight.grad = model.module.last_layer.weight.grad * torch.t(zero_mask)
			'''
			optimizer.step()

		del input
		del target
		del output
		del predicted
		del min_distances

	end = time.time()

	log('\ttime: \t{0}'.format(end -  start))
	log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
	log('\tlocal cluster: \t{0}'.format(total_cluster_cost_local / n_batches))
	log('\tglobal cluster: \t{0}'.format(total_cluster_cost_global / n_batches))
	if class_specific:
		log('\tlocal separation:\t{0}'.format(total_separation_cost_local / n_batches))
		log('\tglobal separation:\t{0}'.format(total_separation_cost_global / n_batches))
		log('\tlocal avg separation:\t{0}'.format(total_avg_separation_cost_local / n_batches))
		log('\tglobal avg separation:\t{0}'.format(total_avg_separation_cost_global / n_batches))
	log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
	log('\tl1: \t\t{0}'.format(total_l1 / n_batches))
	p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
	with torch.no_grad():
		p_avg_pair_dist = torch.mean(list_of_distances(p, p))
	log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

	traintest = 'training ' if is_train else 'testing '
	
	if not is_train and writer is not None: 
		writer.add_scalar('Test Accuracy', n_correct / n_examples * 100, iter)
		writer.add_scalar('Test CrsEnt', total_cross_entropy / n_batches, iter)
		writer.add_scalar('Test Clst Local', total_cluster_cost_local / n_batches, iter)
		writer.add_scalar('Test Clst Global', total_cluster_cost_global / n_batches, iter)
		writer.add_scalar('Test Sep Local', total_separation_cost_local / n_batches, iter)
		writer.add_scalar('Test Sep Global', total_separation_cost_global / n_batches, iter)
		writer.add_scalar('Test Avg Sep Local', avg_separation_cost_local.item(), iter)
		writer.add_scalar('Test Avg Sep Global', avg_separation_cost_global.item(), iter)
	
	return n_correct / n_examples, iter


def train(model, dataloader, writer=None, iter=0, optimizer=None, class_specific=False, coefs=None, log=print, epoch=0):
	assert(optimizer is not None)
	
	log('\ttrain')
	model.train()
	return _train_or_test(model=model, dataloader=dataloader, writer = writer, iter = iter, optimizer=optimizer,
						  class_specific=class_specific, coefs=coefs, log=log, epoch=epoch)


def test(model, dataloader, writer=None, iter=0, class_specific=False, log=print):
	log('\ttest')
	model.eval()
	return _train_or_test(model=model, dataloader=dataloader, writer = writer, iter = iter, optimizer=None,
						  class_specific=class_specific, log=log)


def last_only(model, log=print):
	if hard_ll:
		if neg_local:
			pci_local, pci_global = local_global(model.module.prototype_class_identity_negc.clone(), 0, model.module.num_classes, model.module.local_prototypes_per_class, pci=True)
			ll_local = torch.clamp(pci_local, min=neg_weight, max=0) #only keep positive local connections
			ll_global =  torch.clamp(pci_global, min=0, max=pos_weight) #only keep negative global connections
			ll_mask = torch.t(torch.cat((ll_local, ll_global), dim=1).reshape(model.module.num_prototypes, model.module.num_classes)).cuda()
			model.module.last_layer.weight.data = ll_mask
		else: 
			ll_mask_t = torch.clamp(model.module.prototype_class_identity_negc.clone(), min=0, max=pos_weight)
			ll_mask = torch.t(ll_mask_t)
			model.module.last_layer.weight.data = ll_mask
	for p in model.module.features.parameters():
		p.requires_grad = False
	for p in model.module.add_on_layers.parameters():
		p.requires_grad = False
	model.module.prototype_vectors.requires_grad = False
	for p in model.module.last_layer.parameters():
		p.requires_grad = True
	
	log('\tlast layer')


def warm_only(model, log=print):
	for p in model.module.features.parameters():
		p.requires_grad = False
	for p in model.module.add_on_layers.parameters():
		p.requires_grad = True
	model.module.prototype_vectors.requires_grad = True
	for p in model.module.last_layer.parameters():
		p.requires_grad = True
	
	log('\twarm')


def joint(model, log=print):
	for p in model.module.features.parameters():
		p.requires_grad = True
	for p in model.module.add_on_layers.parameters():
		p.requires_grad = True
	model.module.prototype_vectors.requires_grad = True
	for p in model.module.last_layer.parameters():
		p.requires_grad = True
	
	log('\tjoint')

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
			logits, _ = model(images)   
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

def separation_cost_ignore(prototypes_of_correct_class, max_dist, min_distances, prototypes_per_class, sep_prototypes_per_class): 
	batch_size = prototypes_of_correct_class.size(0)

	prototypes_of_wrong_class = 1 - prototypes_of_correct_class
	prototypes_of_wrong_class = prototypes_of_wrong_class.view(batch_size, -1, prototypes_per_class)
	sep_mask = torch.tensor([1 if i < sep_prototypes_per_class else 0 for i in range(prototypes_per_class)]).unsqueeze(0).unsqueeze(0).cuda()
	sep_mask = sep_mask.expand_as(prototypes_of_wrong_class)
	prototypes_of_wrong_class = prototypes_of_wrong_class * sep_mask
	prototypes_of_wrong_class = prototypes_of_wrong_class.view(min_distances.size())
	inverted_distances_to_nontarget_prototypes, _ = \
		torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
	separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
	return separation_cost

def distance_loss(attention_maps, loc_x, loc_y):
	#attention_maps: batch_size * num_prototypes * w * h
	#loc_x, loc_y: batch_size * num_prototypes
	batch_size = attention_maps.size(0)
	num_prototypes = attention_maps.size(1)
	w = attention_maps.size(2)
	h = attention_maps.size(3)

	x = torch.linspace(0, w - 1, steps=w).cuda()	
	y = torch.linspace(0, h - 1, steps=h).cuda()
	grid_x, grid_y = torch.meshgrid(x, y)

	exp_loc_x = loc_x.unsqueeze(2).unsqueeze(3).expand(batch_size, num_prototypes, w, h)
	exp_loc_y = loc_y.unsqueeze(2).unsqueeze(3).expand(batch_size, num_prototypes, w, h)
	exp_grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, num_prototypes, w, h)
	exp_grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, num_prototypes, w, h)

	dist_maps = (exp_loc_x - exp_grid_x)**2 + (exp_loc_y - exp_grid_y)**2
	loss_maps = dist_maps * attention_maps

	return torch.mean(loss_maps)

def distance_loss_classspecific(attention_maps, prototype_class_identity, labels, loc_x, loc_y, padding = 1, padding2 = 14, normalize = False, dist_classmax = False):
	#attention_maps: batch_size * num_prototypes * w * h
	#loc_x, loc_y: batch_size * num_prototypes
	batch_size = attention_maps.size(0)
	num_prototypes = attention_maps.size(1)
	w = attention_maps.size(2)
	h = attention_maps.size(3)

	prototypes_of_correct_class = torch.t(prototype_class_identity[:,labels]).unsqueeze(2).unsqueeze(3).cuda()
	prototypes_of_correct_class = prototypes_of_correct_class.expand_as(attention_maps)

	class_maps = attention_maps.clone() 
	if normalize:
		if dist_classmax: 
			prototypes_per_class = num_prototypes / num_classes
			class_maps = class_maps.view(batch_size, num_classes, -1)
			max_vals, _ = torch.max(class_maps, 2, keepdim = True)
			class_maps = class_maps / max_vals
		else: 
			class_maps = class_maps.view(batch_size, num_prototypes, -1)
			max_vals, _ = torch.max(class_maps, 2, keepdim = True)
			class_maps = class_maps / max_vals
	class_maps = class_maps.view(batch_size, num_prototypes, w, h) * prototypes_of_correct_class

	x = torch.linspace(0, w - 1, steps=w).cuda()	
	y = torch.linspace(0, h - 1, steps=h).cuda()
	grid_x, grid_y = torch.meshgrid(x, y)

	exp_loc_x = loc_x.unsqueeze(2).unsqueeze(3).expand(batch_size, num_prototypes, w, h)
	exp_loc_y = loc_y.unsqueeze(2).unsqueeze(3).expand(batch_size, num_prototypes, w, h)
	exp_grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, num_prototypes, w, h)
	exp_grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, num_prototypes, w, h)

	dist_maps = torch.clamp(torch.abs(exp_loc_x - exp_grid_x) - padding, min=0)**2 + torch.clamp(torch.abs(exp_loc_y - exp_grid_y) - padding, min=0)**2
	#dist_maps_l4 = torch.clamp(torch.abs(exp_loc_x - exp_grid_x) - padding2, min=0)**4 + torch.clamp(torch.abs(exp_loc_y - exp_grid_y) - padding2, min=0)**4

	loss_maps = dist_maps * class_maps #+ dist_maps_l4 * class_maps

	return torch.mean(loss_maps)

def diversity_loss(attention_maps, num_classes, margin = 0.02): 
	batch_size = attention_maps.size(0)
	num_prototypes = attention_maps.size(1)
	w = attention_maps.size(2)
	h = attention_maps.size(3)

	prototypes_per_class = int(num_prototypes / num_classes)

	attention_maps = attention_maps.view(batch_size, num_classes, prototypes_per_class, w, h)

	divloss = 0

	for i in range(prototypes_per_class): 
		base_maps = attention_maps[:, :, i, :, :].squeeze()

		sel_mask = [j!=i for j in range(prototypes_per_class)]
		other_maps = attention_maps[:, :, sel_mask, :, :]
		max_other, _ = torch.max(other_maps, 2)

		divloss += torch.mean(base_maps * (max_other - margin))

	return divloss

def diversity_loss_argmax(attention_maps, num_classes, margin = 0.02): 
	batch_size = attention_maps.size(0)
	num_prototypes = attention_maps.size(1)
	w = attention_maps.size(2)
	h = attention_maps.size(3)

	prototypes_per_class = int(num_prototypes / num_classes)

	attention_maps = attention_maps.view(batch_size, num_classes, prototypes_per_class, w, h)

	divloss = 0

	for i in range(prototypes_per_class): 
		base_maps = attention_maps[:, :, i, :, :].squeeze()

		sel_mask = [j!=i for j in range(prototypes_per_class)]
		other_maps = attention_maps[:, :, sel_mask, :, :]
		max_other, _ = torch.max(other_maps, 2)

		base_maps_flat = base_maps.view(batch_size, num_classes, -1)
		max_other_flat = max_other.view(batch_size, num_classes, -1)

		values, indices = torch.max(base_maps_flat, 2, keepdim = True)

		base_maps_vals = values.squeeze()
		max_other_vals = torch.gather(max_other_flat, 2, indices).squeeze()

		divloss += torch.mean(base_maps_vals.detach() * (max_other_vals - margin))

	return divloss

def diversity_loss_argmax_classspecific(attention_maps, labels, num_classes, margin = 0.02):
	batch_size = attention_maps.size(0)
	num_prototypes = attention_maps.size(1)
	w = attention_maps.size(2)
	h = attention_maps.size(3)

	prototypes_per_class = int(num_prototypes / num_classes)

	attention_maps = attention_maps.view(batch_size, num_classes, prototypes_per_class, w, h)
	target = labels.clone().unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
	target = target.expand(batch_size, 1, prototypes_per_class, w, h)
	class_maps = torch.gather(attention_maps, 1, target)

	divloss = 0

	for i in range(prototypes_per_class):
		base_maps = class_maps[:, :, i, :, :].squeeze()
		sel_mask = [j!=i for j in range(prototypes_per_class)]
		other_maps = attention_maps[:, :, sel_mask, :, :]
		max_other, _ = torch.max(other_maps, 2)
		base_maps_flat = base_maps.view(batch_size, 1, -1)
		max_other_flat = max_other.view(batch_size, 1, -1)
		values, indices = torch.max(base_maps_flat, 2, keepdim = True)
		base_maps_vals = values.squeeze()
		max_other_vals = torch.gather(max_other_flat, 2, indices).squeeze()
		divloss += torch.mean(base_maps_vals.detach() * (max_other_vals - margin))

	return divloss

def diversity_loss_noscaling(attention_maps, num_classes, margin = 0.02): 
	batch_size = attention_maps.size(0)
	num_prototypes = attention_maps.size(1)
	w = attention_maps.size(2)
	h = attention_maps.size(3)

	prototypes_per_class = int(num_prototypes / num_classes)

	attention_maps = attention_maps.view(batch_size, num_classes, prototypes_per_class, w, h)

	divloss = 0

	for i in range(prototypes_per_class): 
		base_maps = attention_maps[:, :, i, :, :].squeeze()

		sel_mask = [j!=i for j in range(prototypes_per_class)]
		other_maps = attention_maps[:, :, sel_mask, :, :]
		max_other, _ = torch.max(other_maps, 2)

		base_maps_flat = base_maps.view(batch_size, num_classes, -1)
		max_other_flat = max_other.view(batch_size, num_classes, -1)

		_, indices = torch.max(base_maps_flat, 2, keepdim = True)

		max_other_vals = torch.gather(max_other_flat, 2, indices).squeeze()

		divloss += torch.mean(max_other_vals - margin)

	return divloss

def local_global(x, batch_size, num_classes, local_prototypes_per_class, reshape=True, pci=False):
	if pci: 
		out_local = x.view(num_classes, -1, num_classes)[:, :local_prototypes_per_class, :]
		out_global = x.view(num_classes, -1, num_classes)[:, local_prototypes_per_class:, :]
	else:	
		if reshape: 
			out_local = x.view(batch_size, num_classes, -1)[:, :, :local_prototypes_per_class].reshape(batch_size, -1)
			out_global = x.view(batch_size, num_classes, -1)[:, :, local_prototypes_per_class:].reshape(batch_size, -1)
		else: 
			out_local = x.view(batch_size, num_classes, -1)[:, :, :local_prototypes_per_class]
			out_global = x.view(batch_size, num_classes, -1)[:, :, local_prototypes_per_class:]	

	return out_local, out_global 
