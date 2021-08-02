import time
import torch
import sys

from settings import dist_start, padding, padding2, distnormalize, all_class_specific, num_classes, dist_classmax
from settings import pos_weight, neg_weight, clamp_prototypes, freeze_zeros, hard_ll, neg_local, dropout_p
from settings import num_prototypes_to_add
from settings import sep_margin, proto_k, by_class, sep_crsent, sep_crsent_k
from helpers import list_of_distances, make_one_hot, sizeof_fmt

import torch.autograd.profiler as profiler

def _train_or_test(model, dataloader, writer=None, iter=0, optimizer=None, new_protos=False,class_specific=True, use_l1_mask=True,
					coefs=None, log=print, epoch=0, debug=False):
	'''
	model: the multi-gpu model
	dataloader:
	optimizer: if None, will be test evaluation
	'''
	is_train = optimizer is not None
	start = time.time()
	n_examples = 0
	n_correct = 0
	n_correct_global = 0
	n_correct_2 = 0
	n_correct_5 = 0
	n_batches = 0
	n_examples_localclst = 0
	n_examples_localsep = 0
	total_cross_entropy = 0
	total_cluster_cost_local = 0
	total_cluster_cost_global = 0
	# separation cost is meaningful only for class_specific
	total_separation_cost_local = 0
	total_separation_cost_global = 0
	total_avg_separation_cost_local = 0
	total_avg_separation_cost_global = 0
	total_dist = 0
	total_l1 = 0

	total_loss = 0

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
	zero_mask_t = model.module.prototype_class_identity.clone()
	zero_mask = torch.t(zero_mask_t).cuda()

	l1_mask = 1 - zero_mask
	if is_train:
		sep_cost_matrix = torch.zeros((model.module.num_classes, model.module.num_classes)).cuda()
	 
	for i, (image, label) in enumerate(dataloader):
		#with profiler.profile(profile_memory=True, record_shapes=True) as prof:
		#b1 = time.time()
		if debug: 
			print(i)
			if i > 0: 
				break
		input = image.cuda()
		target = label.cuda()

		batch_size = target.size(0)

		if clamp_prototypes: 
			model.module.prototype_vectors.data = torch.clamp(model.module.prototype_vectors.data, min=0, max=1)
		# torch.enable_grad() has no effect outside of no_grad()
		grad_req = torch.enable_grad() if is_train else torch.no_grad()
		with grad_req:
			# nn.Module has implemented __call__() function
			# so no need to call .forward
			#if is_train: 
			#	output, min_distances = model(input, dropout_p)
			#else: 
			output, global_output, min_distances, dist_vec = model(input, k=proto_k, labels=target, dist_loss = True, by_class=by_class)
			dist_loss = torch.mean(dist_vec)

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
				prototypes_of_correct_class_global = prototypes_of_correct_class[:, :model.module.num_global_prototypes]
				prototypes_of_correct_class_local = prototypes_of_correct_class[:, model.module.num_global_prototypes:]

				if model.module.num_local_prototypes > 0: 
					pcc_indicator = torch.sum(prototypes_of_correct_class_local, dim=1)
					if torch.sum(pcc_indicator) == 0.: 
						cluster_cost_local = torch.tensor(0.)
					else: 
						inv_clst_distances_local, _ = torch.max((max_dist - min_distances_local) * prototypes_of_correct_class_local, dim=1)
						#only want to consider cases where >=1 local prototypes have positive connections to image in question
						cluster_cost_local = torch.sum(max_dist - inv_clst_distances_local[pcc_indicator > 0.]) / batch_size
						n_examples_localclst += torch.sum((pcc_indicator>0.)).item()
				else: 
					cluster_cost_local = torch.tensor(0.)
				clst_distances_global, _ = torch.max((max_dist - min_distances_global) * prototypes_of_correct_class_global, dim=1)
				#clst_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

				cluster_cost_global = torch.mean(max_dist - clst_distances_global)
				#cluster_cost = torch.mean(max_dist - clst_distances)
				# calculate separation cost
				prototypes_of_wrong_class = torch.t(prototype_class_identity_sep[:, label]).cuda()
				prototypes_of_wrong_class_global = prototypes_of_wrong_class[:, :model.module.num_global_prototypes]
				prototypes_of_wrong_class_local = prototypes_of_wrong_class[:, model.module.num_global_prototypes:]

				if model.module.num_local_prototypes > 0: 
					pwc_indicator = torch.sum(prototypes_of_wrong_class_local, dim=1)
					if torch.sum(pwc_indicator) == 0.: 
						separation_cost_local = torch.tensor(0.)
						separation_cost_local_clamped = torch.tensor(0.)
					else: 
						sep_distances_local, _ = \
							torch.max((max_dist - min_distances_local) * prototypes_of_wrong_class_local, dim=1)
						#only want to consider cases where >=1 local prototypes have negative connections to image in question
						separation_cost_local = torch.sum(max_dist - sep_distances_local[pwc_indicator > 0.]) / batch_size
						n_examples_localsep += torch.sum((pwc_indicator>0.)).item()
						if sep_margin != 0.: 
							separation_cost_local_clamped = torch.sum(torch.clamp(max_dist - sep_distances_local[pwc_indicator > 0.], min=0., max = sep_margin)) / torch.sum((pwc_indicator>0.)).item()
				else:
					separation_cost_local = torch.tensor(0.)
					separation_cost_local_clamped = torch.tensor(0.)
				sep_distances_global, sep_indices_global = \
					torch.max((max_dist - min_distances_global) * prototypes_of_wrong_class_global, dim=1)
				sep_distances, _ = \
					torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
				separation_cost_global = torch.mean(max_dist - sep_distances_global)
				if sep_margin != 0.: 
					separation_cost_global_clamped = torch.mean(torch.clamp(max_dist - sep_distances_global, min=0., max=sep_margin))
					#separation_cost = torch.mean(max_dist - sep_distances)

				#b2 = time.time()
				#print('time before sep agg: ', b2 - b1)
				if is_train:
					with torch.no_grad():
						#aggregate entries in separation cost matrix
						sep_cost_activation = model.module.distance_2_similarity(max_dist - sep_distances_global)
						class_sep_indices = model.module.pci_list[sep_indices_global]
						batch_sep = torch.zeros((model.module.num_classes, model.module.num_classes)).cuda()

						batch_sep[class_sep_indices, target] = sep_cost_activation
						sep_cost_matrix += batch_sep * 1e-4

				#b3 = time.time()
				#print('time after sep agg: ', b3 - b2)
				
				if model.module.num_local_prototypes > 0: 
					avg_separation_cost_local = \
						torch.sum(min_distances_local * prototypes_of_wrong_class_local, dim=1) / torch.sum(prototypes_of_wrong_class_local, dim=1)
					if (~torch.isnan(avg_separation_cost_local)).sum() == 0: 
						avg_separation_cost_local = torch.tensor(0.)
					else: 
						avg_separation_cost_local = avg_separation_cost_local[~torch.isnan(avg_separation_cost_local)].mean()
				else: 
					avg_separation_cost_local = torch.tensor(0.)

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
			_, predicted_global = torch.max(global_output.data, 1)
			_, predicted = torch.max(output.data, 1)
			n_examples += target.size(0)
			n_correct += (predicted == target).sum().item()
			n_correct_global += (predicted_global == target).sum().item()
			n_correct_2 += topk_correct_batch(output, target, 2)
			n_correct_5 += topk_correct_batch(output, target, 5)

			if debug: 
				print('cluster cost local', cluster_cost_local.item())
				print('separation cost global', separation_cost_local.item())

			n_batches += 1
			total_cross_entropy += cross_entropy.item()
			total_cluster_cost_local += cluster_cost_local.item() * batch_size
			total_cluster_cost_global += cluster_cost_global.item()
			total_separation_cost_local += separation_cost_local.item() * batch_size
			total_separation_cost_global += separation_cost_global.item()
			total_avg_separation_cost_local += avg_separation_cost_local.item()
			total_avg_separation_cost_global += avg_separation_cost_global.item()
			total_dist += dist_loss.item()
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
			if class_specific:
				if coefs is not None:
					if sep_margin != 0.: 
						loss = (coefs['crs_ent'] * cross_entropy
							  + coefs['clst'] * cluster_cost_global
							  + coefs['clst_l'] * cluster_cost_local
							  + coefs['sep'] * separation_cost_global_clamped
							  + coefs['sep_l'] * separation_cost_local_clamped
							  + coefs['dist'] * dist_loss
							  + coefs['l1'] * l1)
					else: 
						loss = (coefs['crs_ent'] * cross_entropy
							  + coefs['clst'] * cluster_cost_global
							  + coefs['clst_l'] * cluster_cost_local
							  + coefs['sep'] * separation_cost_global
							  + coefs['sep_l'] * separation_cost_local
							  + coefs['dist'] * dist_loss
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
			#b4 = time.time()
			#print('Time calculating losses, etc.', b4 - b3)
			with torch.no_grad(): 
				total_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			
			if freeze_zeros: 
				if hard_ll: 
					model.module.last_layer.weight.grad = model.module.last_layer.weight.grad * (zero_mask)
				'''
				else: 
					#not sure if this actually works, so far only using hard ll
					foo = (torch.abs(model.module.prototype_class_identity_negc.clone()) / min(-1 * neg_weight, pos_weight)).cuda()
					zero_mask = torch.clamp(foo, min=0, max=1)
					model.module.last_layer.weight.grad = model.module.last_layer.weight.grad * torch.t(zero_mask)
				'''
			
			if new_protos: 
				#freeze the prototypes that aren't the new ones
				start_ind = model.module.num_global_prototypes + model.module.num_local_prototypes - num_prototypes_to_add
				end_ind = model.module.num_global_prototypes + model.module.num_local_prototypes
				model.module.prototype_vectors.grad[:start_ind] = 0.
				model.module.prototype_vectors.grad[end_ind:] = 0.
			optimizer.step()
			#b5 = time.time()
			#print('Time for backprop: ', b5 - b4)
		del input
		del target
		del output
		del predicted
		del min_distances
	
	end = time.time()
	
	log('\ttime: \t{0}'.format(end -  start))
	log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
	if n_examples_localclst != 0.: 
		log('\tlocal cluster: \t{0}'.format(total_cluster_cost_local / n_examples_localclst))
	else: 
		log('\t no local cluster examples!')
	log('\tglobal cluster: \t{0}'.format(total_cluster_cost_global / n_batches))
	if class_specific:
		if n_examples_localsep != 0.:
			log('\tlocal separation:\t{0}'.format(total_separation_cost_local / n_examples_localsep))
		else: 
			log('\t no local separation examples!')
		log('\tglobal separation:\t{0}'.format(total_separation_cost_global / n_batches))
		log('\tlocal avg separation:\t{0}'.format(total_avg_separation_cost_local / n_batches))
		log('\tglobal avg separation:\t{0}'.format(total_avg_separation_cost_global / n_batches))
	log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
	log('\taccu (global): \t\t{0}%'.format(n_correct_global / n_examples * 100))
	log('\taccu (top2, global): \t\t{0}%'.format(n_correct_2 / n_examples * 100))
	log('\taccu (top5, global): \t\t{0}%'.format(n_correct_5 / n_examples * 100))
	log('\tdist: \t\t{0}'.format(total_dist / n_batches))
	log('\tl1: \t\t{0}'.format(total_l1 / n_batches))
	log('\tloss: \t\t{0}'.format(total_loss / n_batches))
	#p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
	#with torch.no_grad():
	#	p_avg_pair_dist = torch.mean(list_of_distances(p, p))
	#log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
	
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
	
	#update prototypes
	if is_train:
		return n_correct / n_examples, iter, sep_cost_matrix
	else:
		return n_correct / n_examples, iter


def train(model, dataloader, writer=None, iter=0, optimizer=None, new_protos=False,class_specific=False, coefs=None, log=print, epoch=0, debug=False):
	assert(optimizer is not None)
	
	log('\ttrain')
	model.train()
	return _train_or_test(model=model, dataloader=dataloader, writer = writer, iter = iter, optimizer=optimizer, new_protos=new_protos,
						  class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, debug=debug)


def test(model, dataloader, writer=None, iter=0, class_specific=False, log=print, debug=False):
	log('\ttest')
	model.eval()
	return _train_or_test(model=model, dataloader=dataloader, writer = writer, iter = iter, optimizer=None,
						  class_specific=class_specific, log=log, debug=debug)


def last_only(model, log=print):
	if hard_ll:
		ll_mask_t = model.module.prototype_class_identity.clone().cuda()
		ll_mask = torch.t(ll_mask_t)
		model.module.last_layer.weight.data = model.module.last_layer.weight.data*ll_mask
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

def topk_correct_batch(logits, labels, k=10): 
    _, indices = torch.topk(logits, k, dim=1)
    labels_compare = labels.clone().unsqueeze(1)
    n_correct = torch.eq(indices, labels_compare).sum().item()
    return n_correct