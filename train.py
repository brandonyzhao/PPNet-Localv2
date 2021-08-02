import torch
import torchvision.utils
import torchvision.transforms as transforms
import torch.nn.functional as F
import train_and_test as tnt

from settings import num_classes
from helpers import calculate_sparsity

from collections import OrderedDict
#from settings import unmean, unstd, top_proto, num_classes
#from sklearn.metrics import confusion_matrix

import time


def pp2_step(pp2, optpp2, pp2_in, labels, coefs = None): 
	start = time.time()
	log2, min_distances, _ = pp2(pp2_in)
	cross_entropy = F.cross_entropy(log2, labels)
	
	max_dist = (pp2.module.prototype_shape[1] * pp2.module.prototype_shape[2] * pp2.module.prototype_shape[3])

	# prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
	# calculate cluster cost
	prototypes_of_correct_class = torch.t(pp2.module.prototype_class_identity[:,labels]).cuda()
	inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
	cluster_cost = torch.mean(max_dist - inverted_distances)

	# calculate separation cost
	prototypes_of_wrong_class = 1 - prototypes_of_correct_class
	inverted_distances_to_nontarget_prototypes, _ = \
		torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
	separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

	# calculate avg separation cost
	avg_separation_cost = \
		torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
	avg_separation_cost = torch.mean(avg_separation_cost)

	l1_mask = 1 - torch.t(pp2.module.prototype_class_identity).cuda()
	l1 = (pp2.module.last_layer.weight * l1_mask).norm(p=1)

	if coefs is not None: 
		loss = (coefs['crs_ent'] * cross_entropy
			  + coefs['clst'] * cluster_cost
			  + coefs['sep'] * separation_cost
			  + coefs['l1'] * l1)
	else: 
		loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
	optpp2.zero_grad()
	loss.backward()
	optpp2.step()

	_, predicted2 = torch.max(log2, 1)
	correct2 = (predicted2 == labels).sum()
	print('batch accuracy: {0}'.format(100.0 * correct2 / labels.size(0)), flush = True)

	crsents = []
	for i in range(labels.size(0)): 
		crsent = F.cross_entropy(torch.unsqueeze(log2[i], 0), torch.unsqueeze(labels[i], 0))
		crsents.append(crsent)
	crsents = torch.tensor(crsents)
	maxcrsent = torch.max(crsents)
	print('max cross entropy for batch: {0}'.format(maxcrsent), flush = True)

	bigents = 0
	for i in range(crsents.size(0)): 
		if crsents[i] >= 1: 
			bigents += 1
	print('number of images with cross entropy >= 1: {0}'.format(bigents), flush = True)
	
	end = time.time()
	
	#return only the python number of the softloss so graph is deleted
	return cross_entropy.item(), cluster_cost.item(), separation_cost.item(), avg_separation_cost.item(), maxcrsent.item(), bigents

def pp2_step2(pp2, optpp2, pp2_in, labels, coefs = None): 
	start = time.time()
	log2, min_distances, _, _ = pp2(pp2_in)
	cross_entropy_tensor = F.cross_entropy(log2, labels, reduction = 'none')
	cross_entropy = torch.mean(cross_entropy_tensor)
	
	max_dist = (pp2.module.prototype_shape[1] * pp2.module.prototype_shape[2] * pp2.module.prototype_shape[3])

	# prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
	# calculate cluster cost
	prototypes_of_correct_class = torch.t(pp2.module.prototype_class_identity[:,labels]).cuda()
	inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
	cluster_cost = torch.mean(max_dist - inverted_distances)

	# calculate separation cost
	prototypes_of_wrong_class = 1 - prototypes_of_correct_class
	inverted_distances_to_nontarget_prototypes, _ = \
		torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
	separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

	# calculate avg separation cost
	avg_separation_cost = \
		torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
	avg_separation_cost = torch.mean(avg_separation_cost)

	l1_mask = 1 - torch.t(pp2.module.prototype_class_identity).cuda()
	l1 = (pp2.module.last_layer.weight * l1_mask).norm(p=1)

	if coefs is not None: 
		loss = (coefs['crs_ent'] * cross_entropy
			  + coefs['clst'] * cluster_cost
			  + coefs['sep'] * separation_cost
			  + coefs['l1'] * l1)
	else: 
		loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
	optpp2.zero_grad()
	loss.backward()
	optpp2.step()
	
	_, predicted2 = torch.max(log2, 1)
	correct2 = (predicted2 == labels).sum()
	batch_acc = 100.0 * correct2 / labels.size(0)
	
	end = time.time()
	
	#return only the python number of the softloss so graph is deleted
	return cross_entropy.item(), cross_entropy_tensor.detach(), cluster_cost.item(), separation_cost.item(), avg_separation_cost.item(), batch_acc

def pp2_steprank(pp2, optpp2, pp2_in, labels, coefs = None): 
	start = time.time()
	log2, min_distances, _, _ = pp2(pp2_in)
	cross_entropy_tensor = F.cross_entropy(log2, labels, reduction = 'none')
	cross_entropy = torch.mean(cross_entropy_tensor)
	
	max_dist = (pp2.module.prototype_shape[1] * pp2.module.prototype_shape[2] * pp2.module.prototype_shape[3])

	# prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
	# calculate cluster cost
	prototypes_of_correct_class = torch.t(pp2.module.prototype_class_identity[:,labels]).cuda()
	inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
	cluster_cost = torch.mean(max_dist - inverted_distances)

	# calculate separation cost
	prototypes_of_wrong_class = 1 - prototypes_of_correct_class
	inverted_distances_to_nontarget_prototypes, _ = \
		torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
	separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

	# calculate avg separation cost
	avg_separation_cost = \
		torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
	avg_separation_cost = torch.mean(avg_separation_cost)

	l1_mask = 1 - torch.t(pp2.module.prototype_class_identity).cuda()
	l1 = (pp2.module.last_layer.weight * l1_mask).norm(p=1)

	if coefs is not None: 
		loss = (coefs['crs_ent'] * cross_entropy
			  + coefs['clst'] * cluster_cost
			  + coefs['sep'] * separation_cost
			  + coefs['l1'] * l1)
	else: 
		loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
	optpp2.zero_grad()
	loss.backward()
	optpp2.step()
	
	_, predicted2 = torch.max(log2, 1)
	correct2 = (predicted2 == labels).sum()
	batch_acc = 100.0 * correct2 / labels.size(0)
	
	end = time.time()
	
	#return only the python number of the softloss so graph is deleted
	return cross_entropy.item(), cross_entropy_tensor.detach(), log2.detach(), cluster_cost.item(), separation_cost.item(), avg_separation_cost.item(), batch_acc

def Lrank(log1, log2, labels, num_samples, margin):
	log1 = torch.repeat_interleave(log1, num_samples, dim=0)
	probs1 = F.softmax(log1, dim=1)
	probs2 = F.softmax(log2, dim=1)
	labels = torch.repeat_interleave(labels, num_samples, dim=0)
	probsdiff = torch.clamp(probs1 - probs2 + margin, min=0.0)
	return torch.squeeze(torch.gather(probsdiff, 1, labels.unsqueeze(1)))

def Lrank2(log1, log2, labels, num_samples):
	log1 = torch.repeat_interleave(log1, num_samples, dim=0)
	probs1 = F.softmax(log1, dim=1)
	probs2 = F.softmax(log2, dim=1)
	labels = torch.repeat_interleave(labels, num_samples, dim=0)
	probsdiff = probs1 - probs2
	return torch.squeeze(torch.gather(probsdiff, 1, labels.unsqueeze(1)))

def topkacc(pp1, loader, k=1, train=True): 
	tt = 'Training top ' if train else 'Testing top '
	pp1.eval()
	with torch.no_grad(): 
		correct1 = 0
		total = 0
		for i, (images, labels) in enumerate(loader): 
			images = images.cuda()
			labels = labels.cuda()
			log1, _, _, _ = pp1(images)
			for j in range(images.size(0)): 
				sortind = torch.argsort(log1[j], descending = True)
				for ind in sortind[:k]: 
					if ind == labels[j]: 
						correct1 += 1
			total += images.size(0)
		acc1 = 100 * correct1 / total
		print(tt + str(k) + ' accuracy: {0}'.format(acc1))

def batch_class_acc_topk(labels, logits, k=1): 
	out = torch.zeros(num_classes)
	total = torch.zeros(num_classes)

	batch_size = labels.size(0)
	
	for i in range(batch_size): 
		sortind = torch.argsort(logits[i], descending = True)
		total[labels[i]] += 1
		for ind in sortind[:k]:
			if ind == labels[i]: 
				out[ind] += 1

	return out, total


def accuracy(pp1, pp2, dist, loader, writer, iteration, train=True, log=print): 
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	pp2.eval()
	dist.eval()
	maxl = utils.MaxLayerFixedSize().cuda()
	crop = utils.CropLayer().cuda()
	with torch.no_grad():
		correct1 = 0
		correct2 = 0
		total = 0
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			batch_size = images.size(0)
			log1, _, activation_maps, _ = pp1(images)
			conv = dist(activation_maps, log1, pp1.module.prototype_class_identity.cuda())
			w_offs, w_ends, h_offs, h_ends = maxl(conv)
			scaledA_x = crop(images, w_offs, w_ends, h_offs, h_ends)
			log2, _, _, _ = pp2(scaledA_x)
			total += labels.size(0)
			_, predicted1 = torch.max(log1, 1)
			_, predicted2 = torch.max(log2, 1)
			correct1 += (predicted1 == labels).sum().item()
			correct2 += (predicted2 == labels).sum().item()
		acc1 = 100 * correct1 / total
		acc2 = 100 * correct2 / total
		log(tt + 'accuracy scale 1: {0}'.format(acc1))
		log(tt + 'accuracy scale 2: {0}'.format(acc2))
	if train: 
		writer.add_scalar('trainacc1', acc1, iteration)
		writer.add_scalar('trainacc2', acc2, iteration)
	else: 
		writer.add_scalar('testacc1', acc1, iteration)
		writer.add_scalar('testacc2', acc2, iteration)

def accuracyplus(pp1, pp2, dist, loader, writer, iteration, cutoff, train=True, log=print):
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	pp2.eval()
	dist.eval()
	maxl = utils.MaxLayerFixedSize().cuda()
	crop = utils.CropLayer().cuda()
	with torch.no_grad():
		correct1 = 0
		correct2 = 0
		correct12 = 0
		correct12p = 0
		total = 0
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			log1, min_distances, activation_maps, _ = pp1(images)
			if top_proto: 
				conv = dist(activation_maps, min_distances)
			else: 
				conv = dist(activation_maps, log1, pp1.module.prototype_class_identity.cuda())
			w_offs, w_ends, h_offs, h_ends = maxl(conv)
			scaledA_x = crop(images, w_offs, w_ends, h_offs, h_ends)
			log2, _, _, _ = pp2(scaledA_x)
			log12 = log1 + log2

			prob1 = F.softmax(log1, dim=1)
			prob2 = F.softmax(log2, dim=1)
			prob12 = prob1 + prob2

			total += labels.size(0)

			for i in range(images.size(0)): 
				sortlog1 = torch.argsort(log1[i], descending = True)
				sortlog1 = sortlog1[:cutoff]
				#print('sortlog1: ', sortlog1)
				for k in range(log2[i].size(0)): 
					topk = False
					for l in range(cutoff): 
						if k == sortlog1[l]:
							topk = True
					if not topk: 
						log2[i][k] = float('-inf')

			#print('log2: ', log2)
			_, predicted1 = torch.max(log1, 1)
			_, predicted2 = torch.max(log2, 1)
			_, predicted12 = torch.max(log12, 1)
			_, predicted12p = torch.max(prob12, 1)
			#print('predicted1: ', predicted1)
			correct1 += (predicted1 == labels).sum().item()
			correct2 += (predicted2 == labels).sum().item()
			correct12 += (predicted12 == labels).sum().item()
			correct12p += (predicted12p == labels).sum().item()

			if j == 0: 
				grid = torchvision.utils.make_grid(images)
				zoomgrid = torchvision.utils.make_grid(scaledA_x)
				writer.add_image('images', grid, iteration)
				writer.add_image('zoomed images', zoomgrid, iteration)

		acc1 = 100 * correct1 / total
		acc2 = 100 * correct2 / total
		acc12 = 100 * correct12 / total
		acc12p = 100 * correct12p / total
		log(tt + 'accuracy scale 1: {0}'.format(acc1))
		log(tt + 'accuracy scale 2: {0}'.format(acc2))
		log(tt + 'accuracy scale 1+2 (logit sum): {0}'.format(acc12))
		log(tt + 'accuracy scale 1+2 (prob sum): {0}'.format(acc12p))
	if train: 
		writer.add_scalar('trainacc1', acc1, iteration)
		writer.add_scalar('trainacc2', acc2, iteration)
		writer.add_scalar('trainacc12', acc12, iteration)
	else: 
		writer.add_scalar('testacc1', acc1, iteration)
		writer.add_scalar('testacc2', acc2, iteration)
		writer.add_scalar('testacc12', acc12, iteration)

	return acc12

def accuracyplus2(pp1, pp2, dist, loader, writer, iteration, cutoff, train=True, log=print):
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	pp2.eval()
	dist.eval()
	maxl = utils.MaxLayerFixedSize().cuda()
	crop = utils.CropLayer().cuda()
	with torch.no_grad():
		correct2 = 0
		correct12 = 0
		correct12cut = 0
		total = 0
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			log1, _, activation_maps, _ = pp1(images)
			conv = dist(activation_maps, log1, pp1.module.prototype_class_identity.cuda())
			w_offs, w_ends, h_offs, h_ends = maxl(conv)
			scaledA_x = crop(images, w_offs, w_ends, h_offs, h_ends)
			log2, _, _, _ = pp2(scaledA_x)
			total += labels.size(0)

			log12 = log1 + log2

			for i in range(images.size(0)): 
				sortlog1 = torch.argsort(log1[i], descending = True)
				sortlog1 = sortlog1[:cutoff]
				#print('sortlog1: ', sortlog1)
				for k in range(log2[i].size(0)): 
					topk = False
					for l in range(cutoff): 
						if k == sortlog1[l]:
							topk = True
					if not topk: 
						log2[i][k] = float('-inf')

			log12cut = log1 + log2

			_, predicted2 = torch.max(log2, 1)
			_, predicted12 = torch.max(log12, 1)
			_, predicted12cut = torch.max(log12cut, 1)
			correct2 += (predicted2 == labels).sum().item()
			correct12 += (predicted12 == labels).sum().item()
			correct12cut += (predicted12cut == labels).sum().item()

			if j == 0: 
				unnormalize = transforms.Normalize(unmean, unstd)
				for i in range(images.size(0)):
					images[i] = unnormalize(images[i])
					scaledA_x[i] = unnormalize(scaledA_x[i])
				grid = torchvision.utils.make_grid(images)
				zoomgrid = torchvision.utils.make_grid(scaledA_x)
				writer.add_image(tt + 'images', grid, iteration)
				writer.add_image(tt + 'zoomed images', zoomgrid, iteration)

		acc2 = 100 * correct2 / total
		acc12 = 100 * correct12 / total
		acc12cut = 100 * correct12cut / total
		log(tt + 'accuracy scale 2: {0}'.format(acc2))
		log(tt + 'accuracy scale 1+2: {0}'.format(acc12))
		log(tt + 'accuracy scale 1+2 (cutoff 2): {0}'.format(acc12cut))
	if train: 
		writer.add_scalar('trainacc2', acc2, iteration)
		writer.add_scalar('trainacc12', acc12, iteration)
	else: 
		writer.add_scalar('testacc2', acc2, iteration)
		writer.add_scalar('testacc12', acc12, iteration)

def accuracyplus3(pp1, pp2, dist, fc, loader, writer, iteration, cutoff, train=True, log=print):
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	pp2.eval()
	dist.eval()
	fc.eval()
	maxl = utils.MaxLayerFixedSize().cuda()
	crop = utils.CropLayer().cuda()
	with torch.no_grad():
		correct = 0
		total = 0
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			log1, _, activation_maps, act1 = pp1(images)
			conv = dist(activation_maps, log1, pp1.module.prototype_class_identity.cuda())
			w_offs, w_ends, h_offs, h_ends = maxl(conv)
			scaledA_x = crop(images, w_offs, w_ends, h_offs, h_ends)
			_, _, _, act2 = pp2(scaledA_x)
			total += labels.size(0)

			logits = fc(torch.cat((act1, act2), 1))

			_, predicted = torch.max(logits, 1)
			correct += (predicted == labels).sum().item()

		acc = 100 * correct / total
		log(tt + 'accuracy combined: {0}'.format(acc))
	if train: 
		writer.add_scalar('trainacc', acc, iteration)
	else: 
		writer.add_scalar('testacc', acc, iteration)

def get_sparsities(pp1, loader): 
	pp1.eval()
	with torch.no_grad(): 
		sparsities = []
		for i, (images, labels) in enumerate(loader): 
			images = images.cuda()
			labels = labels.cuda()
			if images.size(0) != 1: 
				print('have to feed with batch size 1 for sparsities')
				return -1
			correct, sparse = calculate_sparsity(images, pp1, labels)
			if not correct: 
				continue
			else: 
				sparsities.append(sparse)
	return sparsities
'''
def evalstats(pp1, pp2, dist, loader, cutoff, train=True, log=print):
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	pp2.eval()
	dist.eval()
	maxl = utils.MaxLayerFixedSize().cuda()
	crop = utils.CropLayer().cuda()
	with torch.no_grad():
		correct1 = 0
		correct2 = 0
		correct3 = 0
		total = 0
		crossentcorrect = 0
		crossentwrong = 0
		crossentcorrectreg = 0
		crossentwrongreg = 0
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			log1, _, activation_maps = pp1(images)
			conv = dist(activation_maps, log1, pp1.module.prototype_class_identity)
			w_offs, w_ends, h_offs, h_ends = maxl(conv)
			scaledA_x = crop(images, w_offs, w_ends, h_offs, h_ends)
			log2, _, _ = pp2(scaledA_x)
			total += labels.size(0)

			log2reg = log2

			for i in range(images.size(0)): 
				sortlog1 = torch.argsort(log1[i], descending = True)
				sortlog1 = sortlog1[:cutoff]
				#print('sortlog1: ', sortlog1)
				for k in range(log2[i].size(0)): 
					topk = False
					for l in range(cutoff): 
						if k == sortlog1[l]:
							topk = True
					if not topk: 
						log2[i][k] = float('-inf')
			#print('log2: ', log2)
			_, predicted1 = torch.max(log1, 1)
			_, predicted2 = torch.max(log2, 1)
			_, predicted2reg = torch.max(log2reg, 1)
			#print('predicted1: ', predicted1)
			correct1 += (predicted1 == labels).sum().item()
			correct2 += (predicted2 == labels).sum().item()
			correct3 += (predicted2reg == labels).sum().item()

			for i in range(images.size(0)): 
				crsent = F.cross_entropy(torch.unsqueeze(log2reg[i], 0), torch.unsqueeze(labels[i], 0))
				#print('crsent: ', crsent, flush=True)
				if predicted2[i] == labels[i]: 
					crossentcorrect += crsent
				else: 
					crossentwrong += crsent

				if predicted2reg[i] == labels[i]: 
					crossentcorrectreg += crsent
				else: 
					crossentwrongreg += crsent

		acc1 = 100 * correct1 / total
		acc2 = 100 * correct2 / total
		acc2reg = 100 * correct3 / total
		log(tt + 'accuracy scale 1: {0}'.format(acc1))
		log(tt + 'boosted accuracy scale 2: {0}'.format(acc2))
		log(tt + 'accuracy scale 2: {0}'.format(acc2reg))

		avgcrsentcorrect = crossentcorrect / correct2
		avgcrsentwrong = crossentwrong / (total - correct2)

		avgcrsentcorrectreg = crossentcorrectreg / correct3
		avgcrsentwrongreg = crossentwrongreg / (total - correct3)

		log(tt + 'average cross entropy (correct, boosted accuracy): {0}'.format(avgcrsentcorrect))
		log(tt + 'average cross entropy (incorrrect, boosted accuracy): {0}'.format(avgcrsentwrong))

		log(tt + 'average cross entropy (correct, regular accuracy): {0}'.format(avgcrsentcorrectreg))
		log(tt + 'average cross entropy (incorrect, regular accuracy): {0}'.format(avgcrsentwrongreg))
'''
def save_nonparallel(net, filename): 
	sd = net.module.state_dict()
	newsd = OrderedDict()
	for key in sd: 
		newkey = key.replace('.module', '')
		newsd[newkey] = sd[key]
	torch.save(newsd, filename)

def get_predicted_logits(ppnet, loader): 
	out = torch.zeros((ppnet.module.num_classes, ppnet.module.num_classes)).cuda()

	for i, (images, labels) in enumerate(loader): 
		with torch.no_grad(): 
			images = images.cuda()
			labels = labels.cuda()
			log1, _, _, _ = ppnet(images)
			out = batch_logits(labels, log1, out)

	return out

def batch_logits(labels, logits, out): 
	for i in range(labels.size(0)): 
		out[labels[i]] += logits[i]

	return out


def batch_class_acc(labels, logits, confusion):
	out = torch.zeros(num_classes)
	total = torch.zeros(num_classes)

	batch_size = labels.size(0)
	_, predicted = torch.max(logits, 1)
	correct = (predicted == labels)
	for i in range(batch_size): 
		total[labels[i]] += 1
		if correct[i]: 
			out[labels[i]] += 1
		confusion[labels[i]][predicted[i]] += 1 

	return out, total, confusion

def classaccuracy(pp1, loader, train=True, log=print, k=-1):
	top_proto = False
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	with torch.no_grad():
		correct1 = torch.zeros(num_classes)
		total = torch.zeros(num_classes)
		confusion1 = torch.zeros((num_classes, num_classes))
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			log1, min_distances = pp1(images)

			if k == -1: 
				batchcorr1, batchtotal1, confusion1 = batch_class_acc(labels, log1, confusion1)
			else: 
				batchcorr1, batchtotal1 = batch_class_acc_topk(labels, log1, k)		


			correct1 += batchcorr1
			total += batchtotal1
		
		acc1 = 100 * correct1 / total

		log('Level 1 Overall Accuracy: {0}'.format(torch.mean(acc1)))

		for i in range(num_classes): 
			log('Class ' + str(i) + ' ' + tt + 'accuracy scale 1(logit sum): {0}'.format(acc1[i]))

		print('Top 10 worst classes: ')

		sorted1, sortind1 = torch.sort(acc1, descending = False)

		print('1st level')
		for i in range(50): 
			log('Class ' + str(sortind1[i]) + ': ' + str(sorted1[i]) + ' (total examples): ' + str(total[sortind1[i]]))
			classnum = sortind1[i]
			foo, bar = torch.sort(confusion1[classnum], descending = True)
			for j in range(10): 
				print('Classified as class ' + str(bar[j]) + ' frequency: ' + str(foo[j]))

	return acc1, confusion1
'''
def classaccuracy_1lvl(pp1, loader, train=True, log=print):
	top_proto = False
	tt = 'Training ' if train else 'Testing '
	pp1.eval()
	with torch.no_grad():
		correct1 = torch.zeros(num_classes)
		total = torch.zeros(num_classes)
		for j, (images, labels) in enumerate(loader):
			images = images.cuda()
			labels = labels.cuda()
			log1, min_distances, activation_maps, _ = pp1(images)

			batchcorr, batchtotal = batch_class_acc(images, labels, log1)

			correct1 += batchcorr
			total += batchtotal
		
		acc1 = 100 * correct1 / total

		for i in range(num_classes): 
			log('Class ' + str(i) + ' ' + tt + 'accuracy (logit sum): {0}'.format(acc1[i]))

		print('Top 10 worst classes: ')

		sorted, sortind = torch.sort(acc1, descending = False)

		for i in range(10): 
			log('Class ' + str(sortind[i]) + ': ' + str(sorted[i]) + ' (total examples): ' + str(total[sortind[i]]))
'''
'''
def batch_confusion_matrix(logits, labels, matrix): 
	#matrix indices should be correct, predicted
	_, predicted = torch.max(logits, 1)
	y_pred = predicted.cpu().numpy()
	y_true = labels.cpu().numpy()
	matrix = matrix + confusion_matrix(y_true, y_pred)
	return matrix
'''
