import torch
import pickle
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def convert(pci, ngp, nlp): 
	pci = pci[ngp:ngp+nlp, :]

	pos_connections = torch.argmax(pci, dim=1).unsqueeze(1)
	neg_connections = torch.argmin(pci, dim=1).unsqueeze(1)
	
	connections = torch.cat((pos_connections, neg_connections), dim=1)
	
	array = np.zeros((200,200))
	
	for i in range(pos_connections.size(0)): 
		array[pos_connections[i]][neg_connections[i]] += 1

	return torch.tensor(array)

#ppnet = torch.load(open('10nopush0.7500.pth', 'rb'))
#pci = ppnet.prototype_class_identity_negc
#pci = pickle.load(open('pci_dump_sym.p', 'rb'))

#array = convert(pci)

#pickle.dump(array, open('connections_matrix.p', 'wb'))

