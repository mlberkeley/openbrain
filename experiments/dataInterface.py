import json
import numpy as np 
import os

def getScalar(dir, stat, layer, neuron, func, run='.'): ### stat is one of max, mean, min, stddev, neuron is something like '1_4', func is one of action, Q or Qtarget
	if func == 'action':
		func = ''
		funcAddon = ''
	else:
		funcAddon = '_2F'
	filename = "scalars_run_{}_tag_{}_2Fsubcritic_layer{}_2Fsubcritic_n{}_2F{}{}Identity_3A0.json".format(run, \
																							stat, \
																							layer, \
																							neuron, \
																							func, \
																							funcAddon)
	with open(os.path.join(dir, filename)) as f:
		arr = np.array(json.loads(f.read()))

	arr = arr[:, 1:] # First Column doesn't contain useful data

	return arr # first column is timestep, second column is value