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

def getDistribution(dir, layer, neuron, func, run='.'):
	if func == 'action':
		func = ''
		funcAddon = ''
	else:
		funcAddon = '_2F'
	filename = "compressedHistograms_run_{}_tag_subcritic_layer{}_2Fsubcritic_n{}_2F{}{}Identity_3A0.json".format(run, \
																									layer, \
																									neuron, \
																									func, \
																									funcAddon)
	with open(os.path.join(dir, filename)) as f:
		arr = list(json.loads(f.read()))

	num_dists = 9
	data = np.zeros(len(arr) * num_dists * 2).reshape((num_dists, len(arr), 2))
	for i in range(len(arr)):
		for j in range(num_dists):
			data[j][i] = [arr[i][1], arr[i][2][j][1]]
	return data

## Example usage
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	max = getScalar('data', 'max', '2', '2', 'Q')
	dist = getDistribution('data', '2', '2', 'Q')
	plt.plot(max[:, 0], max[:, 1])
	plt.show()

	plt.plot(dist[0][:,0], dist[0][:, 1])
	plt.show()