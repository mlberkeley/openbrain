import numpy as np
import networkx
import time
import utils.gnumpy as gpu
import gym

from utils.ubigraph import Ubigraph

class Brain(object):
    """
    The brain class for openbrain.
    """

    def __init__(self, num_input, num_hidden, num_output, density=0.5):
        """
        Constructs an open brain object.
        :param num_input: The number of inputs
        :param num_output: The number of outputs
        :param num_hidden: The number of hidden neurons.
        """
        self.initialize_graph(num_input,num_hidden, num_output,  density=0.5)
        self.visualize()


    def initialize_graph(self, num_input, num_hidden,  num_output,density):
        """
        Initializes the open brain graph.
        :param num_input:
        :param num_hidden:
        :param num_output:
        :param density:
        :return:
        """
        self.num_neurons = num_input + num_output + num_hidden
        self.C = np.zeros((self.num_neurons, self.num_neurons))

        adjacency_list = self.define_toplogy(num_input, num_hidden,  num_output, density)

        for i, adjacent in enumerate(adjacency_list):
            for node in adjacent:
                self.C[i, node] = np.random.rand(1)*0.9

        self.C = gpu.garray(self.C)

    def define_toplogy(self, num_input, num_hidden,  num_output, density):
        """
        Defines the topology of the OpenBrain network.
        :param num_input:
        :param num_hidden:
        :param num_output:
        :param density:
        :return:
        """
        topo = networkx.DiGraph(networkx.watts_strogatz_graph(self.num_neurons, 5, density, seed=None)).to_directed()
        adjacency_list = topo.adjacency_list()


        # Pick the output neurons to be those with highest in degree
        in_deg = np.array([topo.in_degree(x) for x,_ in enumerate(adjacency_list)])
        self.output_neurons = np.argpartition(in_deg, -num_output)[-num_output:]
        print(self.output_neurons)
        print([topo.in_degree(x) for x in self.output_neurons])

        # Pick the input neurons to be those with highest out degree
        out_deg = np.array([topo.out_degree(x) if x not in self.output_neurons else -1
                            for x,_ in enumerate(adjacency_list)])
        self.input_neurons = np.argpartition(out_deg, -num_input)[-num_input:]

        # Output neurons do not fire out.
        for adjacent_neurons in adjacency_list:
            for out_neuron in self.output_neurons:
                if out_neuron in adjacent_neurons:
                    adjacent_neurons.remove(out_neuron)

        # Disconnect input -> output
        for out in self.output_neurons:
            for inp in self.input_neurons:
                if out in adjacency_list[inp]: adjacency_list[inp].remove(out)
                if inp in adjacency_list[out]: adjacency_list[out].remove(inp)


        for i, adjacent in enumerate(adjacency_list):
            if i not in self.input_neurons and i not in self.output_neurons:
                for n in adjacent:
                    if i in adjacency_list[n]:
                        if np.random.rand(1)>0.5:
                            adjacent.remove(n)
                        else:
                            adjacency_list[n].remove(i)

        # Let nothing enter the input neurons
        for inp in self.input_neurons:
            adjacency_list[inp] = []

        return adjacency_list


    def visualize(self):
        self.ubi = Ubigraph(self.C)
        # Color the input and output neurons
        self.ubi.set_properties({
            "color": "#1E90FF"
        }, self.input_neurons)

        self.ubi.set_properties({
            "color": "#FFD700"
        }, self.output_neurons)

def get_random_input(b):
    return gpu.garray(np.array([np.random.rand(1) * 2 - 1.0 if i in b.input_neurons else 0.0 for i in range(b.num_neurons)],
             dtype=np.float32))


if __name__ == '__main__':
    b = Brain(2,120,1, 0.2)
    v =get_random_input(b)
    i =0

    grad = list()

        for i,x in enumerate(v.as_numpy_array()):
            b.ubi.set_properties({
                "size": str(x)
                }, [i])
    for i_episode in range(500):
            for t in range(100):
                



                x = get_random_input(b)
                net = b.C.dot(v*0.9 + x)
                #dsigma = gpu.diagflat(gpu.tanh(net)*(1 - gpu.tanh(net)))
                #dnet = gpu.garray(np.multiply.outer(np.identity(v.shape[0]), v.as_numpy_array()))
                #dsigmadv = gpu.tensordot(dsigma, gpu.diagflat(v))
                #grad.append(dsigmadv)
                v = gpu.tanh(net)

                #diff = gpu.garray (v.as_numpy_array()[b.output_neurons] - np.array([0.2]))
                #dsigmadC = gpu.tensordot(dsigma, dnet, 1)
                #dpidC = gpu.garray(dsigmadC.as_numpy_array()[b.output_neurons,:,:])


        b.C += 0.1*dEdC
        print(i, v.as_numpy_array()[b.output_neurons])

                #dEdC = gpu.tensordot(diff, dpidC,1)


                #b.C += 0.01*dEdC
                #print(i, v.as_numpy_array()[b.output_neurons])

                for n, volt in enumerate(v):
                    if not( n in b.input_neurons or n in b.output_neurons):
                        b.ubi.set_properties({
                            "size": str(abs(volt)*2),
                            "color": "#FF0000"
                            }, [n])
