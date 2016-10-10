import sys
import time
import xmlrpc.client

def default_node_props():
    return {
        'color': "#999999",
        'shape': "sphere"}


class Ubigraph:
    def __init__(self, adjacency,
                 server_url='http://192.168.168.33:20738/RPC2',
                 node_prop=default_node_props()):
        """
        Initializes the Ubigraph connection.
        :param adjacency:
        :param server_url:
        :param node_prop:
        """
        print("Connecting to UbiGraph at {}".format(server_url))

        server = xmlrpc.client.ServerProxy(server_url)
        self.G = server.ubigraph
        self.G.clear()

        print("Connected and cleared.")

        self.nodes = [None for n in range(adjacency.shape[0])]
        self.edges = []
        self.node_props = []
        self.num_edges = 0

        self.build_graph(adjacency, node_prop)

        print("Constructed {} nodes and {} edges."
              .format(len(self.nodes), self.num_edges))

    def build_graph(self, adjacency, node_property):
        for r, row in enumerate(adjacency):
            self.edges += [[None for cell in row]]
            for c, cell in enumerate(row):

                if not self.nodes[c]:
                    self.nodes[c] = self.make_node(node_property)
                if not self.nodes[r]:
                    self.nodes[r] = self.make_node(node_property)

                if cell != 0:
                    self.edges[r][c] = self.make_edge(self.nodes[c], self.nodes[r])
                    self.num_edges += 1


    def make_node(self, node_property):
        """
        Makes a new vertex.
        :param node_property: The properties of the vertex.
        :return:
        """
        # Try except because Ubigraph is old as hell!
        try: n = self.G.new_vertex()
        except: pass
        for prop, val in node_property.items():
            try: self.G.set_vertex_attribute(n, prop, val)
            except: return make_node(node_property)
        return n

    def make_edge(self, a, b):
        """
        Makes an edge from a to be.
        :param a:
        :param b:
        :return:
        """
        try: e = self.G.new_edge(a, b)
        except: return self.G.new_edge(a,b)

        try: self.G.set_edge_attribute(e, "arrow", "true")
        except: return self.G.new_edge(a,b)

        try: self.G.set_edge_attribute(e, "spline", "false")
        except: return self.G.new_edge(a,b)
        return e

    def set_properties(self, props, nodes):
        for node in nodes:
            for prop, val in props.items():
                try: self.G.set_vertex_attribute(self.nodes[node], prop, val)
                except: pass



