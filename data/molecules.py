import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib
import random
import math
import pandas as pd


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]

            assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        count = 0
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

            if count == 0: 
                print(g)
                print(g.nodes)
                print(g.edges)
                print(g.ndata)
                print(g.edata)
                count += 1

        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
        data_dir='./data/molecules'
        
        if self.name == 'ZINC-full':
            data_dir='./data/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        else:            
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """

    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):
    """
        Converting the given graph to fully connected
        This function just makes full connections
        removes available edge features 
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass    
    
    return full_g



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    # print("LapPE = ", g.ndata['lap_pos_enc'].size())
    # print("L.size = ", L.shape)
    # print("idx = ", idx)
    return g

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g

def pre_process(old_list):
    """
    Graph data preprocessing
    """
    new_list = []
    total_nodes = 1         # change as required
    less_node_count = 0
    less_edge_count = 0

    for i in range(len(old_list)):
        row = old_list[i]
        g_tuple = row[:8] 
        score = row[-1]
        sim_scores = list(row[8:-1])     #
        row_list = []
        
        # adding node and edge features
        for grh in g_tuple:
            embed = grh.ndata['feat']
            embed = embed.to(torch.float32)
            node_dim = embed.shape[1]
#             print(">>> Embed shape:", embed.shape, node_dim)
            
            deficiency = 0
            # add extra nodes if less nodes found
            if (len(grh.nodes()) < total_nodes):
                less_node_count += 1
                deficiency = total_nodes - len(grh.nodes())
                grh.add_nodes(deficiency)
            
            # adding node features
            # append a null (0) vector for every extra padded node
            extra_embed = torch.zeros(deficiency, node_dim)
            embed = torch.cat((embed, extra_embed), dim=0)
            grh.ndata['feat'] = embed
#             print(">>> Padded Embed shape:", embed.shape)
            
            # make graph fully connected if zero edges found
#             if (grh.num_edges() == 0):
#                 less_edge_count += 1
#                 grh = make_full_graph(grh)
                
            grh.edata['feat'] = torch.ones(grh.number_of_edges(), 1)
            row_list.append(grh)

        row_list.append(torch.tensor(sim_scores))
        row_list.append(torch.tensor([score]))
        new_list.append(row_list)

    print("\n\n>>>>> LESS NODE COUNT =", less_node_count)
    print(">>>>> LESS EDGE COUNT =", less_edge_count, "\n\n")
    return new_list

class DataUnit(torch.utils.data.Dataset):

    def __init__(self, raw_list):
        self.size = len(raw_list)
        self.graph_lists = [ i[0] for i in raw_list ]
        self.graph_labels = [ i[-1] for i in raw_list ]
        self.sim_scores = [ i[1] for i in raw_list]       #
        # print("DataUnit Constructor sim_scores", self.sim_scores)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.sim_scores[idx], self.graph_labels[idx]

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading ZINC dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        with open('data/dgl_graphs.pickle',"rb") as data:
            data = pickle.load(data)
            print("\nORIGINAL GRAPHS")
            print(f"Total Data size = {len(data)} graphs!")
            print(f"Row size = {len(data[1])}")

            print("\nCONVERTED GRAPHS")
            new_data = pre_process(data)
            print(f"Data size = {len(new_data)} graphs!")

            # Define train/test/val splits in the ratio of 8:1:1
            # First shuffle g_list
            random.shuffle(new_data)
            total = len(new_data)
            part1 = math.ceil(0.8*total)
            part2 = math.ceil(0.9*total)

            train = new_data[:part1]
            val = new_data[part1:part2]
            test = new_data[part2:]
            # val = new_data[part1:]
            # test = new_data[part1:]

            # Train
            self.train_s_arg0 = DataUnit([ (i[0], i[-2], i[-1]) for i in train ])
            self.train_s_arg1 = DataUnit([ (i[1], i[-2], i[-1]) for i in train ])
            self.train_s_def = DataUnit([ (i[2], i[-2], i[-1]) for i in train ])
            self.train_s_rest = DataUnit([ (i[3], i[-2], i[-1]) for i in train ])

            self.train_m_arg0 = DataUnit([ (i[4], i[-2], i[-1]) for i in train ])
            self.train_m_arg1 = DataUnit([ (i[5], i[-2], i[-1]) for i in train ])
            self.train_m_def = DataUnit([ (i[6], i[-2], i[-1]) for i in train ])
            self.train_m_rest = DataUnit([ (i[7], i[-2], i[-1]) for i in train ])

            # Val
            self.val_s_arg0 = DataUnit([ (i[0], i[-2], i[-1]) for i in val ])
            self.val_s_arg1 = DataUnit([ (i[1], i[-2], i[-1]) for i in val ])
            self.val_s_def = DataUnit([ (i[2], i[-2], i[-1]) for i in val ])
            self.val_s_rest = DataUnit([ (i[3], i[-2], i[-1]) for i in val ])

            self.val_m_arg0 = DataUnit([ (i[4], i[-2], i[-1]) for i in val ])
            self.val_m_arg1 = DataUnit([ (i[5], i[-2], i[-1]) for i in val ])
            self.val_m_def = DataUnit([ (i[6], i[-2], i[-1]) for i in val ])
            self.val_m_rest = DataUnit([ (i[7], i[-2], i[-1]) for i in val ])

            # Test 
            self.test_s_arg0 = DataUnit([ (i[0], i[-2], i[-1]) for i in test ])
            self.test_s_arg1 = DataUnit([ (i[1], i[-2], i[-1]) for i in test ])
            self.test_s_def = DataUnit([ (i[2], i[-2], i[-1]) for i in test ])
            self.test_s_rest = DataUnit([ (i[3], i[-2], i[-1]) for i in test ])

            self.test_m_arg0 = DataUnit([ (i[4], i[-2], i[-1]) for i in test ])
            self.test_m_arg1 = DataUnit([ (i[5], i[-2], i[-1]) for i in test ])
            self.test_m_def = DataUnit([ (i[6], i[-2], i[-1]) for i in test ])
            self.test_m_rest = DataUnit([ (i[7], i[-2], i[-1]) for i in test ])

            print(self.train_s_arg0.__len__(), self.val_s_arg0.__len__(), self.test_s_arg0.__len__())

            print("Self.TRAIN.arg0 = ", self.train_s_arg0)
            print(self.train_s_arg0.__getitem__(2))
            print(self.train_s_arg0.__len__())
            print(self.train_s_arg0.__getitem__(2)[0].nodes())
            print(self.train_s_arg0.__getitem__(2)[0].edges())
            print("N_DATA =", self.train_s_arg0.__getitem__(2)[0].ndata['feat'].size(), "\n", self.train_s_arg0.__getitem__(2)[0].ndata['feat'])
            print(self.train_s_arg0.__getitem__(2)[1])
            print(self.train_s_arg0.__getitem__(2)[-1])
            print(type(self.train_s_arg0))

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, sim_scores, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        sim_scores = [ i.tolist() for i in sim_scores ]
        sim_scores = torch.tensor(sim_scores)

        return batched_graph, sim_scores, labels
    
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        print(">>>>> Calling _add_self_loops ")

        # Train
        self.train_s_arg0.graph_lists = [self_loop(g) for g in self.train_s_arg0.graph_lists]
        self.train_s_arg1.graph_lists = [self_loop(g) for g in self.train_s_arg1.graph_lists]
        self.train_s_def.graph_lists = [self_loop(g) for g in self.train_s_def.graph_lists]
        self.train_s_rest.graph_lists = [self_loop(g) for g in self.train_s_rest.graph_lists]

        self.train_m_arg0.graph_lists = [self_loop(g) for g in self.train_m_arg0.graph_lists]
        self.train_m_arg1.graph_lists = [self_loop(g) for g in self.train_m_arg1.graph_lists]
        self.train_m_def.graph_lists = [self_loop(g) for g in self.train_m_def.graph_lists]
        self.train_m_rest.graph_lists = [self_loop(g) for g in self.train_m_rest.graph_lists]

        # Val
        self.val_s_arg0.graph_lists = [self_loop(g) for g in self.val_s_arg0.graph_lists]
        self.val_s_arg1.graph_lists = [self_loop(g) for g in self.val_s_arg1.graph_lists]
        self.val_s_def.graph_lists = [self_loop(g) for g in self.val_s_def.graph_lists]
        self.val_s_rest.graph_lists = [self_loop(g) for g in self.val_s_rest.graph_lists]

        self.val_m_arg0.graph_lists = [self_loop(g) for g in self.val_m_arg0.graph_lists]
        self.val_m_arg1.graph_lists = [self_loop(g) for g in self.val_m_arg1.graph_lists]
        self.val_m_def.graph_lists = [self_loop(g) for g in self.val_m_def.graph_lists]
        self.val_m_rest.graph_lists = [self_loop(g) for g in self.val_m_rest.graph_lists]


        # Test
        self.test_s_arg0.graph_lists = [self_loop(g) for g in self.test_s_arg0.graph_lists]
        self.test_s_arg1.graph_lists = [self_loop(g) for g in self.test_s_arg1.graph_lists]
        self.test_s_def.graph_lists = [self_loop(g) for g in self.test_s_def.graph_lists]
        self.test_s_rest.graph_lists = [self_loop(g) for g in self.test_s_rest.graph_lists]

        self.test_m_arg0.graph_lists = [self_loop(g) for g in self.test_m_arg0.graph_lists]
        self.test_m_arg1.graph_lists = [self_loop(g) for g in self.test_m_arg1.graph_lists]
        self.test_m_def.graph_lists = [self_loop(g) for g in self.test_m_def.graph_lists]
        self.test_m_rest.graph_lists = [self_loop(g) for g in self.test_m_rest.graph_lists]

        print(">>>>> Self Loop Train List")
        print(self.train_s_arg0.graph_lists[0].ndata['feat'])

    def _make_full_graph(self):
        
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True

        print(">>>>> Calling _make_full_graph")

        # Train
        self.train_s_arg0.graph_lists = [make_full_graph(g) for g in self.train_s_arg0.graph_lists]
        self.train_s_arg1.graph_lists = [make_full_graph(g) for g in self.train_s_arg1.graph_lists]
        self.train_s_def.graph_lists = [make_full_graph(g) for g in self.train_s_def.graph_lists]
        self.train_s_rest.graph_lists = [make_full_graph(g) for g in self.train_s_rest.graph_lists]

        self.train_m_arg0.graph_lists = [make_full_graph(g) for g in self.train_m_arg0.graph_lists]
        self.train_m_arg1.graph_lists = [make_full_graph(g) for g in self.train_m_arg1.graph_lists]
        self.train_m_def.graph_lists = [make_full_graph(g) for g in self.train_m_def.graph_lists]
        self.train_m_rest.graph_lists = [make_full_graph(g) for g in self.train_m_rest.graph_lists]

        # Val
        self.val_s_arg0.graph_lists = [make_full_graph(g) for g in self.val_s_arg0.graph_lists]
        self.val_s_arg1.graph_lists = [make_full_graph(g) for g in self.val_s_arg1.graph_lists]
        self.val_s_def.graph_lists = [make_full_graph(g) for g in self.val_s_def.graph_lists]
        self.val_s_rest.graph_lists = [make_full_graph(g) for g in self.val_s_rest.graph_lists]

        self.val_m_arg0.graph_lists = [make_full_graph(g) for g in self.val_m_arg0.graph_lists]
        self.val_m_arg1.graph_lists = [make_full_graph(g) for g in self.val_m_arg1.graph_lists]
        self.val_m_def.graph_lists = [make_full_graph(g) for g in self.val_m_def.graph_lists]
        self.val_m_rest.graph_lists = [make_full_graph(g) for g in self.val_m_rest.graph_lists]

        # Test
        self.test_s_arg0.graph_lists = [make_full_graph(g) for g in self.test_s_arg0.graph_lists]
        self.test_s_arg1.graph_lists = [make_full_graph(g) for g in self.test_s_arg1.graph_lists]
        self.test_s_def.graph_lists = [make_full_graph(g) for g in self.test_s_def.graph_lists]
        self.test_s_rest.graph_lists = [make_full_graph(g) for g in self.test_s_rest.graph_lists]

        self.test_m_arg0.graph_lists = [make_full_graph(g) for g in self.test_m_arg0.graph_lists]
        self.test_m_arg1.graph_lists = [make_full_graph(g) for g in self.test_m_arg1.graph_lists]
        self.test_m_def.graph_lists = [make_full_graph(g) for g in self.test_m_def.graph_lists]
        self.test_m_rest.graph_lists = [make_full_graph(g) for g in self.test_m_rest.graph_lists]

        print("\n>>>>> Full Graph conversion done!\n")
    
    
    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors

        print(">>>>> Calling _add_laplacian_positional_encodings")

        # Train
        self.train_s_arg0.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_s_arg0.graph_lists]
        self.train_s_arg1.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_s_arg1.graph_lists]
        self.train_s_def.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_s_def.graph_lists]
        self.train_s_rest.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_s_rest.graph_lists]

        self.train_m_arg0.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_m_arg0.graph_lists]
        self.train_m_arg1.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_m_arg1.graph_lists]
        self.train_m_def.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_m_def.graph_lists]
        self.train_m_rest.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train_m_rest.graph_lists]

        # Val
        self.val_s_arg0.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_s_arg0.graph_lists]
        self.val_s_arg1.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_s_arg1.graph_lists]
        self.val_s_def.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_s_def.graph_lists]
        self.val_s_rest.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_s_rest.graph_lists]

        self.val_m_arg0.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_m_arg0.graph_lists]
        self.val_m_arg1.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_m_arg1.graph_lists]
        self.val_m_def.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_m_def.graph_lists]
        self.val_m_rest.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val_m_rest.graph_lists]

        # Test
        self.test_s_arg0.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_s_arg0.graph_lists]
        self.test_s_arg1.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_s_arg1.graph_lists]
        self.test_s_def.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_s_def.graph_lists]
        self.test_s_rest.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_s_rest.graph_lists]

        self.test_m_arg0.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_m_arg0.graph_lists]
        self.test_m_arg1.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_m_arg1.graph_lists]
        self.test_m_def.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_m_def.graph_lists]
        self.test_m_rest.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test_m_rest.graph_lists]

        print("\n>>>>> Lap Pos encoding done!\n")


    def _add_wl_positional_encodings(self):
        
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]
