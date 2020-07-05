import math
from collections import Counter

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset


class BiEdgeSampler:

    def __init__(self, G, nodes_1, nodes_2):

        # Load nodes
        self.nodes_1 = list(nodes_1)
        self.idx_1 = np.arange(0, len(self.nodes_1))
        self.nodes_2 = list(nodes_2)
        self.idx_2 = np.arange(len(self.nodes_1), len(self.nodes_1) + len(self.nodes_2))
        self.all_nodes = np.array(nodes_1 + nodes_2)

        # Load graph
        self.G = G
        self.A = nx.adjacency_matrix(self.G, nodelist=self.all_nodes)
        self.edge_index = np.stack((self.A.tocoo().row, self.A.tocoo().col))
        self.edge_set = set(map(tuple, self.edge_index.transpose().tolist()))

        self.pop_1 = self.edge_index[0][np.isin(self.edge_index[0], self.idx_1)]
        self.pop_2 = self.edge_index[0][np.isin(self.edge_index[0], self.idx_2)]

        self.pos_train = None
        self.pos_val = None
        self.pos_test = None
        self.neg_val = None
        self.neg_test = None

        self.all_edges = self.edge_set

    def train_val_test(self, val=0.05, test=0.1):

        np.random.seed(0)

        row, col = self.edge_index
        mask = row < col
        row, col = row[mask], col[mask]

        n_val = math.floor(val * len(row))
        n_test = math.floor(test * len(row))
        n_train = len(row) - n_val - n_test

        perm = np.random.permutation(len(row))

        row, col = row[perm], col[perm]

        self.pos_val = np.stack((row[:n_val], col[:n_val]))
        self.pos_test = np.stack((row[n_val:n_val + n_test], col[n_val:n_val + n_test]))

        pos_train_row = np.concatenate((row[n_val + n_test:], col[n_val + n_test:]))
        pos_train_col = np.concatenate((col[n_val + n_test:], row[n_val + n_test:]))

        self.pos_train = np.stack((pos_train_row, pos_train_col))

        assert len(self.pos_val[0]) == n_val
        assert len(self.pos_test[0]) == n_test
        assert len(self.pos_train[0]) == 2 * n_train

        # Sample negative edges for val and test
        neg_edge_list = []
        rest = n_val + n_test

        while rest > 0:
            rand_1 = np.random.randint(0, len(self.pop_1), rest)
            rand_2 = np.random.randint(0, len(self.pop_2), rest)

            for n_1, n_2 in zip(self.pop_1[rand_1], self.pop_2[rand_2]):
                if (n_1, n_2) not in self.all_edges:
                    neg_edge_list.append((n_1, n_2))
                    self.all_edges.update([(n_1, n_2), (n_2, n_1)])
            rest = n_val + n_test - len(neg_edge_list)

        neg_edge_list_val = neg_edge_list[:n_val]
        neg_edge_list_test = neg_edge_list[n_val:n_val + n_test]

        self.neg_val = np.array(neg_edge_list_val).transpose()
        self.neg_test = np.array(neg_edge_list_test).transpose()

    def negative_edges(self, n):
        neg_edge_set = set()

        rest = n

        while rest > 0:
            rand_1 = np.random.randint(0, len(self.pop_1), rest)
            rand_2 = np.random.randint(0, len(self.pop_2), rest)

            for n_1, n_2 in zip(self.pop_1[rand_1], self.pop_2[rand_2]):
                if (n_1, n_2) not in self.all_edges and (n_1, n_2) not in neg_edge_set and (n_2, n_1) not in neg_edge_set:
                    neg_edge_set.add((n_1, n_2))
            rest = n - len(neg_edge_set)

        assert len(neg_edge_set) == n

        return np.array(list(neg_edge_set)).transpose()


class DerivDataset(Dataset):

    def __init__(self, pos_edges, sampler):

        self.vocab = Counter()

        self.affixes = list()
        self.stems = list()
        self.labels = list()

        for i, j in zip(pos_edges[0], pos_edges[1]):

            if '$' in sampler.all_nodes[i]:
                self.affixes.append(sampler.all_nodes[i])
                self.stems.append(sampler.all_nodes[j])

                self.labels.append(1)

                self.vocab.update(list(self.affixes[-1]))
                self.vocab.update(list(self.stems[-1]))

        self.c2id = {c: i + 2 for i, c in enumerate(c for c, count in self.vocab.most_common())}

    def get_full_batch(self, neg_edges, sampler):

        self.full_affixes = list()
        self.full_stems = list()
        self.full_labels = list()

        self.full_affixes.extend(self.affixes[:])
        self.full_stems.extend(self.stems[:])
        self.full_labels.extend(self.labels[:])

        for i, j in zip(neg_edges[0], neg_edges[1]):

            if '$' in sampler.all_nodes[i]:

                self.full_affixes.append(sampler.all_nodes[i])
                self.full_stems.append(sampler.all_nodes[j])
            else:
                self.full_affixes.append(sampler.all_nodes[j])
                self.full_stems.append(sampler.all_nodes[i])

            self.full_labels.append(0)

    def __len__(self):
        return len(self.full_labels)

    def __getitem__(self, idx):

        a = self.full_affixes[idx]
        s = self.full_stems[idx]
        l = self.full_labels[idx]

        return a, s, l


class DerivCollator():

    def __init__(self, c2id):

        self.c2id = c2id

    def __call__(self, batch):

        batch_size = len(batch)

        affixes = [a for a, s, l in batch]
        stems = [s for a, s, l in batch]
        labels = [l for a, s, l in batch]

        max_a = max(len(a) for a in affixes)
        max_s = max(len(s) for s in stems)

        affixes_pad = np.zeros((batch_size, max_a))
        stems_pad = np.zeros((batch_size, max_s))

        for i, a in enumerate(affixes):
            affixes_pad[i, :len(a)] = [self.c2id[c] if c in self.c2id else 1 for c in a]

        for i, s in enumerate(stems):
            stems_pad[i, :len(s)] = [self.c2id[c] if c in self.c2id else 1 for c in s]

        return torch.tensor(affixes_pad).long(), torch.tensor(stems_pad).long(), torch.tensor(labels).float()


def load_g(file):
    g = nx.read_gpickle(file)
    return g


def print_g(g):
    print('Number of edges: %s' % len(g.edges))
    print('Number of nodes: %s' % len(g.nodes))


def print_cc(g):
    print('Number of nodes in connected components: %s' % set(len(c) for c in nx.connected_components(g)))


def load_v(file):
    vector_dict = dict()
    with open(file, 'r') as f:
        for line in f:
            split_line = line.strip().split()
            w, v = split_line[0], [float(val) for val in split_line[1:]]
            vector_dict[w] = v
    return vector_dict


def sync_g_v(g, vector_dict):
    not_found = list()
    out_dict = dict()
    nodes = list(g.nodes)
    for n in nodes:
        if n in vector_dict:
            out_dict[n] = vector_dict[n]
        elif '$' in n:
            [a_1, a_2] = n.split('$')
            if a_1 in vector_dict and a_2 == '':
                out_dict[n] = vector_dict[a_1]
            elif a_2 in vector_dict and a_1 == '':
                out_dict[n] = vector_dict[a_2]
            elif a_1 in vector_dict and a_2 in vector_dict:
                out_dict[n] = [sum(x) for x in zip(vector_dict[a_1], vector_dict[a_2])]
            else:
                not_found.append(n)
                g.remove_node(n)
        else:
            not_found.append(n)
            g.remove_node(n)
    print('Not found: %s' % len(not_found))
    return out_dict


def clean_g(g):
    max_c = max(len(c) for c in nx.connected_components(g))
    for c in list(nx.connected_components(g)):
        if len(c) != max_c:
            for n in c:
                g.remove_node(n)


def check_g_v(g, vector_dict):
    problem = False
    for n in g.nodes:
        if n not in vector_dict:
            problem = True
            break
    if problem:
        print('Some nodes are missing in the vectors')
    else:
        print('Everything good!')


def get_x(vector_dict, nodes_1, nodes_2):
    x = []
    for n in nodes_1:
        x.append(vector_dict[n])
    for n in nodes_2:
        x.append(vector_dict[n])
    return x


def get_a(g, nodes_1, nodes_2):
    a = nx.adjacency_matrix(g, nodelist=nodes_1 + nodes_2)
    return a


def idx_to_node(idx, nodes):
    return nodes[idx]
