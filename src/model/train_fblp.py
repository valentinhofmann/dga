import argparse
import random
import sys
from math import log

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loading import *


# Define function to find one-hop neighbors of node n
def get_n(n_id, edge_index):
    return set(edge_index[1][np.isin(edge_index[0], [n_id])])


# Get intersection and union of two-hop neighbors of node n and one-hop neighbors of node m
def get_common(n_id, m_id, n_dict):
    set_1 = set()

    for i in n_dict[n_id]:
        set_1 |= n_dict[i]

    set_2 = n_dict[m_id]

    return set_1.intersection(set_2), set_1.union(set_2)


def main():

    sys.stdout.flush()

    parser = argparse.ArgumentParser()

    parser.add_argument('--sr', default=None, type=str, required=True, help='Subreddit.')

    args = parser.parse_args()

    sr = args.sr

    print('Subreddit: {}'.format(sr))

    # Load derivation graph
    g = load_g('../../data/graphs/%s_g.p' % sr)
    print_g(g)

    # Load vectors
    v_temp = load_v('../../data/embeddings/vectors_%s_10_100_m.txt' % sr)

    # Synchronize graph and vectors
    v = sync_g_v(g, v_temp)

    print_g(g)
    print_cc(g)

    # Remove small separate clusters
    clean_g(g)

    print_cc(g)

    print(len(g.nodes), len(v.keys()))

    check_g_v(g, v)

    print('Bipartite: %s' % nx.is_bipartite(g))

    [nodes_1, nodes_2] = sorted([sorted(list(s)) for s in nx.bipartite.sets(g)], key=lambda x: len(x))

    print(len(nodes_1), len(nodes_2))

    # Create edge sampler
    sampler = BiEdgeSampler(g, nodes_1, nodes_2)

    # Split edges into train, val, and test set
    sampler.train_val_test()

    n = len(sampler.all_nodes)

    d_dict = {}
    n_dict = {}

    for i in range(n):
        d_dict[i] = np.count_nonzero(sampler.pos_train[0] == i)
        n_dict[i] = get_n(i, sampler.pos_train)

    pa = []
    cn = []
    js = []
    aa = []
    y = []

    # Loop over test data
    for n, m in zip(sampler.pos_test[0], sampler.pos_test[1]):
        inter_1, union_1 = get_common(n, m, n_dict)
        inter_2, union_2 = get_common(m, n, n_dict)

        # Calculate preferential attachment score
        pa.append(d_dict[n] * d_dict[m])

        # Calculate common neighbors score
        cn.append(len(inter_1) + len(inter_2))

        # Calculate Jaccard similarity score
        js_1 = len(inter_1) / len(union_1) if len(union_1) else random.choice([0, 1])
        js_2 = len(inter_2) / len(union_2) if len(union_2) else random.choice([0, 1])
        js.append(js_1 + js_2)

        # Calculate Adamic Adar score
        aa_1 = sum([1 / log(d_dict[i]) for i in inter_1]) if len(inter_1) else 0
        aa_2 = sum([1 / log(d_dict[i]) for i in inter_2]) if len(inter_2) else 0
        aa.append(aa_1 + aa_2)

        y.append(1)

    for n, m in zip(sampler.neg_test[0], sampler.neg_test[1]):
        inter_1, union_1 = get_common(n, m, n_dict)
        inter_2, union_2 = get_common(m, n, n_dict)

        # Calculate preferential attachment score
        pa.append(d_dict[n] * d_dict[m])

        # Calculate common neighbors score
        cn.append(len(inter_1) + len(inter_2))

        # Calculate Jaccard similarity score
        js_1 = len(inter_1) / len(union_1) if len(union_1) else random.choice([0, 1])
        js_2 = len(inter_2) / len(union_2) if len(union_2) else random.choice([0, 1])
        js.append(js_1 + js_2)

        # Calculate Adamic Adar score
        aa_1 = sum([1 / log(d_dict[i]) for i in inter_1]) if len(inter_1) else 0
        aa_2 = sum([1 / log(d_dict[i]) for i in inter_2]) if len(inter_2) else 0
        aa.append(aa_1 + aa_2)

        y.append(0)

    pa_max = max(pa)
    pa_norm = [v / pa_max for v in pa]
    pa_ap = average_precision_score(y, pa_norm)
    pa_auc = roc_auc_score(y, pa_norm)

    cn_max = max(cn)
    cn_norm = [v / cn_max for v in cn]
    cn_ap = average_precision_score(y, cn_norm)
    cn_auc = roc_auc_score(y, cn_norm)

    js_max = max(js)
    js_norm = [v / js_max for v in js]
    js_ap = average_precision_score(y, js_norm)
    js_auc = roc_auc_score(y, js_norm)

    aa_max = max(aa)
    aa_norm = [v / aa_max for v in aa]
    aa_ap = average_precision_score(y, aa_norm)
    aa_auc = roc_auc_score(y, aa_norm)

    with open('results/js_%s.txt' % sr, 'w') as f:
        f.write("%s %s\n" % (js_ap, js_auc))

    with open('results/pa_cn_aa_%s.txt' % sr, 'w') as f:
        f.write("%s %s\n" % (pa_ap, pa_auc))
        f.write("%s %s\n" % (cn_ap, cn_auc))
        f.write("%s %s\n" % (aa_ap, aa_auc))


if __name__ == '__main__':
    main()
