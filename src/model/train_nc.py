import argparse
import sys

import torch
from torch_geometric.data import Data

from data_loading import *
from model import *


def main():

    sys.stdout.flush()

    parser = argparse.ArgumentParser()

    parser.add_argument('--sr', default=None, type=str, required=True, help='Subreddit.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')

    args = parser.parse_args()

    print('Subreddit: {}'.format(args.sr))
    print('Number of epochs: {}'.format(args.n_epochs))

    sr = args.sr
    n_epochs = int(args.n_epochs)

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

    # Get feature matrix
    x = get_x(v, nodes_1, nodes_2)

    # Create edge sampler
    sampler = BiEdgeSampler(g, nodes_1, nodes_2)

    # Split edges into train, val, and test set
    sampler.train_val_test()

    pos_train = torch.tensor(sampler.pos_train.tolist(), dtype=torch.long)
    pos_val = torch.tensor(sampler.pos_val.tolist(), dtype=torch.long)
    neg_val = torch.tensor(sampler.neg_val.tolist(), dtype=torch.long)
    pos_test = torch.tensor(sampler.pos_test.tolist(), dtype=torch.long)
    neg_test = torch.tensor(sampler.neg_test.tolist(), dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)

    data = Data(x=x, pos_train=pos_train, pos_val=pos_val, neg_val=neg_val, pos_test=pos_test, neg_test=neg_test)
    print(data)

    input_dim = data.num_features
    latent_dim = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = GAE(MLPEncoder(input_dim, latent_dim, p1=0.1, p2=0.7), Decoder()).to(device)

    print(model)

    x, train_pos_edge_index = data.x.to(device), data.pos_train.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x)
        loss = model.bce_loss(z, train_pos_edge_index, sampler)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x)
            auc, ap = model.test(z, data.pos_val, data.neg_val)
            if epoch % 80 == 0:
                print(ap, auc)

    print()
    with torch.no_grad():
        z = model.encode(x)
        auc, ap = model.test(z, data.pos_test, data.neg_test)
        print(ap, auc)
        print()
        print()

    with open('results/nc_%s.txt' % sr, 'a') as f:
        f.write("%s %s\n" % (ap, auc))


if __name__ == '__main__':
    main()
