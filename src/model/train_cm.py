import argparse
import pickle
import sys

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data_loading import *
from model import *


def main():

    sys.stdout.flush()

    parser = argparse.ArgumentParser()

    parser.add_argument('--sr', default=None, type=str, required=True, help='Subreddit.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')

    args = parser.parse_args()

    print('Subreddit: {}'.format(args.sr))
    print('Learning rate: {}'.format(args.lr))
    print('Number of epochs: {}'.format(args.n_epochs))

    sr = args.sr

    # Load POS dictionary
    with open('../../data/pos/{}.pkl'.format(sr), 'rb') as f:
        pos_dict = pickle.load(f)

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

    train_data = DerivDataset(sampler.pos_train, sampler)
    deriv_collator = DerivCollator(train_data.c2id)

    val_data = DerivDataset(sampler.pos_val, sampler)
    val_data.get_full_batch(sampler.neg_val, sampler)
    val_loader = DataLoader(val_data, batch_size=128, collate_fn=deriv_collator)

    test_data = DerivDataset(sampler.pos_test, sampler)
    test_data.get_full_batch(sampler.neg_test, sampler)
    test_loader = DataLoader(test_data, batch_size=128, collate_fn=deriv_collator)

    INPUT_DIM = len(train_data.c2id) + 2
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    DROPOUT = 0.2
    N_EPOCHS = args.n_epochs

    predictor = CM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(predictor.parameters(), lr=args.lr)
    predictor = predictor.to(device)

    for epoch in range(1, N_EPOCHS + 1):

        neg_edges = sampler.negative_edges(int(sampler.pos_train.shape[1] / 2))
        train_data.get_full_batch(neg_edges, sampler)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=deriv_collator)

        for i, batch in enumerate(train_loader):
            a, s, l = batch
            a, s, l = a.to(device), s.to(device), l.to(device)

            predictor.train()
            optimizer.zero_grad()
            output = predictor(a, s)
            loss = criterion(output, l)
            loss.backward()
            optimizer.step()

        predictor.eval()

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in val_loader:
                a, s, l = batch
                a, s, l = a.to(device), s.to(device), l.to(device)

                output = predictor(a, s)

                y_true.extend(l.tolist())
                y_pred.extend(output.tolist())

        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        print('Epoch {}:\t{:.3f}\t{:.3f}'.format(epoch, ap, auc))

    predictor.eval()

    y_true = list()
    y_pred = list()

    with torch.no_grad():

        for batch in test_loader:

            a, s, l = batch
            a, s, l = a.to(device), s.to(device), l.to(device)

            output = predictor(a, s)

            y_true.extend(l.tolist())
            y_pred.extend(output.tolist())

    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print('Test:\t{:.3f}\t{:.3f}'.format(ap, auc))

    with open('results/cm_%s.txt' % sr, 'w') as f:
        f.write("%s %s\n" % (ap, auc))


if __name__ == '__main__':
    main()
