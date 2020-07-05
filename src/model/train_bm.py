import argparse
import pickle
import re
import sys
from collections import defaultdict
from math import log

import spacy
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loading import *
from model import *


def main():

    sys.stdout.flush()

    parser = argparse.ArgumentParser()

    parser.add_argument('--sr', default=None, type=str, required=True, help='Subreddit.')

    args = parser.parse_args()

    sr = args.sr

    print('Subreddit: {}'.format(sr))

    # Load NLP model
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

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

    # Reconvert training edges into affix bundles and stems
    affixes = []
    stems = []
    for i, j in zip(sampler.pos_train[0], sampler.pos_train[1]):
        if '$' in sampler.all_nodes[i]:
            affixes.append(sampler.all_nodes[i])
            stems.append(sampler.all_nodes[j])

    print(len(affixes), len(stems))

    # Split affix bundles into affixes
    segments = []
    for affix in affixes:
        segment = [re.sub('[^a-zA-Z]', ' ', a).split() for a in affix.split('$')]
        segments.append(segment)

    p_trans_counts = defaultdict(Counter)
    s_trans_counts = defaultdict(Counter)

    p_counts = Counter()
    s_counts = Counter()

    errors = 0

    for segment, stem in zip(segments, stems):

        try:
            pos = pos_dict[stem]
        except KeyError:
            errors += 1
            doc = nlp(str(stem))
            pos = doc[0].tag_[:2]

        if len(segment[0]) > 0:

            prefixes = segment[0][::-1]

            for i in range(len(prefixes)):
                if i == 0:
                    p_trans_counts[pos][prefixes[i]] += 1
                    p_trans_counts['STEM'][prefixes[i]] += 1
                else:
                    p_trans_counts[prefixes[i - 1]][prefixes[i]] += 1
                p_counts[prefixes[i]] += 1

        if len(segment[1]) > 0:

            suffixes = segment[1]

            for i in range(len(suffixes)):
                if i == 0:
                    s_trans_counts[pos][suffixes[i]] += 1
                    s_trans_counts['STEM'][suffixes[i]] += 1
                else:
                    s_trans_counts[suffixes[i - 1]][suffixes[i]] += 1
                s_counts[suffixes[i]] += 1

    print(errors)

    # Normalize transition probabilities
    total_p = sum(p_counts.values())
    for p in p_counts:
        p_counts[p] /= total_p

    total_s = sum(s_counts.values())
    for s in s_counts:
        s_counts[s] /= total_s

    for counts in [p_trans_counts, s_trans_counts]:
        for affix in counts:
            total = sum(counts[affix].values())
            for a in counts[affix]:
                counts[affix][a] /= total

    ma_1 = []
    ma_2 = []
    y = []

    errors = 0

    # Calculate probabilites of test data
    for n, m in zip(sampler.pos_test[0], sampler.pos_test[1]):

        score_1 = 0
        score_2 = 0

        m_1 = sampler.all_nodes[n]
        m_2 = sampler.all_nodes[m]

        if '$' in m_1:
            affix = m_1
            stem = m_2
        else:
            stem = m_1
            affix = m_2

        try:
            pos = pos_dict[stem]
        except KeyError:
            errors += 1
            doc = nlp(str(stem))
            pos = doc[0].tag_[:2]

        segment = [re.sub('[^a-zA-Z]', ' ', a).split() for a in affix.split('$')]

        if len(segment[0]) > 0:

            prefixes = segment[0][::-1]

            for i in range(len(prefixes)):
                if i == 0:
                    if prefixes[i] in p_trans_counts[pos]:
                        score_1 += log(p_trans_counts[pos][prefixes[i]])
                    else:
                        score_1 += log(p_trans_counts['STEM'][prefixes[i]])
                else:
                    if prefixes[i] in p_trans_counts[prefixes[i - 1]]:
                        score_1 += log(p_trans_counts[prefixes[i - 1]][prefixes[i]])
                    else:
                        score_1 -= float('inf')
                score_2 += log(p_counts[prefixes[i]])

        if len(segment[1]) > 0:

            suffixes = segment[1]

            for i in range(len(suffixes)):
                if i == 0:
                    if suffixes[i] in s_trans_counts[pos]:
                        score_1 += log(s_trans_counts[pos][suffixes[i]])
                    else:
                        score_1 += log(s_trans_counts['STEM'][suffixes[i]])
                else:
                    if suffixes[i] in s_trans_counts[suffixes[i - 1]]:
                        score_1 += log(s_trans_counts[suffixes[i - 1]][suffixes[i]])
                    else:
                        score_1 -= float('inf')
                score_2 += log(s_counts[suffixes[i]])

        ma_1.append(score_1)
        ma_2.append(score_2)

        y.append(1)

    for n, m in zip(sampler.neg_test[0], sampler.neg_test[1]):

        score_1 = 0
        score_2 = 0

        m_1 = sampler.all_nodes[n]
        m_2 = sampler.all_nodes[m]

        if '$' in m_1:
            affix = m_1
            stem = m_2
        else:
            stem = m_1
            affix = m_2

        try:
            pos = pos_dict[stem]
        except KeyError:
            errors += 1
            doc = nlp(str(stem))
            pos = doc[0].tag_[:2]

        segment = [re.sub('[^a-zA-Z]', ' ', a).split() for a in affix.split('$')]

        if len(segment[0]) > 0:

            prefixes = segment[0][::-1]

            for i in range(len(prefixes)):
                if i == 0:
                    if prefixes[i] in p_trans_counts[pos]:
                        score_1 += log(p_trans_counts[pos][prefixes[i]])
                    else:
                        score_1 += log(p_trans_counts['STEM'][prefixes[i]])
                else:
                    if prefixes[i] in p_trans_counts[prefixes[i - 1]]:
                        score_1 += log(p_trans_counts[prefixes[i - 1]][prefixes[i]])
                    else:
                        score_1 -= float('inf')
                score_2 += log(p_counts[prefixes[i]])

        if len(segment[1]) > 0:

            suffixes = segment[1]

            for i in range(len(suffixes)):
                if i == 0:
                    if suffixes[i] in s_trans_counts[pos]:
                        score_1 += log(s_trans_counts[pos][suffixes[i]])
                    else:
                        score_1 += log(s_trans_counts['STEM'][suffixes[i]])
                else:
                    if suffixes[i] in s_trans_counts[suffixes[i - 1]]:
                        score_1 += log(s_trans_counts[suffixes[i - 1]][suffixes[i]])
                    else:
                        score_1 -= float('inf')
                score_2 += log(s_counts[suffixes[i]])

        ma_1.append(score_1)
        ma_2.append(score_2)

        y.append(0)

    if -float('inf') in ma_1:
        min_v = min(v for v in ma_1 if v != -float('inf'))
        ma_1 = [v if v != -float('inf') else min_v for v in ma_1]

    ma_ap = average_precision_score(y, ma_1)
    ma_auc = roc_auc_score(y, ma_1)

    print(average_precision_score(y, ma_1), average_precision_score(y, ma_2))

    with open('results/bm_%s.txt' % sr, 'w') as f:
        f.write("%s %s\n" % (ma_ap, ma_auc))


if __name__ == '__main__':
    main()
