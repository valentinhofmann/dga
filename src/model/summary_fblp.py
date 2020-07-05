import numpy as np

sr_list = ['CFB', 'nba', 'nfl', 'gaming', 'leagueoflegends', 'movies', 'politics', 'science', 'technology']

with open ('summary_fblp.txt', 'w') as outfile:
    ap_list = [[] for i in range(3)]
    auc_list = [[] for i in range(3)]
    for sr in sr_list:
        with open('results/pa_cn_aa_{}.txt'.format(sr), 'r') as f:
            for i, line in enumerate(f):
                ap_list[i].append(float(line.strip().split()[0]))
                auc_list[i].append(float(line.strip().split()[1]))
    models = ['PA', 'CN', 'AA']
    for aps, aucs, model in zip(ap_list, auc_list, models):
        outfile.write(model + '&' + '&'.join('{:04.3f}&{:04.3f}'.format(ap, auc) for ap, auc in zip(aps, aucs)) + '&{:04.3f}$\pm${:04.3f}&{:04.3f}$\pm${:04.3f}\\\\\n'.format(
            np.mean(aps), np.std(aps),  np.mean(aucs), np.std(aucs)))
