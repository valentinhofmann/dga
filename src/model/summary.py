import numpy as np

sr_list = ['CFB', 'nba', 'nfl', 'gaming', 'leagueoflegends', 'movies', 'politics', 'science', 'technology']

with open('results/summary.txt', 'w') as outfile:
    ap_list = [[] for i in range(11)]
    auc_list = [[] for i in range(11)]
    for sr in sr_list:
        with open('results/results_{}.txt'.format(sr), 'r') as f:
            for i, line in enumerate(f):
                ap_list[i].append(float(line.strip().split()[0]))
                auc_list[i].append(float(line.strip().split()[2]))
    models = ['DGA+', 'DGA', 'CM', 'NC+', 'NC', 'JS', 'BM']
    for aps, aucs, model in zip(ap_list, auc_list, models):
        if model == 'PA':
            continue
        string = model + '&' + '&'.join('{:.3f}&{:.3f}'.format(ap, auc) for ap, auc in zip(aps, aucs)) + '&{:.3f}$\pm${:.3f}&{:.3f}$\pm${:.3f}\\\\\n'.format(
            np.mean(aps), np.std(aps),  np.mean(aucs), np.std(aucs))
        string = '&'.join(t.lstrip('0') for t in string.split('&'))
        outfile.write(string)
        if model in {'DGA', 'JS', 'NC', 'CM'}:
            outfile.write('\midrule\n')
