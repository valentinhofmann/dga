import argparse
import sys


def main():

    sys.stdout.flush()

    parser = argparse.ArgumentParser()

    parser.add_argument('--sr', default=None, type=str, required=True, help='Subreddit.')

    args = parser.parse_args()

    sr = args.sr

    print('Subreddit: {}'.format(sr))

    with open('results/results_%s.txt' % sr, 'w') as outfile:

        for model in ['dgaplus', 'dga']:
            with open('results/%s_%s.txt' % (model, sr), 'r') as f:
                for line in f:
                    [ap, auc] = line.strip().split()
                    outfile.write('{:04.3f} & {:04.3f} &\n'.format(float(ap), float(auc)))
                    break

        with open('results/cm_%s.txt' % sr, 'r') as f:
            for line in f:
                [ap, auc] = line.strip().split()
                outfile.write('{:04.3f} & {:04.3f} &\n'.format(float(ap), float(auc)))
                break

        for model in ['ncplus', 'nc']:
            with open('results/%s_%s.txt' % (model, sr), 'r') as f:
                for line in f:
                    [ap, auc] = line.strip().split()
                    outfile.write('{:04.3f} & {:04.3f} &\n'.format(float(ap), float(auc)))
                    break

        with open('results/js_%s.txt' % sr, 'r') as f:
            for line in f:
                [ap, auc] = line.strip().split()
                outfile.write('{:04.3f} & {:04.3f} &\n'.format(float(ap), float(auc)))
                break

        with open('results/bm_%s.txt' % sr, 'r') as f:
            for line in f:
                [ap, auc] = line.strip().split()
                outfile.write('{:04.3f} & {:04.3f} &\n'.format(float(ap), float(auc)))
                break


if __name__ == '__main__':
    main()
