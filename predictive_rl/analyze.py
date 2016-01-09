import pandas as pd
import os
import itertools as it
from argparse import ArgumentParser


def savemean(basedir):
    count = 0

    for dir in it.islice(os.walk(basedir), 1, None):
        resultsfile = os.path.join(dir[0], 'onresults.csv')
        csv = pd.read_csv(resultsfile)
        if count == 0:
            csvsum = csv
        else:
            csvsum += csv
        count += 1
    savepath = os.path.join(basedir, 'onresults.mean.csv')
    (csvsum / count).to_csv(savepath, index=False)


def printbest(basedir, subdirs=False):
    maxdir = "none"
    maxmeanreward = float("-inf")
    maxind = -1
    if subdirs:
        for dir in it.islice(os.walk(basedir), 1, None):
            resultsfile = os.path.join(dir[0], 'onresults.csv')
            if not os.path.isfile(resultsfile):
                continue
            try:
                csv = pd.read_csv(resultsfile)
            except:
                continue
            greatestind = csv.mean_reward.argmax()
            greatestval = csv.mean_reward[greatestind]
            if greatestval > maxmeanreward:
                maxdir = dir[0]
                maxmeanreward = greatestval
                maxind = greatestind
    else:
        resultsfile = os.path.join(basedir, 'onresults.csv')
        csv = pd.read_csv(resultsfile)
        maxdir = basedir
        maxind = csv.mean_reward.argmax()
        maxmeanreward = csv.mean_reward[maxind]
    print("Best mean reward of {0}".format(maxmeanreward))
    print("achieved in epoch {0}".format(maxind))
    maxfile = "network_file_{0}.pkl".format(maxind)
    print("with {0}".format(os.path.join(basedir, maxdir, maxfile)))


def main():
    parser = ArgumentParser(description='A tool to analyze RL experiment results.')
    parser.add_argument('-s', '--subdirs', action='store_true',
                        help='If set, search all of the subdirectories of DIRECTORY for experiment results.')
    parser.add_argument('directory', help='The directory to look for experiment results.')
    args = parser.parse_args()
    printbest(args.directory, args.subdirs)

if __name__ == '__main__':
    main()
