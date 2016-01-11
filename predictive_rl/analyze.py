import os
import itertools as it
from argparse import ArgumentParser
import re
from matplotlib import pyplot as plt
import pandas as pd
# pd.options.display.mpl_style = 'default'
plt.style.use('ggplot')


def diverged(expdir):
    expfile = None
    expre = re.compile(r"^experiment_\d+\.out$")
    for name in os.listdir(expdir):
        if os.path.isfile(os.path.join(expdir, name)) and expre.match(name):
            expfile = os.path.join(expdir, name)
            break
    if expfile is None:
        return None
    divergedre = re.compile("The agent has diverged")
    with open(expfile) as f:
        lines = f.readlines()
        linesfromend = 4
        for i in range(linesfromend):
            if len(lines) > i:
                if divergedre.search(lines[-i - 1]):
                    return True
            else:
                break
    return False


def savemean(basedir, plot=False):
    count = 0

    for dir in it.islice(os.walk(basedir), 1, None):
        if diverged(dir[0]):
            print("Diverged: " + dir[0])
            continue
        resultsfile = os.path.join(dir[0], 'onresults.csv')
        if not os.path.isfile(resultsfile):
            continue
        csv = pd.read_csv(resultsfile)
        if count == 0:
            csvsum = csv
        else:
            csvsum += csv
        count += 1
    savepath = os.path.join(basedir, 'onresults.mean.csv')
    meancsv = csvsum / count
    meancsv.to_csv(savepath, index=False)
    if plot:
        plotcount = len(plot)
        for i, toplot in enumerate(plot):
            runaverages = False
            if isinstance(toplot, dict):
                runaverages = toplot.get('runaverages', False)
                toplot = toplot.get('plot')
            plt.subplot(1, plotcount, i + 1)
            if runaverages:
                pd.expanding_mean(meancsv[toplot]).plot()
            else:
                meancsv[toplot].plot()
        _setplotlabels(plot)
        plt.show()
    return savepath


def plotall(basedir, plot=None):
    dirnames = [name for name in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, name))]
    shortnames = getshortnames(dirnames)
    plotcount = len(plot)
    for dirname, shortname in zip(dirnames, shortnames):
        resultspath = os.path.join(basedir, dirname)
        hasdiverged = False
        if diverged(resultspath):
            hasdiverged = True
        resultsfile = os.path.join(resultspath, 'onresults.csv')
        if not os.path.isfile(resultsfile):
            continue
        csv = pd.read_csv(resultsfile)
        if isinstance(plot, list):
            for i, toplot in enumerate(plot):
                runaverages = False
                if isinstance(toplot, dict):
                    runaverages = toplot.get('runaverages', False)
                    toplot = toplot.get('plot')
                label = shortname + (" (diverged)" if hasdiverged else "")
                plt.subplot(1, plotcount, i + 1)
                if runaverages:
                    pd.expanding_mean(csv[toplot]).plot(label=label, legend=True)
                else:
                    csv[toplot].plot(label=label, legend=True)
        else:
            csv.plot(subplots=True)
    _setplotlabels(plot)
    plt.show()


def _setplotlabels(plot):
    plotcount = len(plot)
    for i, toplot in enumerate(plot):
        sub = plt.subplot(1, plotcount, i + 1)
        xlabel = toplot.get('xlabel', '')
        ylabel = toplot.get('ylabel', '')
        title = toplot.get('title', '')
        sub.set_xlabel(xlabel)
        sub.set_ylabel(ylabel)
        sub.set_title(title)
        # plotname = toplot.get('plot')


def getshortnames(names):
    shortre = re.compile(r"^(.*)_\d{2}(?:-\d{2}){3}_(.*)$")  # this is the format in which ExperimenterAgent names
    # its results folder
    shortnames = [prefix + ".." + appendix for prefix, appendix in (shortre.match(name).groups() for name in names)]
    if len(shortnames) == len(set(shortnames)):
        return shortnames
    return names


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
    print("with {0}".format(os.path.join(maxdir, maxfile)))


def main():
    parser = ArgumentParser(description='A tool to analyze RL experiment results.')
    parser.add_argument('-s', '--subdirs', action='store_true',
                        help='If set, search all of the subdirectories of DIRECTORY for experiment results.')
    parser.add_argument('-m', '--mean', action='store_true',
                        help='If set, get mean results of all experiment results in DIRECTORY.'
                             'The files should have equal line count.'
                             'Can be used only together with --subdirs.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='If set, plot the mean results of all experiment results in DIRECTORY.'
                             'Can be used only together with --mean.')
    parser.add_argument('-P', '--plotall', action='store_true',
                        help='If set, plot the results of all experiment results in DIRECTORY.'
                             'Can be used only together with --mean.')
    parser.add_argument('directory', help='The directory to look for experiment results.')
    args = parser.parse_args()
    if args.mean and not args.subdirs:
        parser.error('--mean can be used only together with --subdirs.')
    if args.plot and not args.mean:
        parser.error('--plot can be used only together with --mean.')
    if args.plotall and not args.subdirs:
        parser.error('--plotall can be used only together with --subdirs.')
    printbest(args.directory, args.subdirs)
    plotcolumns = [{'plot': 'total_reward', 'xlabel': 'Epoch',
                    'ylabel': 'Total reward', 'title': 'Total reward per epoch'},
                   {'plot': 'mean_reward', 'xlabel': 'Epoch', 'ylabel': 'Running average reward',
                    'runaverages': True, 'title': 'Running average reward in epoch'}]
    if args.mean:
        savemean(args.directory, plotcolumns if args.plot else False)
    if args.plotall:
        plotall(args.directory, plotcolumns)

if __name__ == '__main__':
    main()
