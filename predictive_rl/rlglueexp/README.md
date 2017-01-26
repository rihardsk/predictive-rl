# rlglueexp
This is a module for RL-Glue experiment automation. To run
an experiment execute `config.py` and supply an experiment
configuration file. See `sample.config` for an example or
the [experiments directory](../../experiments/README.md)
for the experiment configuration files to reproduce the
results in my thesis.

A few different backends are supported for running the
experiments:
* IPython Parallel
* Spawning all the necessary processes locally
* (probabably broken) Jobman

For more info see the documentation within `config.py`.

## Running
### Locally
To start an experiment on the local machine from the project root run
```bash
python -m predictive_rl.rlglueexp.config -b local experiments/short.config
```
This will start a process that executes the experiments specified in the
`experiments/short.config` one by one.

### On an IPython cluster 
To run the experiment in multiple parallel instances using IPython
Parallel first you need to start an IPython cluster from the project
root
```bash
# start an IPython cluster with 4 workers
ipcluster start -n4
```
Then you may launch the experiment using the `ipython` backend
```bash
python -m predictive_rl.rlglueexp.config -b ipython experiments/short.config
```
All of the experiments specified in `experiments/short.config` will be
executed in parallel on the 4 cluster workers.

This can also be configured to be run on remote workers. For more info
see the [IPython Parallel documentation](http://ipyparallel.readthedocs.io/).
