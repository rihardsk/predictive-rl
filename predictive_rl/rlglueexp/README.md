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

