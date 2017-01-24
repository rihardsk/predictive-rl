# rlglueexp
This is a module for RL-Glue experiment automation. To run
an experiment execute `config.py` and supply an experiment
configuration file. See `sample.config` for an example.

A few different backends are supported for running the
experiments:
* IPython Parallel
* Spawning all the necessary processes locally
* (probabably broken) Jobman

For more info see the documentation within `config.py`.

