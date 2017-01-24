# predictive-rl
Everything to reproduce reinforcement learning experiments done
in my [Master's thesis](https://github.com/rihardsk/rlthesis).
For now you'll have to find your way around by yourself. In
time i'll try to come up with some documentation.

## Installation
Python 2.7 is used. The following packages are required:
* [Theano](http://deeplearning.net/software/theano/)
* [RL-Glue](http://glue.rl-community.org/)
* [configobj](https://github.com/DiffSK/configobj)
* (optional) [IPython Parallel](http://ipyparallel.readthedocs.io/)
for running the experiments remotelly and/or in parallel.

To run the environments you will also have to set up
[rihardsk/predictive-rl_envs](https://github.com/rihardsk/predictive-rl_envs)

## Run
See [rlglueexp module](predictive_rl/rlglueexp/README.md) for information on
automating the experiment process.

See the [experiments directory](experiments/README.md)
for the experiment configuration files to reproduce the
results in my thesis.
