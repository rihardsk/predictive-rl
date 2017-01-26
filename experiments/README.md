# Experiment configuration files
This directory contains the configuration files used to
reproduce the results in my thesis. See the
[rlglueexp](../predictive_rl/rlglueexp/README.md) module
for information on how to run them.

The files contain experiment specifications for 4 different
reinforcement learning agents each run in 3 different RL
environments. Every agent and environment combination is
run 10 times. Experiments are grouped by environment,
namely:
* [mean_acrobot.config](mean_acrobot.config) for the
acrobot environment;
* [mean_car.config](mean_car.config) for the mountain-car
environment;
* [mean_pole.config](mean_pole.config) fo the cart-pole
environment.

The reward obtained during each of the experiment instances
is logged in a file placed in the respective instances
folder. To plot the reward curves obtained by each of the
agents use the [analyze.py](../predictive_rl/analyze.py)
script.
