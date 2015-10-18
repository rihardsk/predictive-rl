__author__ = 'rihards'

import random
import sys
import copy
import pickle
import cPickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
#from pylearn2.models import mlp
import mlp
import time
import nn
import layers
import numpy as np
import data_set
import argparse
import os
from experimenter_agent import ExperimenterAgent


class CaclaAgentExperimenter(ExperimenterAgent):
    randGenerator = np.random

    def __init__(self):
        """
        Mostly just read command line arguments here. We do this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        super(CaclaAgentExperimenter, self).__init__()

        # Handle command line argument:
        parser = argparse.ArgumentParser(description='Neural rl agent.')
        parser.add_argument('--action_learning_rate', type=float, default=.01,
                            help='Learning rate')
        parser.add_argument('--value_learning_rate', type=float, default=.01,
                            help='Learning rate')
        parser.add_argument('--action_stdev', type=str, default=0.1,
                            help='Action space exploration standard deviation for Gaussian distribution.')
        parser.add_argument('--dir', type=str, default="experiments_minimal",
                            help='Directory to save results')
        parser.add_argument('--nn_action_file', type=str, default=None,
                            help='Pickle file containing trained action net.')
        parser.add_argument('--nn_value_file', type=str, default=None,
                            help='Pickle file containing trained value net.')
        parser.add_argument('--collect_rewards', type=bool, default=True,
                            help='If set to true, testing episode mean rewards will be saved to a file.')
        # Create instance variables directy from the arguments:
        # parser.parse_known_args(namespace=self)

        args = parser.parse_args()
        self.action_learning_rate = args.action_learning_rate
        self.value_learning_rate = args.value_learning_rate
        self.exp_dir = args.dir
        self.nn_action_file = args.nn_action_file
        self.nn_value_file = args.nn_value_file
        self.action_stdev = args.action_stdev
        self.collect_rewards = args.collect_rewards

    def agent_init(self, taskSpecification):
        """
        This function is called once at the beginning of an experiment.

        :param taskSpecification: A string defining the task.  This string
        is decoded using TaskSpecVRLGLUE3.TaskSpecParser
        :return:
        """

        # DO SOME SANITY CHECKING ON THE TASKSPEC
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecification)

        if TaskSpec.valid:

            assert ((len(TaskSpec.getIntObservations()) == 0) !=
                    (len(TaskSpec.getDoubleObservations()) == 0)), \
                "expecting continous or discrete observations.  Not both."
            assert not TaskSpec.isSpecial(TaskSpec.getDoubleActions()[0][0]), \
                " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getDoubleActions()[0][1]), \
                " expecting max action to be a number not a special value"
            #self.num_actions = TaskSpec.getIntActions()[0][1]+1
        else:
            print "INVALID TASK SPEC"

        self.observation_ranges = TaskSpec.getDoubleObservations()  # TODO: take care of int observations
        self.observation_size = len(self.observation_ranges)

        self.action_ranges = TaskSpec.getDoubleActions()
        self.action_size = len(self.action_ranges)

        if self.nn_action_file is None:
            self.action_network = self._init_action_network(self.observation_size,
                                                            self.action_size,
                                                            minibatch_size=1)
        else:
            handle = open(self.nn_action_file, 'r')
            self.action_network = cPickle.load(handle)

        if self.nn_value_file is None:
            self.value_network = self._init_value_network(self.observation_size,
                                                          1,
                                                          minibatch_size=1)
        else:
            handle = open(self.nn_value_file, 'r')
            self.value_network = cPickle.load(handle)


        self.discount = TaskSpec.getDiscountFactor()

        self.action_ranges = np.asmatrix(self.action_ranges)
        self.observation_ranges = np.asmatrix(self.observation_ranges)

    def _init_action_network(self, input_dims, output_dims, minibatch_size=32):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        scale_factor = 1
        layer1 = layers.FlatInputLayer(minibatch_size, input_dims, np.asarray(self.observation_ranges, dtype='float32'), scale_factor)
        layer2 = layers.DenseLayer(layer1, 15, 0.1, 0, layers.tanh)
        layer3 = layers.DenseLayer(layer2, output_dims, 0.1, 0, layers.identity)
        layer4 = layers.OutputLayer(layer3)
        return nn.NN([layer1, layer2, layer3, layer4], batch_size=minibatch_size, learning_rate=self.action_learning_rate)

    def _init_value_network(self, input_dims, output_dims, minibatch_size=32):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        scale_factor = 2
        layer1 = layers.FlatInputLayer(minibatch_size, input_dims, np.asarray(self.observation_ranges, dtype='float32'), scale_factor)
        layer2 = layers.DenseLayer(layer1, 15, 0.1, 0, layers.tanh)
        layer3 = layers.DenseLayer(layer2, output_dims, 0.1, 0, layers.identity)
        layer4 = layers.OutputLayer(layer3)
        return nn.NN([layer1, layer2, layer3, layer4], batch_size=minibatch_size, learning_rate=self.value_learning_rate)

    def agent_start(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - An observation of type rlglue.types.Observation

        Returns:
           An action of type rlglue.types.Action
        """

        # this_int_action = self.randGenerator.randint(0, self.num_actions-1)
        observation_matrix = np.asmatrix(observation.doubleArray, dtype='float32')
        actions = self.action_network.fprop(observation_matrix)
        return_action = Action()
        return_action.doubleArray = actions

        self.last_action = copy.deepcopy(actions)

        self.last_observation = observation.doubleArray

        return return_action

    def _choose_action(self, cur_observation, reward, action_stdev=None):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """
        self.training_sample = (np.asmatrix(self.last_observation, dtype='float32'),
                            np.asmatrix(self.last_action, dtype='float32'), reward,
                            np.asmatrix(cur_observation, dtype='float32'), False)

        double_action = self.action_network.fprop(np.asmatrix(cur_observation, dtype='float32'))

        # in order for the agent to learn we need some exploration
        gaussian = 0 if action_stdev is None or action_stdev == 0 else self.randGenerator.normal(0, action_stdev, len(double_action))
        exploration = gaussian * (self.action_ranges[:, 1] - self.action_ranges[:, 0])
        double_action += exploration
        return np.clip(double_action, self.action_ranges[:, 0], self.action_ranges[:, 1])

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        state, action, reward, next_state, terminal = self.training_sample
        value = self.value_network.fprop(state)
        target_value = reward + np.multiply(self.discount * self.value_network.fprop(next_state), not terminal)

        mask = target_value > value

        if mask[0, 0]:
            return self.action_network.train_model(state, action)
        else:
            return None

    def exp_step(self, reward, observation, is_testing):
        return_action = Action()
        cur_observation = observation.doubleArray
        double_action = self._choose_action(cur_observation, np.clip(reward, -1, 1), self.action_stdev)
        loss = None
        if not is_testing:
            loss = self._do_training()
        self.last_action = copy.deepcopy(double_action)
        self.last_observation = cur_observation
        return_action.doubleArray = [double_action]
        return return_action if is_testing else (return_action, loss)


    def exp_end(self, reward, is_testing):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.

        Returns:
            None
        """

        if reward is not None:
            if not is_testing:
                # Store the latest sample.
                self.training_sample = (np.asmatrix(self.last_observation, dtype='float32'),
                                np.asmatrix(self.last_action, dtype='float32'), np.clip(reward, -1, 1),
                                np.asmatrix(np.zeros_like(self.last_observation, dtype='float32')),
                                True)

                loss = self._do_training()
                return loss

    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """
        pass

    def _open_learning_file(self):
        try:
            os.stat(self.exp_dir)
        except:
            os.makedirs(self.exp_dir)

        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,action_learning_rate,value_learning_rate\n')

    def save_agent(self, epoch):
            action_net_file = open(os.path.join(self.exp_dir, 'network_action_file_' + str(epoch) +
                                   '.pkl'), 'w')
            cPickle.dump(self.action_network, action_net_file, -1)
            action_net_file.close()
            value_net_file = open(os.path.join(self.exp_dir + 'network_value_file_' + str(epoch) +
                                  '.pkl'), 'w')
            cPickle.dump(self.value_network, value_net_file, -1)
            value_net_file.close()


def main():
    AgentLoader.loadAgent(CaclaAgentExperimenter())


if __name__ == "__main__":
    main()
