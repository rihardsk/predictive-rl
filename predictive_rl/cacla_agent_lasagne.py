__author__ = 'rihards'

import copy
import cPickle
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3
import numpy as np
import argparse
import os
from experimenter_agent import ExperimenterAgent
import theano
import theano.tensor as T
from lasagne import layers
from lasagne import updates
from lasagne import nonlinearities
from lasagne import objectives

floatX = theano.config.floatX


class Mock(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CaclaAgentLasagne(ExperimenterAgent):
    randGenerator = np.random

    def __init__(self):
        """
        Mostly just read command line arguments here. We do this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        super(CaclaAgentLasagne, self).__init__()

        # Handle command line argument:
        parser = argparse.ArgumentParser(description='Neural rl agent.')
        self._add_parse_args(parser)

        # Create instance variables directy from the arguments:
        # parser.parse_known_args(namespace=self)

        args = parser.parse_args()
        self._get_parsed_args(args)

    def _add_parse_args(self, parser):
        parser.add_argument('--action_learning_rate', type=float, default=.01,
                            help='Learning rate')
        parser.add_argument('--value_learning_rate', type=float, default=.01,
                            help='Learning rate')
        parser.add_argument('--action_stdev', type=float, default=0.1,
                            help='Action space exploration standard deviation for Gaussian distribution. '
                                 'Applied to action range.')
        parser.add_argument('--noise_stdev', type=float, default=0.01,
                            help='Action space exploration standard deviation for Gaussian distribution. '
                                 'Applied to the actions magnitude.')
        parser.add_argument('--dir', type=str, default="experiments_minimal",
                            help='Directory to save results')
        parser.add_argument('--nn_action_file', type=str, default=None,
                            help='Pickle file containing trained action net.')
        parser.add_argument('--nn_value_file', type=str, default=None,
                            help='Pickle file containing trained value net.')
        parser.add_argument('--collect_rewards', type=bool, default=True,
                            help='If set to true, testing episode mean rewards will be saved to a file.')

    def _get_parsed_args(self, args):
        self.action_learning_rate = args.action_learning_rate
        self.value_learning_rate = args.value_learning_rate
        self.exp_dir = args.dir
        self.nn_action_file = args.nn_action_file
        self.nn_value_file = args.nn_value_file
        self.action_stdev = args.action_stdev
        self.noise_stdev = args.noise_stdev
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

        self._init_action_network()
        self._init_value_network()

        self.discount = TaskSpec.getDiscountFactor()

        self.action_ranges = np.asmatrix(self.action_ranges, dtype=floatX)
        self.observation_ranges = np.asmatrix(self.observation_ranges, dtype=floatX)

    def _init_value_network(self):
        if self.nn_value_file is None:
            self.value_network = self._create_nnet(self.observation_size,
                                                   1,
                                                   self.value_learning_rate,
                                                   batch_size=1)
        else:
            handle = open(self.nn_value_file, 'r')
            self.value_network = cPickle.load(handle)

    def _init_action_network(self):
        if self.nn_action_file is None:
            self.action_network = self._create_nnet(self.observation_size,
                                                    self.action_size,
                                                    self.action_learning_rate,
                                                    batch_size=1)
        else:
            handle = open(self.nn_action_file, 'r')
            self.action_network = cPickle.load(handle)

    @staticmethod
    def _create_nnet(input_dims, output_dims, learning_rate, num_hidden_units=15, batch_size=32, max_train_epochs=1,
                     hidden_nonlinearity=nonlinearities.rectify, output_nonlinearity=None, update_method=updates.sgd):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        nnlayers = []
        nnlayers.append(layers.InputLayer(shape=(None, input_dims)))
        nnlayers.append(layers.DenseLayer(nnlayers[-1], num_hidden_units, nonlinearity=hidden_nonlinearity))
        nnlayers.append(layers.DenseLayer(nnlayers[-1], output_dims, nonlinearity=output_nonlinearity))

        prediction = layers.get_output(nnlayers[-1])

        input_var = nnlayers[0].input_var
        target = T.matrix(name="target", dtype=floatX)

        loss = objectives.squared_error(prediction, target).mean()

        params = layers.get_all_params(nnlayers[-1], trainable=True)

        updates = update_method(loss, params, learning_rate)

        fit = theano.function([input_var, target], loss, updates=updates)

        predict = theano.function([input_var], prediction)

        nnet = Mock(
            fit=fit,
            predict=predict,
        )
        return nnet

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
        actions = self.action_network.predict(observation_matrix)

        return_action = Action()
        return_action.doubleArray = actions

        self.last_action = copy.deepcopy(actions)
        self.last_state = np.asmatrix(observation.doubleArray, dtype=floatX)

        return return_action

    def _choose_action(self, cur_observation, action_stdev=None, noise_stdev=None):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """
        double_action = self.action_network.predict(np.asmatrix(cur_observation, dtype='float32'))

        # in order for the agent to learn we need some exploration
        gaussian1 = 0 if noise_stdev is None or noise_stdev == 0 else self.randGenerator.normal(0, noise_stdev, len(double_action))
        gaussian2 = 0 if action_stdev is None or action_stdev == 0 else self.randGenerator.normal(0, action_stdev, len(double_action))
        exploration = gaussian1 * (self.action_ranges[:, 1] - self.action_ranges[:, 0]) + gaussian2 * double_action
        # double_action += exploration
        double_action += gaussian2
        return np.asmatrix(np.clip(double_action, self.action_ranges[:, 0], self.action_ranges[:, 1]), dtype=floatX)

    def _do_training(self, reward, observation, action, terminal):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        cur_state = np.asmatrix(observation, dtype=floatX)
        last_state_value = self.value_network.predict(self.last_state)
        if terminal:
            target_value = reward
        else:
            target_value = reward + self.discount * self.value_network.predict(cur_state)
        self.value_network.fit(self.last_state, target_value)

        # this reflects whether the value of the last_state has risen after the value_network update above
        mask = target_value > last_state_value

        if mask[0, 0]:
            loss = self.action_network.fit(self.last_state, self.last_action)
        else:
            loss = None
        self.last_state = cur_state
        self.last_action = action
        return loss

    @staticmethod
    def _scale_inputs(inputs, ranges, target_amplitude=1):
        minranges = ranges[:, 0].T
        maxranges = ranges[:, 1].T
        scale = target_amplitude
        scaled = (inputs - minranges) / (maxranges - minranges) * 2 * scale - scale
        return np.asmatrix(scaled, dtype=floatX)

    def exp_step(self, reward, observation, is_testing):
        return_action = Action()
        cur_observation = self._scale_inputs(observation.doubleArray, self.observation_ranges)
        double_action = self._choose_action(cur_observation, self.action_stdev, self.noise_stdev)
        loss = None
        if not is_testing:
            loss = self._do_training(np.asmatrix(reward, dtype=floatX), cur_observation, double_action, False)
        return_action.doubleArray = [copy.deepcopy(double_action)]
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
                loss = self._do_training(np.asmatrix(reward, dtype=floatX),
                                         # doesn't really matter what we pass as state and action
                                         np.zeros_like(self.last_state, dtype=floatX),
                                         np.zeros_like(self.last_action, dtype=floatX),
                                         True)
                return loss

    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """
        pass

    def save_agent(self, epoch):
            action_net_file = open(os.path.join(self.exp_dir, 'network_action_file_' + str(epoch) +
                                   '.pkl'), 'w')
            cPickle.dump(self.action_network, action_net_file, -1)
            action_net_file.close()
            value_net_file = open(os.path.join(self.exp_dir, 'network_value_file_' + str(epoch) +
                                  '.pkl'), 'w')
            cPickle.dump(self.value_network, value_net_file, -1)
            value_net_file.close()


def main():
    AgentLoader.loadAgent(CaclaAgentLasagne())


if __name__ == "__main__":
    main()
