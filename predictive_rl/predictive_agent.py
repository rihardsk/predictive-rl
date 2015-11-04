__author__ = 'rihards'

from experimenter_agent import ExperimenterAgent
import argparse
import copy
import cPickle
import os
from rlglue.utils import TaskSpecVRLGLUE3
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
import numpy as np
import theano
from lasagne import layers
from lasagne import updates
from lasagne import nonlinearities
from lasagne import objectives
import nolearn
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import theano.tensor as T

floatX = theano.config.floatX


class PredictiveAgent(ExperimenterAgent):
    randGenerator = np.random

    def __init__(self):
        """
        Mostly just read command line arguments here. We do this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        super(PredictiveAgent, self).__init__()

        # Handle command line argument:
        parser = argparse.ArgumentParser(description='Neural rl agent.')
        self._add_parse_args(parser)

        # Create instance variables directy from the arguments:
        # parser.parse_known_args(namespace=self)

        args = parser.parse_args()
        self._get_parsed_args(args)

    def _add_parse_args(self, parser):
        parser.add_argument('--learning_rate', type=float, default=.01,
                            help='Learning rate')
        parser.add_argument('--action_stdev', type=float, default=0.1,
                            help='Action space exploration standard deviation for Gaussian distribution. '
                                 'Applied to action range.')
        # parser.add_argument('--noise_stdev', type=float, default=0.01,
        #                     help='Action space exploration standard deviation for Gaussian distribution. '
        #                          'Applied to the actions magnitude.')
        parser.add_argument('--dir', type=str, default="experiments_replay",
                            help='Directory to save results')
        parser.add_argument('--nn_file', type=str, default=None,
                            help='Pickle file containing trained neural net.')
        parser.add_argument('--collect_rewards', type=bool, default=True,
                            help='If set to true, testing episode mean rewards will be saved to a file.')

    def _get_parsed_args(self, args):
        self.learning_rate = args.learning_rate
        self.exp_dir = args.dir
        self.nn_file = args.nn_file
        self.action_stdev = args.action_stdev
        # self.noise_stdev = args.noise_stdev
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

        self._init_network()

        self.discount = TaskSpec.getDiscountFactor()

        self.action_ranges = np.asmatrix(self.action_ranges, dtype=floatX)
        self.observation_ranges = np.asmatrix(self.observation_ranges, dtype=floatX)
        self.testing = False

    def _init_network(self):
        if self.nn_file is None:
            self.nnet = self.create_nnet(self.observation_size,
                                         self.action_size,
                                         self.observation_size + 1,
                                         self.learning_rate,
                                         batch_size=1)
        else:
            handle = open(self.nn_file, 'r')
            self.nnet = cPickle.load(handle)

    @staticmethod
    def create_nnet(input_dims, action_dims, observation_dims, learning_rate, num_hidden_units=20, batch_size=32,
                    max_train_epochs=1, hidden_nonlinearity=nonlinearities.rectify, output_nonlinearity=None,
                    update_method=updates.sgd):
        commonlayers = []
        commonlayers.append(layers.InputLayer(shape=(None, input_dims)))
        commonlayers.append(layers.DenseLayer(commonlayers[-1], num_hidden_units, nonlinearity=hidden_nonlinearity))
        actionlayers = [layers.DenseLayer(commonlayers[-1], action_dims, nonlinearity=output_nonlinearity)]
        observlayers = [layers.DenseLayer(commonlayers[-1], observation_dims, nonlinearity=output_nonlinearity)]
        concatlayers = [layers.ConcatLayer([actionlayers[-1], observlayers[-1]])]
        action_prediction = layers.get_output(actionlayers[-1])
        observ_prediction = layers.get_output(observlayers[-1])
        concat_prediction = layers.get_output(concatlayers[-1])
        input_var = commonlayers[0].input_var
        # action_target_type = T.TensorType(floatX, [False] * action_dims)
        # observ_target_type = T.TensorType(floatX, [False] * observation_dims)
        # concat_target_type = T.TensorType(floatX, [False] * (action_dims + observation_dims))
        # action_target = action_target_type()
        # observ_target = observ_target_type()
        # concat_target = concat_target_type()
        action_target = T.matrix(name="action_target", dtype=floatX)
        observ_target = T.matrix(name="observ_target", dtype=floatX)
        concat_target = T.matrix(name="concat_target", dtype=floatX)
        action_loss = objectives.squared_error(action_prediction, action_target).mean()
        observ_loss = objectives.squared_error(observ_prediction, observ_target).mean()
        concat_loss = objectives.squared_error(concat_prediction, concat_target).mean()
        action_params = layers.get_all_params(actionlayers[-1], trainable=True)
        observ_params = layers.get_all_params(observlayers[-1], trainable=True)
        concat_params = layers.get_all_params(concatlayers[-1], trainable=True)
        action_updates = update_method(action_loss, action_params, learning_rate)
        observ_updates = update_method(observ_loss, observ_params, learning_rate)
        concat_updates = update_method(concat_loss, concat_params, learning_rate)

        fit_action = theano.function([input_var, action_target], action_loss, updates=action_updates)
        fit_observ = theano.function([input_var, observ_target], observ_loss, updates=observ_updates)
        fit_concat = theano.function([input_var, concat_target], concat_loss, updates=concat_updates)

        predict_action = theano.function([input_var], action_prediction)
        predict_observ = theano.function([input_var], observ_prediction)
        predict_concat = theano.function([input_var], concat_prediction)

        class Mock(object):
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        nnet = Mock(
            fit_action=fit_action,
            fit_observ=fit_observ,
            fit_both=fit_concat,
            predict_action=predict_action,
            predict_observ=predict_observ,
            predict_both=predict_concat,
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

        cur_observation = self._scale_inputs(observation.doubleArray, self.observation_ranges)
        pred_action, pred_observation, cur_observation_value = self._predict(cur_observation)
        double_action = self._explore(pred_action, self.action_stdev)

        return_action = Action()
        return_action.doubleArray = double_action

        self.last_state = cur_observation
        self.last_action = copy.deepcopy(double_action)
        self.last_original_action = pred_action
        self.last_state_value = cur_observation_value

        return return_action

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
        pred_action, pred_observation, cur_observation_value = self._predict(cur_observation)
        double_action = self._explore(pred_action, self.action_stdev)
        loss = None
        if not is_testing:
            loss = self._do_training(np.asmatrix(reward, dtype=floatX), cur_observation, double_action, False,
                                     cur_observation_value, pred_observation, pred_action)
        return_action.doubleArray = [copy.deepcopy(double_action)]
        return return_action if is_testing else (return_action, loss)

    def _predict(self, observation):
        pred_matrix = self.nnet.predict_both(observation)
        next_action = pred_matrix[:, 0:self.action_size]
        next_observation = pred_matrix[:, self.action_size:self.action_size + self.observation_size]
        observation_value = pred_matrix[:, self.action_size + self.observation_size]
        return next_action, next_observation, observation_value

    def _explore(self, action, stdev):
        gaussian = 0 if stdev is None or stdev == 0 else self.randGenerator.normal(0, stdev, len(action))
        double_action = action + gaussian
        return np.asmatrix(np.clip(double_action, self.action_ranges[:, 0], self.action_ranges[:, 1]), dtype=floatX)

    def _do_training(self, reward, observation, action, terminal, observation_value, pred_observation, pred_action):
        cur_state = np.asmatrix(observation, dtype=floatX)
        last_state_value = self.last_state_value
        if terminal:
            target_value = reward
        else:
            target_value = reward + self.discount * observation_value

        # this reflects whether the value of the last_state has risen after the value_network update above
        mask = target_value > last_state_value

        if mask[0, 0]:
            loss = self.nnet.fit_both(self.last_state, np.hstack((self.last_action, observation, target_value)))
        else:
            # TODO: should not train  on action at all here
            loss = self.nnet.fit_observ(self.last_state, np.hstack((observation, target_value)))
        self.last_state = cur_state
        self.last_action = action
        self.last_original_action = pred_action
        self.last_state_value = observation_value
        self.pred_observation = pred_observation
        return loss

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
                                         # TODO: not true any more
                                         # doesn't really matter what we pass as state and action
                                         np.zeros_like(self.last_state, dtype=floatX),
                                         np.zeros_like(self.last_action, dtype=floatX),
                                         True,
                                         np.zeros_like(self.last_state_value, dtype=floatX),
                                         np.zeros_like(self.last_state, dtype=floatX),
                                         np.zeros_like(self.last_original_action, dtype=floatX),
                                         )
                return loss

    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """
        pass

    def save_agent(self, epoch):
            net_file = open(os.path.join(self.exp_dir, 'network_file_' + str(epoch) +
                                   '.pkl'), 'w')
            cPickle.dump(self.nnet, net_file, -1)
            net_file.close()


def main():
    AgentLoader.loadAgent(PredictiveAgent())


if __name__ == "__main__":
    main()
