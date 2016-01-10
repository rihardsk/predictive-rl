from argsagent import ArgsAgent
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
from lasagne.regularization import l2, l1
from lasagne.regularization import regularize_layer_params
import theano.tensor as T
from lasagne.layers import DenseLayer

__author__ = 'rihards'

floatX = theano.config.floatX


class Mock(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PredictiveAgent(ArgsAgent):
    def __init__(self, **kwargs):
        """
        Mostly just read command line arguments here. We do this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        self.randGenerator = np.random.RandomState()
        super(PredictiveAgent, self).__init__(**kwargs)

    def _add_parse_args(self, parser):
        parser.add_argument('-r', '--learning_rate', type=float, default=.001,
                            help='Learning rate')
        parser.add_argument('-c', '--grad_clipping', type=float, default=None,
                            help='Gradient clipping range (symetric). None to disable.')
        parser.add_argument('-s', '--action_stdev', type=float, default=1,
                            help='Action space exploration standard deviation for Gaussian distribution. '
                                 'Applied to action range.')
        parser.add_argument('-l1', '--l1_weight', type=float, default=None,
                            help='L1 regularization weight.')
        parser.add_argument('-l2', '--l2_weight', type=float, default=None,
                            help='L2 regularization weight.')
        # parser.add_argument('--noise_stdev', type=float, default=0.01,
        #                     help='Action space exploration standard deviation for Gaussian distribution. '
        #                          'Applied to the actions magnitude.')
        parser.add_argument('-d', '--dir', type=str, default="experiments_replay",
                            help='Directory to save results')
        parser.add_argument('-f', '--nn_file', type=str, default=None,
                            help='Pickle file containing trained neural net.')
        parser.add_argument('-hs', '--nn_hidden_size', type=int, default=20,
                            help='Neural net\'s layer hidden size')
        parser.add_argument('-as', '--nn_hidden_action_size', type=int, default=None,
                            help='Neural net\'s hidden action layer size')
        parser.add_argument('-os', '--nn_hidden_observ_size', type=int, default=None,
                            help='Neural net\'s hidden observation layer size')
        parser.add_argument('-vs', '--nn_hidden_value_size', type=int, default=None,
                            help='Neural net\'s hidden value layer size')
        parser.add_argument('--collect_rewards', type=bool, default=True,
                            help='If set to true, testing episode mean rewards will be saved to a file.')
        parser.add_argument('--testing', type=bool, default=False,
                            help='Set true to disable learning and exploration.')
        parser.add_argument('--activation', type=str, default='rectify',
                            help='The hidden layer activation function')
        parser.add_argument('--scale_range', type=float, default=None,
                            help='The range to scale the inputs in when preprocessing.')

    def _get_parsed_args(self, args):
        self.learning_rate = args.learning_rate
        self.grad_clipping = args.grad_clipping
        self.l1_weight = args.l1_weight
        self.l2_weight = args.l2_weight
        self.exp_dir = args.dir
        self.nn_file = args.nn_file
        self.nn_hidden_size = args.nn_hidden_size
        self.nn_hidden_action_size = args.nn_hidden_action_size
        self.nn_hidden_observ_size = args.nn_hidden_observ_size
        self.nn_hidden_value_size = args.nn_hidden_value_size
        self.action_stdev = args.action_stdev
        # self.noise_stdev = args.noise_stdev
        self.collect_rewards = args.collect_rewards
        self.testing = args.testing
        self.scale_range = args.scale_range
        if args.activation == "rectify":
            self.activation = nonlinearities.rectify
        elif args.activation == "tanh":
            self.activation = nonlinearities.tanh
        elif args.activation == "sigmoid":
            self.activation = nonlinearities.sigmoid

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
        # self.testing = False

    def _init_network(self):
        if self.nn_file is None:
            self.nnet = self.create_nnet(input_dims=self.observation_size,
                                         action_dims=self.action_size,
                                         observation_dims=self.observation_size,
                                         value_dims=1,
                                         learning_rate=self.learning_rate,
                                         grad_clip=self.grad_clipping,
                                         l1_weight=self.l1_weight,
                                         l2_weight=self.l2_weight,
                                         num_hidden_units=self.nn_hidden_size,
                                         num_hidden_action_units=self.nn_hidden_action_size,
                                         num_hidden_observ_units=self.nn_hidden_observ_size,
                                         num_hidden_value_units=self.nn_hidden_value_size,
                                         hidden_nonlinearity=self.activation,
                                         batch_size=1)
        else:
            handle = open(self.nn_file, 'r')
            self.nnet = cPickle.load(handle)

    @staticmethod
    def create_nnet(input_dims, action_dims, observation_dims, value_dims, learning_rate, grad_clip=None, l1_weight=None, l2_weight=None,
                    num_hidden_units=20, num_hidden_action_units=None, num_hidden_observ_units=None, num_hidden_value_units=None,
                    batch_size=32, max_train_epochs=1, hidden_nonlinearity=nonlinearities.rectify,
                    output_nonlinearity=None, update_method=updates.sgd):
        commonlayers = []
        commonlayers.append(layers.InputLayer(shape=(None, input_dims)))
        commonlayers.append(DenseLayer(commonlayers[-1], num_hidden_units,
                                               nonlinearity=hidden_nonlinearity))
        if num_hidden_action_units is None:
            actionlayers = [DenseLayer(commonlayers[-1], action_dims,
                                               nonlinearity=output_nonlinearity)]
        else:
            actionlayers = [DenseLayer(commonlayers[-1], num_hidden_action_units,
                                               nonlinearity=output_nonlinearity)]
            actionlayers.append(DenseLayer(actionlayers[-1], action_dims,
                                                   nonlinearity=output_nonlinearity))
        if num_hidden_observ_units is None:
            observlayers = [DenseLayer(commonlayers[-1], observation_dims,
                                               nonlinearity=output_nonlinearity)]
        else:
            observlayers = [DenseLayer(commonlayers[-1], num_hidden_observ_units,
                                               nonlinearity=output_nonlinearity)]
            observlayers.append(DenseLayer(observlayers[-1], observation_dims, nonlinearity=output_nonlinearity))
        if num_hidden_value_units is None:
            dvaluelayers = [DenseLayer(commonlayers[-1], value_dims,
                                               nonlinearity=output_nonlinearity)]
        else:
            dvaluelayers = [DenseLayer(commonlayers[-1], num_hidden_value_units,
                                               nonlinearity=output_nonlinearity)]
            dvaluelayers.append(DenseLayer(dvaluelayers[-1], value_dims,
                                                   nonlinearity=output_nonlinearity))
        actvallayers = [layers.ConcatLayer([actionlayers[-1], dvaluelayers[-1]])]
        obsvallayers = [layers.ConcatLayer([observlayers[-1], dvaluelayers[-1]])]
        concatlayers = [layers.ConcatLayer([actionlayers[-1], observlayers[-1], dvaluelayers[-1]])]
        action_prediction = layers.get_output(actionlayers[-1])
        dvalue_prediction = layers.get_output(dvaluelayers[-1])
        actval_prediction = layers.get_output(actvallayers[-1])
        obsval_prediction = layers.get_output(obsvallayers[-1])
        concat_prediction = layers.get_output(concatlayers[-1])
        input_var = commonlayers[0].input_var
        action_target = T.matrix(name="action_target", dtype=floatX)
        dvalue_target = T.matrix(name="value_target", dtype=floatX)
        actval_target = T.matrix(name="actval_target", dtype=floatX)
        obsval_target = T.matrix(name="obsval_target", dtype=floatX)
        concat_target = T.matrix(name="concat_target", dtype=floatX)
        action_loss = objectives.squared_error(action_prediction, action_target).mean()
        obsval_loss = objectives.squared_error(obsval_prediction, obsval_target).mean()
        dvalue_loss = objectives.squared_error(dvalue_prediction, dvalue_target).mean()
        actval_loss = objectives.squared_error(actval_prediction, actval_target).mean()
        concat_loss = objectives.squared_error(concat_prediction, concat_target).mean()
        if l1_weight is not None:
            action_l1penalty = regularize_layer_params(commonlayers + actionlayers, l1) * l1_weight
            obsval_l1penalty = regularize_layer_params(commonlayers + observlayers + dvaluelayers, l1) * l1_weight
            dvalue_l1penalty = regularize_layer_params(commonlayers + dvaluelayers, l1) * l1_weight
            actval_l1penalty = regularize_layer_params(commonlayers + actionlayers + dvaluelayers, l1) * l1_weight
            concat_l1penalty = regularize_layer_params(commonlayers + actionlayers + observlayers + dvaluelayers, l1) * l1_weight
            action_loss += action_l1penalty
            obsval_loss += obsval_l1penalty
            dvalue_loss += dvalue_l1penalty
            actval_loss += actval_l1penalty
            concat_loss += concat_l1penalty
        if l2_weight is not None:
            action_l2penalty = regularize_layer_params(commonlayers + actionlayers, l2) * l2_weight
            obsval_l2penalty = regularize_layer_params(commonlayers + observlayers + dvaluelayers, l2) * l2_weight
            dvalue_l2penalty = regularize_layer_params(commonlayers + dvaluelayers, l2) * l2_weight
            actval_l2penalty = regularize_layer_params(commonlayers + actionlayers + dvaluelayers, l2) * l2_weight
            concat_l2penalty = regularize_layer_params(commonlayers + actionlayers + observlayers + dvaluelayers, l2) * l2_weight
            action_loss += action_l2penalty
            obsval_loss += obsval_l2penalty
            dvalue_loss += dvalue_l2penalty
            actval_loss += actval_l2penalty
            concat_loss += concat_l2penalty
        action_params = layers.get_all_params(actionlayers[-1], trainable=True)
        obsval_params = layers.get_all_params(obsvallayers[-1], trainable=True)
        dvalue_params = layers.get_all_params(dvaluelayers[-1], trainable=True)
        actval_params = layers.get_all_params(actvallayers[-1], trainable=True)
        concat_params = layers.get_all_params(concatlayers[-1], trainable=True)
        if grad_clip is not None:
            action_grads = theano.grad(action_loss, action_params)
            obsval_grads = theano.grad(obsval_loss, obsval_params)
            dvalue_grads = theano.grad(dvalue_loss, dvalue_params)
            actval_grads = theano.grad(actval_loss, actval_params)
            concat_grads = theano.grad(concat_loss, concat_params)
            action_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in action_grads]
            obsval_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in obsval_grads]
            dvalue_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in dvalue_grads]
            actval_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in actval_grads]
            concat_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in concat_grads]
            action_updates = update_method(action_grads, action_params, learning_rate)
            obsval_updates = update_method(obsval_grads, obsval_params, learning_rate)
            dvalue_updates = update_method(dvalue_grads, dvalue_params, learning_rate)
            actval_updates = update_method(actval_grads, actval_params, learning_rate)
            concat_updates = update_method(concat_grads, concat_params, learning_rate)
        else:
            action_updates = update_method(action_loss, action_params, learning_rate)
            obsval_updates = update_method(obsval_loss, obsval_params, learning_rate)
            dvalue_updates = update_method(dvalue_loss, dvalue_params, learning_rate)
            actval_updates = update_method(actval_loss, actval_params, learning_rate)
            concat_updates = update_method(concat_loss, concat_params, learning_rate)

        fit_action = theano.function([input_var, action_target], action_loss, updates=action_updates)
        fit_obsval = theano.function([input_var, obsval_target], obsval_loss, updates=obsval_updates)
        fit_dvalue = theano.function([input_var, dvalue_target], dvalue_loss, updates=dvalue_updates)
        fit_actval = theano.function([input_var, actval_target], actval_loss, updates=actval_updates)
        fit_concat = theano.function([input_var, concat_target], concat_loss, updates=concat_updates)

        predict_action = theano.function([input_var], action_prediction)
        predict_obsval = theano.function([input_var], obsval_prediction)
        predict_dvalue = theano.function([input_var], dvalue_prediction)
        predict_actval = theano.function([input_var], actval_prediction)
        predict_concat = theano.function([input_var], concat_prediction)

        nnet = Mock(
            fit_action=fit_action,
            fit_obsval=fit_obsval,
            fit_value=fit_dvalue,
            fit_actval=fit_actval,
            fit_both=fit_concat,
            predict_action=predict_action,
            predict_obsval=predict_obsval,
            predict_value=predict_dvalue,
            predict_actval=predict_actval,
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

    @staticmethod
    def _scale_outputs(ouputs, ranges, source_amplitude=1):
        """
        Scale the outputs back to the original amplitude.
        :param ouputs:
        :param ranges:
        :param source_amplitude:
        :return:
        """
        minranges = ranges[:, 0].T
        maxranges = ranges[:, 1].T
        scale = source_amplitude
        # scaled = (inputs - minranges) / (maxranges - minranges) * 2 * scale - scale
        scaled_back = (ouputs + scale) * (maxranges - minranges) / (2 * scale) + minranges
        return np.asmatrix(scaled_back, dtype=floatX)

    def exp_step(self, reward, observation, is_testing):
        return_action = Action()
        cur_observation = np.asmatrix(observation.doubleArray, dtype=floatX)
        if self.scale_range is not None:
            cur_observation = self._scale_inputs(cur_observation, self.observation_ranges, self.scale_range)
        pred_action, pred_observation, cur_observation_value = self._predict(cur_observation)
        double_action = self._explore(pred_action, self.action_stdev)
        loss = None
        if not is_testing:
            self.diverging = np.isnan(cur_observation).any() or np.isnan(double_action).any()
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
            if not terminal:
                loss = self.nnet.fit_both(self.last_state, np.hstack((self.last_action, observation, target_value)))
            else:
                loss = self.nnet.fit_actval(self.last_state, np.hstack((self.last_action, target_value)))
        else:
            if not terminal:
                loss = self.nnet.fit_obsval(self.last_state, np.hstack((observation, target_value)))
            else:
                loss = self.nnet.fit_value(self.last_state, target_value)
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

    # def run(self, logtofile=False):
    #     AgentLoader.loadAgent(self)


def main():
    AgentLoader.loadAgent(PredictiveAgent())


if __name__ == "__main__":
    main()
