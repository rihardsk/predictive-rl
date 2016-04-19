from predictive_agent import PredictiveAgent
import copy
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


class PredictiveBasicAgent(PredictiveAgent):
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
        if num_hidden_value_units is None:
            dvaluelayers = [DenseLayer(commonlayers[-1], value_dims,
                                               nonlinearity=output_nonlinearity)]
        else:
            dvaluelayers = [DenseLayer(commonlayers[-1], num_hidden_value_units,
                                               nonlinearity=output_nonlinearity)]
            dvaluelayers.append(DenseLayer(dvaluelayers[-1], value_dims,
                                                   nonlinearity=output_nonlinearity))
        actvallayers = [layers.ConcatLayer([actionlayers[-1], dvaluelayers[-1]])]
        action_prediction = layers.get_output(actionlayers[-1])
        dvalue_prediction = layers.get_output(dvaluelayers[-1])
        actval_prediction = layers.get_output(actvallayers[-1])
        input_var = commonlayers[0].input_var
        action_target = T.matrix(name="action_target", dtype=floatX)
        dvalue_target = T.matrix(name="value_target", dtype=floatX)
        actval_target = T.matrix(name="actval_target", dtype=floatX)
        action_loss = objectives.squared_error(action_prediction, action_target).mean()
        dvalue_loss = objectives.squared_error(dvalue_prediction, dvalue_target).mean()
        actval_loss = objectives.squared_error(actval_prediction, actval_target).mean()
        if l1_weight is not None:
            action_l1penalty = regularize_layer_params(commonlayers + actionlayers, l1) * l1_weight
            dvalue_l1penalty = regularize_layer_params(commonlayers + dvaluelayers, l1) * l1_weight
            actval_l1penalty = regularize_layer_params(commonlayers + actionlayers + dvaluelayers, l1) * l1_weight
            action_loss += action_l1penalty
            dvalue_loss += dvalue_l1penalty
            actval_loss += actval_l1penalty
        if l2_weight is not None:
            action_l2penalty = regularize_layer_params(commonlayers + actionlayers, l2) * l2_weight
            dvalue_l2penalty = regularize_layer_params(commonlayers + dvaluelayers, l2) * l2_weight
            actval_l2penalty = regularize_layer_params(commonlayers + actionlayers + dvaluelayers, l2) * l2_weight
            action_loss += action_l2penalty
            dvalue_loss += dvalue_l2penalty
            actval_loss += actval_l2penalty
        action_params = layers.get_all_params(actionlayers[-1], trainable=True)
        dvalue_params = layers.get_all_params(dvaluelayers[-1], trainable=True)
        actval_params = layers.get_all_params(actvallayers[-1], trainable=True)
        if grad_clip is not None:
            action_grads = theano.grad(action_loss, action_params)
            dvalue_grads = theano.grad(dvalue_loss, dvalue_params)
            actval_grads = theano.grad(actval_loss, actval_params)
            action_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in action_grads]
            dvalue_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in dvalue_grads]
            actval_grads = [updates.norm_constraint(grad, grad_clip, range(grad.ndim)) for grad in actval_grads]
            action_updates = update_method(action_grads, action_params, learning_rate)
            dvalue_updates = update_method(dvalue_grads, dvalue_params, learning_rate)
            actval_updates = update_method(actval_grads, actval_params, learning_rate)
        else:
            action_updates = update_method(action_loss, action_params, learning_rate)
            dvalue_updates = update_method(dvalue_loss, dvalue_params, learning_rate)
            actval_updates = update_method(actval_loss, actval_params, learning_rate)

        fit_action = theano.function([input_var, action_target], action_loss, updates=action_updates)
        fit_dvalue = theano.function([input_var, dvalue_target], dvalue_loss, updates=dvalue_updates)
        fit_actval = theano.function([input_var, actval_target], actval_loss, updates=actval_updates)

        predict_action = theano.function([input_var], action_prediction)
        predict_dvalue = theano.function([input_var], dvalue_prediction)
        predict_actval = theano.function([input_var], actval_prediction)

        nnet = Mock(
            fit_action=fit_action,
            fit_value=fit_dvalue,
            fit_actval=fit_actval,
            predict_action=predict_action,
            predict_value=predict_dvalue,
            predict_actval=predict_actval,
        )
        return nnet

    def _predict(self, observation):
        pred_matrix = self.nnet.predict_actval(observation)
        next_action = pred_matrix[:, 0:self.action_size]
        next_observation = None  # For compatability with base class
        observation_value = pred_matrix[:, -1]
        return next_action, next_observation, observation_value

    def _do_training(self, reward, observation, action, terminal, observation_value, pred_observation, pred_action):
        cur_state = np.asmatrix(observation, dtype=floatX)  # TODO: check if conversion can be removed
        last_state_value = self.last_state_value
        if terminal:
            target_value = reward
        else:
            target_value = reward + self.discount * observation_value

        # this reflects whether the value of the last_state has risen after the value_network update above
        mask = target_value > last_state_value

        if mask[0, 0]:
            loss = self.nnet.fit_actval(self.last_state, np.hstack((self.last_action, target_value)))
        else:
            loss = self.nnet.fit_value(self.last_state, target_value)
        self.last_state = cur_state
        self.last_action = action
        self.last_original_action = pred_action
        self.last_state_value = observation_value
        self.pred_observation = pred_observation
        return loss

def main():
    AgentLoader.loadAgent(PredictiveBasicAgent())


if __name__ == "__main__":
    main()
