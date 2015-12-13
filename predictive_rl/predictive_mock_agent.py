from predictive_agent import PredictiveAgent
from cacla_agent_lasagne import CaclaAgentLasagne as cal
from rlglue.agent import AgentLoader as AgentLoader
import numpy as np
import theano
from lasagne import updates
from lasagne import nonlinearities

__author__ = 'rihards'
floatX = theano.config.floatX


class MockNnet(object):
    def __init__(self, nn_act, nn_val, action_dims):
        self.nn_act = nn_act
        self.nn_val = nn_val
        self.action_dims = action_dims
        self.fit_action = nn_act.fit
        self.fit_value = nn_val.fit
        self.predict_action = nn_act.predict
        self.predict_value = nn_val.predict

    def fit_both(self, state, action_and_value):
        action = action_and_value[:,0:self.action_dims]
        value = action_and_value[:,self.action_dims:]
        action_loss = self.nn_act.fit(state, action)
        value_loss = self.nn_val.fit(state, value)
        return (action_loss + value_loss) / 2.

    def predict_both(self, state):
        action = self.nn_act.predict(state)
        value = self.nn_val.predict(state)
        return np.hstack((action, value))


class PredictiveMockAgent(PredictiveAgent):
    randGenerator = np.random

    def __init__(self):
        """
        Mostly just read command line arguments here. We do this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        super(PredictiveMockAgent, self).__init__()

    # def _init_network(self):
    #     if self.nn_file is None:
    #         self.nnet = self.create_nnet(input_dims=self.observation_size,
    #                                      action_dims=self.action_size,
    #                                      observation_dims=self.observation_size,
    #                                      value_dims=1,
    #                                      learning_rate=self.learning_rate,
    #                                      l1_weight=self.l1_weight,
    #                                      l2_weight=self.l2_weight,
    #                                      num_hidden_units=self.nn_hidden_size,
    #                                      num_hidden_action_units=self.nn_hidden_action_size,
    #                                      num_hidden_observ_units=self.nn_hidden_observ_size,
    #                                      num_hidden_value_units=self.nn_hidden_value_size,
    #                                      batch_size=1)
    #     else:
    #         handle = open(self.nn_file, 'r')
    #         self.nnet = cPickle.load(handle)


    @staticmethod
    def create_nnet(input_dims, action_dims, observation_dims, value_dims, learning_rate, l1_weight=None, l2_weight=None,
                    num_hidden_units=20, num_hidden_action_units=None, num_hidden_observ_units=None, num_hidden_value_units=None,
                    batch_size=32, max_train_epochs=1, hidden_nonlinearity=nonlinearities.rectify,
                    output_nonlinearity=None, update_method=updates.sgd):

        nn_val = cal._create_nnet(input_dims, value_dims, learning_rate, num_hidden_units, batch_size, max_train_epochs,
                                  hidden_nonlinearity, output_nonlinearity, update_method)

        nn_act = cal._create_nnet(input_dims, action_dims, learning_rate, num_hidden_units, batch_size, max_train_epochs,
                                  hidden_nonlinearity, output_nonlinearity, update_method)


        nnet = MockNnet(
            nn_act,
            nn_val,
            action_dims,
        )
        return nnet

    def _predict(self, observation):
        pred_matrix = self.nnet.predict_both(observation)
        next_action = pred_matrix[:, 0:self.action_size]
        next_observation = None
        observation_value = pred_matrix[:, self.action_size:]
        return next_action, next_observation, observation_value

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
            loss = self.nnet.fit_both(self.last_state, np.hstack((self.last_action, target_value)))
        else:
            loss = self.nnet.fit_value(self.last_state, target_value)
        self.last_state = cur_state
        self.last_action = action
        self.last_original_action = pred_action
        self.last_state_value = observation_value
        self.pred_observation = pred_observation
        return loss


def main():
    AgentLoader.loadAgent(PredictiveMockAgent())


if __name__ == "__main__":
    main()
