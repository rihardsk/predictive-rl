from predictive_agent import PredictiveAgent
import copy
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
import numpy as np
import theano

__author__ = 'rihards'
floatX = theano.config.floatX


class Mock(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PredictiveFutureAgent(PredictiveAgent):
    randGenerator = np.random
    
    def __init__(self, **kwargs):
        """
        Mostly just read command line arguments here. We do this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        super(PredictiveFutureAgent, self).__init__(**kwargs)

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
        pred_action, pred_observation, pred_observation_value = self._predict(cur_observation)
        double_action = self._explore(pred_action, self.action_stdev)

        return_action = Action()
        return_action.doubleArray = double_action

        self.last_state = cur_observation
        self.last_action = copy.deepcopy(double_action)
        self.last_predicted_value = pred_observation_value
        self.lastlast_state = None
        # self.last_original_action = pred_action

        return return_action

    def _do_training(self, reward, observation, action, terminal, pred_observation_value, pred_observation, pred_action):
        cur_state = np.asmatrix(observation, dtype=floatX)
        if self.lastlast_state is not None:
            if terminal:
                target_value = reward
            else:
                target_value = reward + self.discount * pred_observation_value

            # this reflects whether the value of the last_state has risen after the value_network update above
            mask = target_value > self.last_predicted_value

            if mask[0, 0]:
                if not terminal:
                    loss = self.nnet.fit_both(self.lastlast_state, np.hstack((self.last_action, observation, target_value)))
                else:
                    loss = self.nnet.fit_actval(self.lastlast_state, np.hstack((self.last_action, target_value)))
            else:
                if not terminal:
                    loss = self.nnet.fit_obsval(self.lastlast_state, np.hstack((observation, target_value)))
                else:
                    loss = self.nnet.fit_value(self.last_state, target_value)
        else:
            loss = None
        self.lastlast_state = self.last_state
        self.last_state = cur_state
        self.last_action = action
        self.last_predicted_value = pred_observation_value
        # self.last_pred_observation = pred_observation
        # self.last_pred_action = pred_action
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
                                         np.zeros_like(self.last_predicted_value, dtype=floatX),
                                         np.zeros_like(self.last_state, dtype=floatX),
                                         np.zeros_like(self.last_action, dtype=floatX),
                                         )
                return loss


def main():
    AgentLoader.loadAgent(PredictiveFutureAgent())


if __name__ == "__main__":
    main()
