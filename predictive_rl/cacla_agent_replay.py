__author__ = 'rihards'

import cPickle
import numpy as np
import theano
from cacla_agent_lasagne import CaclaAgentLasagne
from ale_data_set import DataSet
from rlglue.agent import AgentLoader as AgentLoader

floatX = theano.config.floatX


class CaclaAgentReplay(CaclaAgentLasagne):
    randGenerator = np.random

    def __init__(self):
        super(CaclaAgentReplay, self).__init__()
        self.data_set = None

    def _add_parse_args(self, parser):
        super(CaclaAgentReplay, self)._add_parse_args(parser)
        parser.add_argument('--batch_size', type=int, default=256,
                            help="The neural nets' minibatch size")
        parser.add_argument('--replay_size', type=int, default=1024,
                            help='The number of examples to retrieve from the replay memory when training')
        parser.add_argument('--max_history', type=int, default=10000,
                            help='The max replay history to store from which to choose the random batches of samples')

    def _get_parsed_args(self, args):
        super(CaclaAgentReplay, self)._get_parsed_args(args)
        self.batch_size = args.batch_size
        self.replay_size = args.replay_size
        self.max_history = args.max_history

    def agent_init(self, taskSpecification):
        super(CaclaAgentReplay, self).agent_init(taskSpecification)
        self.data_set = DataSet(self.observation_size, 1, self.randGenerator, self.max_history, 1)

    def _init_value_network(self):
        if self.nn_value_file is None:
            self.value_network = self._create_nnet(self.observation_size,
                                                   1,
                                                   self.value_learning_rate,
                                                   batch_size=self.batch_size)
        else:
            handle = open(self.nn_value_file, 'r')
            self.value_network = cPickle.load(handle)

    def _init_action_network(self):
        if self.nn_action_file is None:
            self.action_network = self._create_nnet(self.observation_size,
                                                    self.action_size,
                                                    self.action_learning_rate,
                                                    batch_size=self.batch_size)
        else:
            handle = open(self.nn_action_file, 'r')
            self.action_network = cPickle.load(handle)

    # def agent_start(self, observation):
    #     return_action = super(CaclaAgentReplay, self).agent_start(observation)
    #     self.data_set.add_sample(self.last_state, self.last_action, None, False)
    #     return return_action

    def _do_training(self, reward, observation, action, terminal):
        terminal = np.asmatrix(terminal)
        cur_state = np.asmatrix(observation, dtype=floatX)
        if self.data_set.size > 0:
            re_states, re_actions, re_rewards, re_next_states, re_terminals = self.data_set.random_batch(self.replay_size)
            # ale_data_set stores states as batches of imgs. convert them to a 2D array
            re_states = re_states[:,0,0,:]
            re_next_states = re_next_states[:,0,0,:]

            states = np.vstack((self.last_state, re_states))
            actions = np.vstack((self.last_action, re_actions))
            rewards = np.vstack((reward, re_rewards))
            next_states = np.vstack((cur_state, re_next_states))
            terminals = np.vstack((terminal, re_terminals))
        else:
            states = self.last_state
            actions = self.last_action
            rewards = reward
            next_states = cur_state
            terminals = terminal

        initial_state_values = self.value_network.predict(states)
        target_values = rewards + np.multiply(self.discount * self.value_network.predict(next_states), ~terminals)
        self.value_network.fit(states, target_values)

        # this reflects whether the value of the last_state has risen after the value_network update above
        mask = target_values > initial_state_values

        if mask.any():
            mask = np.asarray(mask).ravel()
            net = self.action_network.fit(states[mask], actions[mask])
            loss = net.train_history_[-1]['train_loss']
        else:
            loss = None

        self.data_set.add_sample(self.last_state, self.last_action, reward, terminal)
        self.last_state = cur_state
        self.last_action = action
        return loss


def main():
    AgentLoader.loadAgent(CaclaAgentReplay())


if __name__ == "__main__":
    main()
