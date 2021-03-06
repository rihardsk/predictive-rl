__author__ = 'rihards'

import copy
import cPickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3
import time
import nn
import layers
import numpy as np
import data_set
import argparse
import os


class cacla_agent(Agent):
    randGenerator = np.random

    def __init__(self):
        """
        Mostly just read command line arguments here. We dostart this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        # Handle command line argument:
        parser = argparse.ArgumentParser(description='Neural rl agent.')
        parser.add_argument('--action_learning_rate', type=float, default=.0002,
                            help='Learning rate')
        parser.add_argument('--value_learning_rate', type=float, default=.0002,
                            help='Learning rate')
        parser.add_argument('--discount', type=float, default=.95,
                            help='Discount rate')
        parser.add_argument('--exp_pref', type=str, default="",
                            help='Experiment name prefix')
        parser.add_argument('--nn_file', type=str, default=None,
                            help='Pickle file containing trained net.')
        # Create instance variables directy from the arguments:
        parser.parse_known_args(namespace=self)


        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = "experiments/" + self.exp_pref + time_str + \
                        "a-{}_v-{}".format(self.action_learning_rate, self.value_learning_rate).replace(".", "p") + \
                        "_" + "{}".format(self.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except:
            os.makedirs(self.exp_dir)
        return

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

        observations = TaskSpec.getDoubleObservations() # TODO: take care of int observations
        self.observation_size = len(observations)

        actions = TaskSpec.getDoubleActions()
        self.action_size = len(actions)

        self.testing = False
        self.batch_size = 32
        self.episode_counter = 0
        self.step_counter = 0

        if self.nn_file is None:
            self.action_network = self._init_action_network(len(observations), len(actions))
            self.value_network = self._init_value_network(len(observations), 1)
        else:
            handle = open(self.nn_file, 'r')
            self.network = cPickle.load(handle)

        self.action_stdev = 0.01
        self.gamma = 0.9 # TaskSpec.getDiscountFactor()

        self.data_set = data_set.DataSet(
            len(observations),
            len(actions),
            observation_dtype='float32',
            action_dtype='float32',
        )
        # just needs to be big enough to create phi's
        self.test_data_set = data_set.DataSet(
            len(observations),
            len(actions),
            observation_dtype='float32',
            action_dtype='float32'
        )

    def _init_action_network(self, input_dims, output_dims, minibatch_size=32):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        layer1 = layers.FlatInputLayer(minibatch_size, input_dims)
        layer2 = layers.DenseLayer(layer1, 15, 0.1, 0, layers.sigmoid)
        layer3 = layers.DenseLayer(layer2, output_dims, 0.1, 0, layers.sigmoid)
        layer4 = layers.OutputLayer(layer3)
        return nn.NN([layer1, layer2, layer3, layer4], batch_size=minibatch_size, learning_rate=self.action_learning_rate)

    def _init_value_network(self, input_dims, output_dims, minibatch_size=32):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        layer1 = layers.FlatInputLayer(minibatch_size, input_dims)
        layer2 = layers.DenseLayer(layer1, 15, 0.1, 0, layers.sigmoid)
        layer3 = layers.DenseLayer(layer2, output_dims, 0.1, 0, layers.sigmoid)
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

        self.step_counter = 0
        self.batch_counter = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        #this_int_action = self.randGenerator.randint(0, self.num_actions-1)
        observation_matrix = np.asmatrix(observation.doubleArray, dtype='float32')
        actions = self.action_network.fprop(observation_matrix)
        return_action = Action()
        return_action.doubleArray = [actions]

        self.last_action = copy.deepcopy(return_action)

        self.last_observation = observation.doubleArray

        return return_action

    def _choose_action(self, data_set, cur_observation, reward, action_stdev=None):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """
        data_set.add_sample(self.last_observation,
                            self.last_action.doubleArray,
                            reward, False)

        double_action = self.action_network.fprop(np.asmatrix(cur_observation, dtype='float32'))

        # in order for the agent to learn we need some exploration
        double_action = double_action + 0 if action_stdev is None else self.randGenerator.normal(0, action_stdev, len(double_action))
        return np.clip(double_action, 0, 1)

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, actions, rewards, next_states, terminals = \
                                self.data_set.random_batch(self.batch_size)
        values = self.value_network.fprop(states)
        target_values = rewards + np.multiply(self.gamma * self.value_network.fprop(next_states), ~terminals)
        self.value_network.train_model_batch(states, target_values)
        updated_values = self.value_network.fprop(states)
        mask = updated_values > values
        return self.action_network.train_model_batch(states[mask[:, 0]], actions[mask[:, 0]])

    def agent_step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - An observation of type rlglue.types.Observation

        Returns:
           An action of type rlglue.types.Action

        """

        self.step_counter += 1
        return_action = Action()
        cur_observation = observation.doubleArray

        #TESTING---------------------------
        if self.testing:
            self.total_reward += reward
            double_action = self._choose_action(self.test_data_set,
                                             cur_observation, np.clip(reward, -1, 1))
            # if self.pause > 0:
            #     time.sleep(self.pause)

        #NOT TESTING---------------------------
        else:
            double_action = self._choose_action(self.data_set, cur_observation,
                                             np.clip(reward, -1, 1), self.action_stdev)

            if len(self.data_set) > self.batch_size:
                loss = self._do_training()
                self.batch_counter += 1
                self.loss_averages.append(loss)

        return_action.doubleArray = [double_action]

        self.last_action = copy.deepcopy(return_action)
        self.last_observation = cur_observation

        return_action.doubleArray = [double_action * 2 - 1]
        return return_action

    def agent_end(self, reward):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.

        Returns:
            None
        """
        self.episode_counter += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            self.total_reward += reward
        else:
            print "Simulated at a rate of {}/s \n Average loss: {}".format(\
                self.batch_counter/total_time,
                np.mean(self.loss_averages))

            #self._update_learning_file()

            # Store the latest sample.
            self.data_set.add_sample(self.last_observation,
                                     self.last_action.doubleArray,
                                     np.clip(reward, -1, 1),
                                     True)

    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """
        pass

    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file.
        """

        #WE NEED TO DO THIS BECAUSE agent_end is not called
        # we run out of steps.
        if in_message.startswith("episode_end"):
            self.agent_end(0)

        elif in_message.startswith("finish_epoch"):
            epoch = int(in_message.split(" ")[1])
            net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                            '.pkl', 'w')
            cPickle.dump(self.action_network, net_file, -1)
            net_file.close()

        elif in_message.startswith("start_testing"):
            self.testing = True
            self.total_reward = 0
            self.episode_counter = 0

        elif in_message.startswith("finish_testing"):
            self.testing = False
            holdout_size = 3200
            epoch = int(in_message.split(" ")[1])

            #if self.holdout_data is None:
            #    self.holdout_data = self.data_set.random_batch(holdout_size)[0]

            holdout_sum = 0
            # for i in range(holdout_size):
            #     holdout_sum += np.mean(
            #         self.network.q_vals(self.holdout_data[i, ...]))
            #
            # self._update_results_file(epoch, self.episode_counter,
            #                           holdout_sum / holdout_size)
        else:
            return "I don't know how to respond to your message"

def main():
    AgentLoader.loadAgent(cacla_agent())


if __name__ == "__main__":
    main()
