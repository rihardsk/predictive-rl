__author__ = 'rihards'


from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from abc import ABCMeta, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt


class ExperimenterAgent(object, Agent):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.testing = False
        # self.total_reward = 0
        # self.episode_counter = 0
        # self.step_counter = 0
        self.epoch_losses = []
        self.epoch_rewards = []
        self.exp_dir = ""
        self.learning_file = None
        self.results_file = None
        self.learning_file_header = "epoch,loss"
        self.results_file_header = "epoch,episodes,total_reward,mean_reward"
        self.collect_rewards = True

    def agent_init(self, taskSpecification):
        pass

    def agent_start(self, observation):
        pass

    def agent_step(self, reward, observation):
        # self.step_counter += 1
        if self.testing:
            if self.collect_rewards:
                self.epoch_rewards.append(reward)
            action = self.exp_step(reward, observation, self.testing)
        else:
            action, loss = self.exp_step(reward, observation, self.testing)
            if loss is not None:
                self.epoch_losses.append(loss)
        return action

    def agent_end(self, reward):
        loss = self.exp_end(reward, self.testing)
        if not self.testing and loss is not None:
            self.epoch_losses.append(loss)
        elif self.testing and reward is not None:
            self.epoch_rewards.append(reward)

    def agent_cleanup(self):
        pass

    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file.
        """
        params = in_message.split(" ")

        # WE NEED TO DO THIS BECAUSE agent_end is not called
        # we run out of steps (experiment ended the episode manually).
        if params[0] == "episode_end":
            self.agent_end(None)

        elif params[0] == "start_epoch":
            self.epoch_losses = []

        elif params[0] == "finish_epoch":
            epoch = int(in_message.split(" ")[1])
            self.save_agent(epoch)
            self.save_losses(epoch, self.epoch_losses)
            self.epoch_losses = []

        elif params[0] == "start_testing":
            self.testing = True
            # self.total_reward = 0
            # self.episode_counter = 0

        elif params[0] == "finish_testing":
            epoch = int(in_message.split(" ")[1])
            self.testing = False
            if self.collect_rewards:
                self.save_results(epoch, self.epoch_rewards)
                self.epoch_rewards = []

        elif params[0] == "set_dir":
            self.exp_dir = params[1]
        else:
            return "I don't know how to respond to your message"

    def open_learning_file(self, dir_name):
        try:
            os.stat(dir_name)
        except:
            os.makedirs(dir_name)
        self.learning_file = open(os.path.join(dir_name, 'learning.csv'), 'w', 0)
        self.learning_file.write(self.learning_file_header + '\n')

    def open_results_file(self, dir_name):
        try:
            os.stat(dir_name)
        except:
            os.makedirs(dir_name)
        self.results_file = open(os.path.join(dir_name, 'results.csv'), 'w', 0)
        self.results_file.write(self.results_file_header + '\n')

    def save_losses(self, epoch, losses):
        if self.learning_file is None:
            self.open_learning_file(self.exp_dir)
        out = "{},{}\n".format(epoch, np.mean(losses))
        self.learning_file.write(out)

    def save_results(self, epoch, rewards):
        if self.results_file is None:
            self.open_results_file(self.exp_dir)
        out = "{},{},{},{}\n".format(epoch, len(rewards), np.sum(rewards), np.mean(rewards))
        self.results_file.write(out)

    @abstractmethod
    def save_agent(self, epoch):
        pass

    @abstractmethod
    def exp_step(self, reward, observation, is_testing):
        pass

    @abstractmethod
    def exp_end(self, reward, is_testing):
        pass

    @staticmethod
    def plot_results(file):
        # Modify this to do some smoothing...
        kernel = np.array([1.] * 1)
        kernel = kernel / np.sum(kernel)

        results = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1)
        plt.subplot(1, 2, 1)
        plt.plot(results[:, 0], np.convolve(results[:, 2], kernel, mode='same'), '-*')
        #plt.ylim([0, 250])
        plt.subplot(1, 2, 2)
        plt.plot(results[:, 0], results[:, 3], '--')
        #plt.ylim([0, 4])
        plt.show()\

    @staticmethod
    def plot_learning(file):
        # Modify this to do some smoothing...
        kernel = np.array([1.] * 1)
        kernel = kernel / np.sum(kernel)

        results = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1)
        plt.plot(results[:, 0], np.convolve(results[:, 1], kernel, mode='same'), '-*')
        plt.show()
