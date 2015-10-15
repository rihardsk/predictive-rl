__author__ = 'rihards'


from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from abc import ABCMeta, abstractmethod
import os
import numpy as np


class ExperimenterAgent(Agent):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.testing = False
        # self.total_reward = 0
        # self.episode_counter = 0
        # self.step_counter = 0
        self.epoch_losses = []
        # self.epoch_rewards = []
        self.exp_dir = ""
        self.learning_file = None
        self.learning_file_header = "epoch,loss"

    def agent_init(self, taskSpecification):
        pass

    def agent_start(self, observation):
        pass

    def agent_step(self, reward, observation):
        self.step_counter += 1
        if self.testing:
            action = self.step_test(reward, observation)
        else:
            action, loss = self.step_train(reward, observation)
            self.epoch_losses.append(loss)
        return action

    def agent_end(self, reward):
        pass

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

        elif params[0] == "start_testing":
            self.testing = True
            # self.total_reward = 0
            # self.episode_counter = 0

        elif params[0] == "finish_testing":
            self.testing = False

        elif params[0] == "set_dir":
            self.exp_dir = params[1]
        else:
            return "I don't know how to respond to your message"

    def open_learning_file(self, filename):
        try:
            os.stat(filename)
        except:
            os.makedirs(filename)
        self.learning_file = open(filename + '/learning.csv', 'w', 0)
        self.learning_file.write(self.learning_file_header + '\n')

    def save_losses(self, epoch, losses):
        if self.learning_file is None:
            self.open_learning_file(self.exp_dir)
        out = "{},{}\n".format(epoch, np.mean(losses))
        self.learning_file.write(out)

    @abstractmethod
    def save_agent(self, epoch):
        pass

    @abstractmethod
    def step_train(self, reward, observation):
        pass

    @abstractmethod
    def step_test(self, reward, observation):
        pass


