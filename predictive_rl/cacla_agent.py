__author__ = 'rihards'

import random
import sys
import copy
import pickle
import cPickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
#from pylearn2.models import mlp
import mlp
import time


class cacla_agent(Agent):
    randGenerator=random.Random()

    def __init__(self):
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
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), \
                " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), \
                " expecting max action to be a number not a special value"
            #self.num_actions = TaskSpec.getIntActions()[0][1]+1
        else:
            print "INVALID TASK SPEC"

        observations = TaskSpec.getDoubleObservations() # TODO: take care of int observations
        actions = TaskSpec.getDoubleObservations()

        if self.nn_file is None:
            self.action_network = self._init_network(len(observations), len(actions))
            self.value_network = self._init_network(len(observations), 1)
        else:
            handle = open(self.nn_file, 'r')
            self.network = cPickle.load(handle)

    def _init_network(self, input_dims, output_dims):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        return mlp.MLP(self.randGenerator, input_dims, input_dims + output_dims, output_dims)

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
        this_int_action = self.action_network.
        return_action = Action()
        return_action.intArray = [this_int_action]

        self.last_action = copy.deepcopy(return_action)

        self.last_img = self._resize_observation(observation.intArray)

        return return_action