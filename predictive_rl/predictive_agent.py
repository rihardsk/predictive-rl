__author__ = 'rihards'

from experimenter_agent import ExperimenterAgent
import argparse
from rlglue.utils import TaskSpecVRLGLUE3
import numpy as np
import theano

floatX = theano.config.floatX


class PredictiveAgent(ExperimenterAgent):
    def __init__(self):
        """
        Mostly just read command line arguments here. We do this here
        instead of agent_init to make it possible to use --help from
        the command line without starting an experiment.
        """
        super(ExperimenterAgent, self).__init__()

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
        # parser.add_argument('--action_stdev', type=float, default=0.1,
        #                     help='Action space exploration standard deviation for Gaussian distribution. '
        #                          'Applied to action range.')
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
        # self.action_stdev = args.action_stdev
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
