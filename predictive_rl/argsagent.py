from experimenter_agent import ExperimenterAgent
import argparse
from abc import ABCMeta, abstractmethod


class ArgsAgent(ExperimenterAgent):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(ArgsAgent, self).__init__()
        # Handle command line argument:
        parser = argparse.ArgumentParser(description=self._get_description())
        self._add_parse_args(parser)

        # Create instance variables directy from the arguments:
        # parser.parse_known_args(namespace=self)

        if len(kwargs) == 0:
            args = parser.parse_args()
        else:
            ns = argparse.Namespace(**kwargs)
            args = parser.parse_args([], ns)
        self._get_parsed_args(args)

    def _get_description(self):
        return "RL-glue agent"

    @abstractmethod
    def _add_parse_args(self, parser):
        pass

    @abstractmethod
    def _get_parsed_args(self, args):
        pass
