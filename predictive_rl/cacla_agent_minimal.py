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
import nn
import layers
import numpy as np
import data_set

class cacla_agent_minimal(Agent):
    pass