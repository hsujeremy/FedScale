import torch
import logging
import math
from torch.autograd import Variable
import numpy as np

import sys, os

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.executor import Executor
### On CPU
parser.args.use_cuda = "False"
Demo_Executor = Executor(parser.args)
Demo_Executor.run()