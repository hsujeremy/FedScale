import sys, os

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.aggregation.aggregator import Aggregator
Demo_Aggregator = Aggregator(parser.args)
### On CPU
parser.args.use_cuda = "False"
Demo_Aggregator.run()