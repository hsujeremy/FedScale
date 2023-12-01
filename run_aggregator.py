import fedscale.cloud.config_parser as parser
from fedscale.cloud.aggregation.aggregator import Aggregator


if __name__ == '__main__':
    agg = Aggregator(parser.args)
    agg.run()
