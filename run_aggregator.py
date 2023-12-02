import fedscale.cloud.config_parser as parser
from fedscale.cloud.gossip.gossip_aggregator import GossipAggregator


if __name__ == '__main__':
    agg = GossipAggregator(parser.args)
    agg.run()
