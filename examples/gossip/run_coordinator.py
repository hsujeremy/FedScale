import fedscale.cloud.config_parser as parser
from fedscale.cloud.gossip.gossip_coordinator import GossipCoordinator


if __name__ == '__main__':
    print(parser.args)
    coord = GossipCoordinator(parser.args)
    coord.run()
