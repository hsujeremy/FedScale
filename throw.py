import argparse
import grpc
import concurrent.futures as futures
import logging
import fedscale.cloud.config_parser as parser

from fedscale.cloud.gossip.gossip_executor import Executor


def create_server(port: int):
    e = Executor(args=parser.args, client_id=port)
    e.run()


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="Run the server.")
    aparser.add_argument('port', type=int, help='The port of the server.')
    args = aparser.parse_args()
    create_server(args.port)


