import fedscale.cloud.config_parser as parser
from fedscale.cloud.gossip.gossip_executor import Executor


if __name__ == "__main__":
        # Usage: python throw.py --port PORT --num_executors N
    # Maybe not the best idea to have each executor know num_executors
    # Instead, have the coordiantor broadcast the number of clients to each executor when it starts training
    e = Executor(args=parser.args, client_id=parser.args.port)
    e.run()


