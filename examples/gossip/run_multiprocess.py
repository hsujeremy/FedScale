import argparse
import multiprocessing
import time
import fedscale.cloud.config_parser as parser
from fedscale.cloud.gossip.gossip_coordinator import GossipCoordinator
from fedscale.cloud.gossip.gossip_executor import Executor

import datetime

def run_executor(port):
    parser.args.port = port
    e = Executor(args=parser.args, client_id=parser.args.port)
    e.run()


def run_coordinator(num_executors):
    parser.args.num_executors = num_executors
    c = GossipCoordinator(args=parser.args)
    c.run()


def main():
    parser.args.time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument("--num_executors", type=int, default=50)
    local_args = local_parser.parse_args()
    num_processes = local_args.num_executors + 1
    if num_processes < 2:
        print('Need at least 2 processes for at least one coordinator and executor')
        return
    print('Starting 1 coordinator and {} executors'.format(num_processes-1))
    cdtr_process = multiprocessing.Process(target=run_coordinator, args=(num_processes-1,))
    cdtr_process.start()

    executor_processes = []
    for i in range(1, num_processes):
        executor_process = multiprocessing.Process(target=run_executor, args=(i-1,))
        executor_process.start()
        executor_processes.append(executor_process)

    cdtr_process.join()
    for executor_process in executor_processes:
        executor_process.join()


if __name__ == "__main__":
    main()
