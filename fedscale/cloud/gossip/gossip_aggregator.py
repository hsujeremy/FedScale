import torch
import grpc
from concurrent import futures
import logging
import random
import collections
import numpy as np

import fedscale.cloud.logger.aggregator_logging as logger
import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc
from fedscale.cloud.resource_manager import ResourceManager
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud import commons

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB

class GossipAggregator(job_api_pb2_grpc.JobServiceServicer):
    def __init__(self, args):
        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu'
        )

        self.round_duration = 0.
        self.resource_manager = ResourceManager(commons.SIMULATION_MODE)
        self.client_manager = self.init_client_manager(args=args)

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queue =======
        self.individual_client_events = {}  # Unicast
        self.server_events_queue = collections.deque()

        self.client_training_results = {}
        
        pass 

    def run(self):
        self.setup_env()

        self.init_control_communication()
        self.event_monitor()
        self.stop()

    def add_event_handler(self, client_id, event, meta, data):
        """ Due to the large volume of requests, we will put all events into a queue first.

        Args:
            client_id (int): The client id.
            event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
            meta (dictionary or string): Meta message for grpc communication, could be event.
            data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

        """
        self.server_events_queue.append((client_id, event, meta, data))
    

    def setup_env(self):
        """Set up experiments environment and server optimizer
        """
        self.setup_seed(seed=1)

    def setup_seed(self, seed=1):
        """Set global random seed for better reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """

        # simulation mode
        num_of_executors = 0
        for ip_numgpu in self.args.executor_configs.split("="):
            ip, numgpu = ip_numgpu.split(':')
            for numexe in numgpu.strip()[1:-1].split(','):
                for _ in range(int(numexe.strip())):
                    num_of_executors += 1
        self.executors = list(range(num_of_executors))

        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )

        job_api_pb2_grpc.add_JobServiceServicer_to_server(
            self, self.grpc_server)
        
        port = '[::]:{}'.format(self.args.ps_port)
        logging.info(f'%%%%%%%%%% Opening aggregator server using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()
    
    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")

        # TODO: figure out what events to monitor - we don't send, so only sever events
        # TODO: wandb logging
        while True: 
            pass

    def stop(self):
        """
            TODO: wandb logging
        """

