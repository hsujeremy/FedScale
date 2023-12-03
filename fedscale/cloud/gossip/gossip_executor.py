import collections
import copy
import math
import random
import threading
import time
import grpc
import torch
import wandb
import pickle
import logging

from concurrent import futures
from random import Random
from argparse import Namespace

import numpy as np

import fedscale.cloud.gossip.job_api_pb2 as job_api_pb2
import fedscale.cloud.gossip.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.rl_client import RLClient
from fedscale.cloud.execution.tensorflow_client import TensorflowClient
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.fllibs import *
from fedscale.cloud.gossip.gossip_channel_context import GossipClientConnections
from fedscale.cloud.internal.tensorflow_model_adapter import TensorflowModelAdapter
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset

"""
Make a server for each client
"""
MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB
AGGREGATION_FREQUENCY = 5  # Number of iterations between aggregation

DEFAULT_NUM_ITERATIONS = 10


class Executor(job_api_pb2_grpc.JobServiceServicer):
    # TODO: for debugging purposes, maybe lower the number of rounds so things finish quicker
    def __init__(self, args, num_iterations=DEFAULT_NUM_ITERATIONS, client_id=0):
        logger.initiate_client_setting(client_id)

        self.num_iterations = num_iterations
        self.client_id = client_id
        self.executor_id = client_id
        self.max_retries = 10

        self.training_sets = self.test_dataset = None

        # init model weights here? training config contains model weights under "model"
        # model weights are stored under self.model_adapter.get_weights
        self.model_adapter = self.get_client_trainer(
            args).get_model_adapter(init_model())

        self.args = args
        self.num_executors = args.num_executors

        self.num_neighbors = 0
        self.neighbor_threshold = 0.7
        self.model_updates_threshold = 0.7

        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')
        # ======== Env Information ========
        self.this_rank = args.this_rank

        # ======== Event Queue ========
        self.individual_client_events = {}  # Unicast
        self.receive_events_queue = collections.deque()
        self.send_events_queue = collections.deque()
        self.coordinator_events_queue = collections.deque()
        self.waiting_for_start = True
        # ======== Model and Data ========
        self.training_sets = self.test_dataset = None
        self.model_wrapper = None
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.model_weights = None
        # self.temp_model_path = os.path.join(
        #     logger.logDir, 'model_'+str(args.this_rank)+".npy")
        self.last_saved_round = 0

        self.client_manager = self.init_client_manager(args=args)

        # ======== channels ========
        self.client_communicator = GossipClientConnections(
            args.ps_ip, client_id)

        # ======== runtime information ========
        self.collate_fn = None
        self.round = 0
        self.start_run_time = time.time()
        self.received_stop_request = False

        # ======== Wandb ========
        if args.wandb_token:
            os.environ['WANDB_API_KEY'] = args.wandb_token
            self.wandb = wandb
            if not self.wandb.run:
                self.wandb.init(project=f'fedscale-{args.job_name}',
                                name=f'executor{args.this_rank}-{args.time_stamp}',
                                group=f'{args.time_stamp}')
            else:
                logging.error("Warning: wandb has already been initialized")

        else:
            self.wandb = None

        # TODO: register neighbors
        super(Executor, self).__init__()

    def init_control_communication(self):
        """creates a server on the client
        """

        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )

        job_api_pb2_grpc.add_JobServiceServicer_to_server(
            self, self.grpc_server)

        port = '[::]:{}'.format(self.client_id + 4001)
        logging.info(
            f'%%%%%%%%%% Opening client for client {self.client_id} server using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()
        self.client_communicator.connect_to_coordinator()

    # TODO Figure out diff between adapter vs client and init_model vs get_client_trainer
    def init_model(self):
        """Initialize the model"""
        if self.args.engine == commons.TENSORFLOW:
            self.model_wrapper = TensorflowModelAdapter(init_model())
        elif self.args.engine == commons.PYTORCH:
            self.model_wrapper = TorchModelAdapter(
                init_model(),
                optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device))
        else:
            raise ValueError(f"{self.args.engine} is not a supported engine.")

    def get_client_trainer(self, conf):
        """
        Returns a framework-specific client that handles training and evaluation.
        :param conf: job config
        :return: framework-specific client instance
        """
        if conf.engine == commons.TENSORFLOW:
            return TensorflowClient(conf)
        elif conf.engine == commons.PYTORCH:
            if conf.task == 'rl':
                return RLClient(conf)
            else:
                return TorchClient(conf)
        raise "Currently, FedScale supports tensorflow and pytorch."

    def setup_env(self, seed=1):
        """Set up experiments environment
        """
        logging.info(f"(EXECUTOR:{self.client_id}) is setting up environ ...")
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
        if self.args.task == "rl":
            return train_dataset, test_dataset
        if self.args.task == 'nlp':
            self.collate_fn = collate
        elif self.args.task == 'voice':
            self.collate_fn = voice_collate_fn
        # load data partitionxr (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(
            data=test_dataset, args=self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")

        return training_sets, testing_sets

    # TODO: maybe change variable names later since we're no including the model
    # weights in the RPC reply
    def select_neighbors(self, min_replies, cur_time=0, buffer=0):
        """Randomly select neighbors to request weights from. This should be
        some order of magnitude higher than the minimum amount of clients we
        want to receive weight updates from (e.g., if we want at least 5
        replies, then we should select 10 neighbors).
        """
        logging.info("Selecting neighbors to send weights too...")

        # TODO: handle edge cases:
        # - Not enough clients in total to select from
        # - Enough total clients, but not any other active ones that are selected

        # clients_online = self.client_manager.getFeasibleClients(cur_time)
        # rng = Random()
        # rng.seed(233)
        # rng.shuffle(clients_online)
        # client_len = min(min_replies + buffer, len(clients_online)-1)
        # return clients_online[:client_len]

        # For now, just return everything
        total_executors = list(range(self.num_executors))
        return [i for i in total_executors if i != self.client_id]

    def override_conf(self, config):
        """ Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def train(self):
        train_config = {
            'learning_rate': self.args.learning_rate,
        }
        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(
            conf=client_conf, model_weights=self.model_adapter.get_weights())

        # use torchclient trainer
        return train_res

    def training_handler(self, conf, model_weights):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        self.model_adapter.set_weights(model_weights)
        conf.tokenizer = tokenizer
        conf.client_id = self.client_id
        client_data = self.training_sets if self.args.task == "rl" else \
            select_dataset(self.client_id, self.training_sets,
                           batch_size=conf.batch_size, args=self.args,
                           collate_fn=self.collate_fn
                           )

        # TODO figure out if current trainer handles set number of iterations at a time
        train_res = TorchClient(self.args).train(
            client_data=client_data, model=self.model_adapter.get_model(), conf=conf)

        return train_res

    def train_and_monitor(self):
        weights = None
        for i in range(self.num_iterations):
            if i > 0 and i % AGGREGATION_FREQUENCY == 0:
                logging.info("Selecting neighbors to send weights too...")
                neighbors = self.select_neighbors(
                    min_replies=self.neighbor_threshold)
                self.num_neighbors = len(neighbors)

                for neighbor in neighbors:
                    retries = 0
                    stub = self.client_communicator.stubs[neighbor]

                    # retry max_retries times
                    while retries < self.max_retries:
                        try:
                            logging.info(
                                f"Requesting weights from client {neighbor}...")
                            stub.REQUEST_WEIGHTS(
                                job_api_pb2.WeightRequest(
                                    client_id=str(self.client_id),
                                    executor_id=str(self.executor_id)
                                )
                            )
                            break
                        except Exception as e:
                            logging.info(
                                f"Failed to send request weights ping to {neighbor} with exception {e}, retrying...")
                            time.sleep(0.1)
                            retries += 1

                # Wait for neighbors to send back weights
                # TODO replace temp placeholder with min_num_neighbors = int(self.model_updates_threshold * self.num_neighbors)
                self.min_num_neighbors = self.num_neighbors
                self.model_weights = self.model_wrapper.get_weights()
                logging.info(
                    f"Client {self.client_id} waiting to receive weights from {self.min_num_neighbors} neighbors")
                while self.model_in_update < self.min_num_neighbors:
                    if self.receive_events_queue:
                        client_id, received_weights = self.receive_events_queue.popleft()
                        received_weights = self.deserialize_response(
                            received_weights)
                        self.aggregate_weights_handler(received_weights)

                    # check queue
                    # if we have any requests to send out weights
                    while self.send_events_queue:
                        client_id, _ = self.send_events_queue.popleft()
                        # process event
                        if weights:
                            self.client_send_weights_handler(
                                int(client_id), weights)

                    time.sleep(0.1)

                self.model_in_update = 0

            # check queue
            # if we have any requests to send out weights
            while self.send_events_queue:
                client_id, _ = self.send_events_queue.popleft()
                # process event
                if weights:
                    self.client_send_weights_handler(
                        int(client_id), weights)

            logging.info(
                f"Training iteration {i + 1} of {self.num_iterations}")
            weights = self.train()["update_weight"]

    def run(self):
        """
            after each training loop,
                check # of incoming requests:
                    clients to send weights
        """
        self.setup_env()
        self.init_control_communication()
        self.init_model()
        self.training_sets, self.testing_sets = self.init_data()
        time.sleep(10)

        logging.info('Registering client with coordinator')
        self.client_register()

        while self.waiting_for_start:
            logging.info(f"Executor {self.client_id} waiting to start...")
            time.sleep(2)

        self.client_communicator.connect_to_executors(self.num_executors)
        logging.info("Starting loop...")
        # while True:
        #     neighbor = 0 if self.client_id == 1 else 1
        #     logging.info(f"Pinging client {neighbor}...")
        #     stub = self.client_communicator.stubs[0]
        #     response = self.client_ping(stub)

        #     event = response.event
        #     logging.info(event)
        #     time.sleep(5)

        #     break

        self.train_and_monitor()
        self.stop()
        pass

    def deserialize_response(self, responses):
        """Deserialize the response from executor

        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """ Serialize the response to send to server upon assigned job completion

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.

        """
        return pickle.dumps(responses)

    def client_send_weights_handler(self, client_id, train_res):
        """Uploads model to client at client_id.

        Args:
            client_id (int): The client id.
            train_res (dictionary): The results from training.

        """
        logging.info(f"Sending weights to client {client_id}")

        # Send model to client
        # TODO figure out if we should implement using futures
        future_call = self.client_communicator.stubs[client_id].UPLOAD_WEIGHTS(
            job_api_pb2.UploadWeightRequest(client_id=str(self.client_id),
                                            executor_id=str(self.executor_id),
                                            weights=self.serialize_response(
                                                train_res)
                                            ))
        # future_call.add_done_callback(
        #     lambda _response: self.dispatch_worker_events(_response.result()))

    def dispatch_receive_events(self, request):
        """Add new events to worker queue for sending out weights.

        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to events_queue.

        """
        self.receive_events_queue.append(
            (request.client_id, request.weights))

    def dispatch_send_events(self, request):
        """Add new events to worker queue for receiving weights.

        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to events_queue.
        """
        self.send_events_queue.append((request.client_id, request.executor_id))

    def dispatch_coordinator_events(self, event):
        self.coordinator_events_queue.append(event)

    def init_client_manager(self, args):
        """ Initialize client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            ClientManager: The client manager class

        Currently we implement two client managers:

        1. Random client sampler - it selects participants randomly in each round
        [Ref]: https://arxiv.org/abs/1902.01046

        2. Oort sampler
        Oort prioritizes the use of those clients who have both data that offers the greatest utility
        in improving model accuracy and the capability to run training quickly.
        [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai

        """
        # TODO: do only selected clients
        # sample_mode: random or oort
        client_manager = ClientManager(args.sample_mode, args=args)

        return client_manager

    def aggregate_weights_handler(self, weights):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result
        """
        self.update_lock.acquire()

        self.model_in_update += 1
        self.update_weight_aggregation(weights)

        self.update_lock.release()

    def update_weight_aggregation(self, weights):
        """Updates the aggregation with weights received from neighbor.

        Args:
            weights (list): The weights received from neighbor.
        """
        if type(weights) is dict:
            weights = [x for x in weights.values()]
        self.model_weights = list(
            map(lambda x, y: x+y, self.model_weights, weights))
        if self._is_last_result_in_round():
            self.model_weights = list(
                map(lambda x: x/int(self.num_neighbors * self.model_updates_threshold), self.model_weights))
            self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))

    def _is_last_result_in_round(self):
        return self.model_in_update == self.min_num_neighbors

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def executor_info_handler(self, executor_id, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.

        Args:
            executor_id (int): Executor Id
            info (dictionary): Executor information

        """
        # TODO Figure out how much of this logic we actually need
        self.registered_executor_info.add(executor_id)
        logging.info(
            f"Received executor {executor_id} information, {len(self.registered_executor_info)}/{len(self.executors)}")

        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout
        if self.experiment_mode == commons.SIMULATION_MODE:
            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executor_id, info)
                # start to sample clients
                self.round_completion_handler()
        else:
            # In real deployments, we need to register for each client
            self.client_register_handler(executor_id, info)
            if len(self.registered_executor_info) == len(self.executors):
                self.round_completion_handler()

    def client_register_handler(self, executor_id, info):
        """Triggered once receive new executor registration.

        Args:
            executor_id (int): Executor Id
            info (dictionary): Executor information

        """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info['size']:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = 1
            if self.client_profiles:
                mapped_id = (self.num_of_clients +
                             1) % len(self.client_profiles)

            systemProfile = self.client_profiles.get(
                mapped_id, {'computation': 1.0, 'communication': 1.0})

            # interchangable since we're only putting one client per executor
            client_id = executor_id
            self.client_manager.register_client(
                executor_id, client_id, size=_size, speed=systemProfile)
            self.client_manager.registerDuration(
                client_id,
                batch_size=self.args.batch_size,
                local_steps=self.args.local_steps,
                upload_size=self.model_update_size,
                download_size=self.model_update_size
            )
            self.num_of_clients += 1

        logging.info("Info of all feasible clients {}".format(
            self.client_manager.getDataInfo()))

    def client_register(self):
        """Register the client information to neighbors
        """
        start_time = time.time()

        while time.time() - start_time < 180:
            try:
                # is there a race condition where True is set after the response but
                # the aggregator has already broadcasted?
                self.waiting_for_start = True
                response = self.client_communicator.aggregator_stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=str(self.executor_id),
                        executor_id=str(self.executor_id),
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler())
                    )
                )
                # self.dispatch_worker_events(response)
                break
            except Exception as e:
                self.waiting_for_start = False
                logging.warning(
                    f"Failed to connect to coordinator, with error:\n{e}\nWill retry in 5 sec.")
                time.sleep(5)

    def CLIENT_REGISTER(self, request, context):
        """FL TorchClient register to the aggregator

        Args:
            request (RegisterRequest): Registeration request info from executor.

        Returns:
            ServerResponse: Server response to registeration request

        """
        pass
        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id uses the same executor_id (VMs) in simulations
        executor_id = request.executor_id
        executor_info = self.deserialize_response(request.executor_info)
        if executor_id not in self.individual_client_events:
            # logging.info(f"Detect new client: {executor_id}, executor info: {executor_info}")
            self.individual_client_events[executor_id] = collections.deque()
        else:
            logging.info(f"Previous client: {executor_id} resumes connecting")

        # We can customize whether to admit the clients here
        self.executor_info_handler(executor_id, executor_info)
        dummy_data = self.serialize_response(commons.DUMMY_RESPONSE)

        return job_api_pb2.ServerResponse(event=commons.DUMMY_EVENT,
                                          meta=dummy_data, data=dummy_data)

    def REQUEST_WEIGHTS(self, request, context):
        """Handle incoming requests for model weights.

        Should send back a response to the sender immediately, but then submit
        an internal queue "request" to send the weights to the sender via the
        UploadWeights RPC.

        Args:
            request (WeightRequest): Ping request info from neighbor.

        Returns:
            ServerResponse: Server response to weight request

        """
        self.dispatch_send_events(request)

        event = commons.GL_ACK
        response_data = response_msg = self.serialize_response(
            commons.GL_ACK_RESPONSE)
        response = job_api_pb2.ServerResponse(event=event,
                                              meta=response_msg, data=response_data)

        return response

    def UPLOAD_WEIGHTS(self, request, context):
        """Handle incoming model weights from neighbors.

        Args:
            request (UploadWeightRequest): Upload weight request info from neighbor.

        Returns:
            ServerResponse: Server response to job completion request

        """
        logging.info("Received weights from neighbor")
        self.dispatch_receive_events(request)

        event = commons.GL_ACK
        response_data = response_msg = self.serialize_response(
            commons.GL_ACK_RESPONSE)
        response = job_api_pb2.ServerResponse(event=event,
                                              meta=response_msg, data=response_data)

        return response

    # This is called by the coordinator to either notify the client to:
    # - Start training
    # - Stop training and shut down
    def CLIENT_PING(self, request, context):
        if self.waiting_for_start:
            self.waiting_for_start = False

        executor_id = request.executor_id
        client_id = request.client_id
        num_executors = int(request.num_executors)

        logging.info(
            f"Received ping from coordinator, setting num_executors: {num_executors}")
        self.num_executors = num_executors
        response = job_api_pb2.ServerResponse(event=commons.GL_ACK,
                                              meta=self.serialize_response("test"), data=self.serialize_response("test"))
        return response
        # TODO implement handling for other startgin and stopping training
        # event = request.event
        # if event == commons.START_ROUND:
        #     num_executors = request.num_executors

        #     if self.waiting_for_start:
        #         self.waiting_for_start = False

        #     executor_id = request.executor_id
        #     client_id = request.client_id
        #     logging.info(f"Received ping request from client {client_id}, executor {executor_id}")

        #     response = job_api_pb2.ServerResponse(event=commons.GL_ACK,
        #                                         meta=self.serialize_response("test"), data=self.serialize_response("test"))

        #     return response
        # elif event == commons.SHUT_DOWN:
        #     self.received_stop_request = True

    def client_ping(self, stub):
        """Ping the aggregator for new task
        """
        try:
            response = stub.CLIENT_PING(job_api_pb2.PingRequest(
                client_id=str(self.client_id),
                executor_id=str(self.executor_id)
            ))

            return response
        except:
            response = job_api_pb2.ServerResponse(event="Failed to connect to aggregator.",
                                                  meta=self.serialize_response("test"), data=self.serialize_response("test"))
            return response
        # self.dispatch_worker_events(response)

    def stop(self):
        self.grpc_server.stop(None)
        logging.info(f"Teminating client {self.client_id}")


if __name__ == "__main__":
    executor = Executor(parser.args)
    executor.run()
