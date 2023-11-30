import collections
import copy
import math
import random
import threading
import time
from concurrent import futures
from random import Random

import grpc
import numpy as np
import torch
import wandb

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.rl_client import RLClient
from fedscale.cloud.execution.tensorflow_client import TensorflowClient
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.fllibs import *
from fedscale.cloud.gossip.gossip_channel_context import ClientConnections
from fedscale.cloud.internal.tensorflow_model_adapter import \
    TensorflowModelAdapter
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset

"""
Make a server for each client 
"""
MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


class Executor(job_api_pb2_grpc.JobServiceServicer):
    def __init__(self, args, num_iterations=100, client_id=0, ports=10):
        self.num_iterations = num_iterations
        self.model = None
        self.client_id = client_id
        self.executor_id = client_id

        self.training_sets = self.test_dataset = None

        # init model weights here? training config contains model weights under "model"
        # model weights are stored under self.model_adapter.get_weights
        self.model_adapter = self.get_client_trainer(
            args).get_model_adapter(init_model())

        self.args = args
        self.num_executors = args.num_executors

        # ======== Env Information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)

        # ======== Event Queue ========
        self.individual_client_events = {}  # Unicast
        self.receive_events_queue = collections.deque()
        self.send_events_queue = collections.deque()
        # ======== Model and Data ========
        self.training_sets = self.test_dataset = None
        self.model_wrapper = None
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.model_weights = None
        self.temp_model_path = os.path.join(
            logger.logDir, 'model_'+str(args.this_rank)+".npy")
        self.last_saved_round = 0

        self.client_manager = self.init_client_manager(args=args)

        # ======== channels ========
        # TODO Make connections to other executors
        self.client_communicator = ClientConnections(
            args.ps_ip, client_id, ports)

        # ======== runtime information ========
        self.collate_fn = None
        self.round = 0
        self.start_run_time = time.time()
        self.received_stop_request = False

        if args.wandb_token != "":
            os.environ['WANDB_API_KEY'] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
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

        # simulation mode
        # num_of_executors = 0
        # for ip_numgpu in self.args.executor_configs.split("="):
        #     ip, numgpu = ip_numgpu.split(':')
        #     for numexe in numgpu.strip()[1:-1].split(','):
        #         for _ in range(int(numexe.strip())):
        #             num_of_executors += 1
        # self.executors = list(range(num_of_executors))

        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )

        job_api_pb2_grpc.add_JobServiceServicer_to_server(
            self, self.grpc_server)

        port = '[::]:{}'.format(self.client_id)
        logging.info(
            f'%%%%%%%%%% Opening aggregator server using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()
        self.client_communicator.connect_to_server()

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
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
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
    def select_neighbors(self, min_replies, cur_time=0, buffer_factor=2):
        """Randomly select neighbors to request weights from. This should be
        some order of magnitude higher than the minimum amount of clients we
        want to receive weight updates from (e.g., if we want at least 5
        replies, then we should select 10 neighbors).
        """

        clients_online = self.client_manager.getFeasibleClients(cur_time)
        rng = Random()
        rng.seed(233)
        rng.shuffle(clients_online)
        client_len = min(min_replies * buffer_factor, len(clients_online)-1)
        return clients_online[:client_len]

    def train(self, config):

        # TODO should deal with a set number of iterations at a time
        # config is usually passed as a message, but config can be a set 
        # dict each time 

        # few batches
        # for i in range(10):
        #     response = self.client_communicator.stubs[client_id].CLIENT_EXECUTE_COMPLETION(
        #         job_api_pb2.CompleteRequest(
        #             client_id=str(client_id), executor_id=self.executor_id,
        #             event=commons.CLIENT_TRAIN, status=True, msg=None,
        #             meta_result=None, data_result=None
        #         )
        #     )
        # client_id, train_config = config['client_id'], config['task_config']
        # assert 'model' in config
        # train_res = self.training_handler(
        #     client_id=client_id, conf=train_config, model=config['model'])

        # use torchclient trainer
        pass

    def training_handler(self, conf, model):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        self.model_adapter.set_weights(model)
        conf.tokenizer = tokenizer
        client_data = self.training_sets if self.args.task == "rl" else \
            select_dataset(self.client_id, self.training_sets,
                           batch_size=conf.batch_size, args=self.args,
                           collate_fn=self.collate_fn
                           )
        
        # TODO figure out if current trainer handles set number of iterations at a time
        train_res = self.get_client_trainer(self.args).train(
            client_data=client_data, model=self.model_adapter.get_model(), conf=conf)

        return train_res

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
        train_res = None

        # TODO fix how training works
        # Should train a few batches at a time and check for intermitent events between
        for i in range(self.num_iterations):
            if i > 0 and i % 10 == 0: 
                # TODO after a set number of iterations (e.g. 10), request weights from clients
                # TODO: in the case of error 
                neighbors = self.select_neighbors(min_replies=0.7)
                counter = 0 
                while True:
                    # get stubs from client communicator 
                    for neighbor in neighbors:
                        stub = self.client_communicator.stubs[neighbor]
                        res = stub.REQUEST_WEIGHTS(
                            job_api_pb2.WeightRequest(
                                client_id=f"{self.client_id}",
                                executor_id=f"{self.executor_id}"
                            )
                        )

                        res_client_id = res.client_id
                        res_exeuctor_id = res.executor_id
                        # TODO: check for error
                        counter += 1                        
                    if counter >= len(neighbors):
                        break

                # Wait for neighbors to send back weights
                logging.info(f"Client {self.client_id} waiting to receive weights...")
                while len(self.receive_events_queue) < int(0.5 * len(neighbors)):
                    time.sleep(0.1)
    
                logging.info(f"Client {self.client_id} aggregating weights...")
                self.model_weights = self.model_wrapper.get_weights()
                # check queue
                while self.receive_events_queue:
                    client_id, executor_id, weights = self.receive_events_queue.popleft()

                    # process event
                    self.client_completion_handler(weights)


            # check queue
            # if we have any requests to send out weights 
            while self.send_events_queue:
                client_id, current_event, meta, data = self.send_events_queue.popleft()
                # process event
                if current_event == commons.GL_REQUEST_WEIGHTS and train_res:
                    self.client_send_weights_handler(
                        client_id, train_res)
            train_res = self.train()

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
        # Send model to client
        future_call = self.client_communicator.stubs[client_id].UPLOAD_WEIGHTS.future(
            job_api_pb2.UploadWeightRequest(client_id=str(self.client_id),
                                            executor_id=self.executor_id,
                                            weights=self.serialize_response(train_res)
                                        ))
        future_call.add_done_callback(
            lambda _response: self.dispatch_worker_events(_response.result()))

    def dispatch_receive_events(self, request):
        """Add new events to worker queue for sending out weights.
        
        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to events_queue.

        """
        self.receive_events_queue.append(request)

    def dispatch_send_events(self, request):
        """Add new events to worker queue for receiving weights.

        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to events_queue.

        """
        self.send_events_queue.append(request)

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

    def client_completion_handler(self, weights):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        # TODO Check if these is necessary
        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.register_feedback(results['client_id'], results['utility'],
                                              auxi=math.sqrt(
                                                  results['moving_loss']),
                                              time_stamp=self.round,
                                              duration=self.virtual_client_clock[results['client_id']]['computation'] +
                                              self.virtual_client_clock[results['client_id']
                                                                        ]['communication']
                                              )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()

        self.model_in_update += 1
        self.update_weight_aggregation(weights)

        self.update_lock.release()

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def executor_info_handler(self, executorId, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        # TODO Figure out how much of this logic we actually need
        self.registered_executor_info.add(executorId)
        logging.info(
            f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout
        if self.experiment_mode == commons.SIMULATION_MODE:

            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executorId, info)
                # start to sample clients
                self.round_completion_handler()
        else:
            # In real deployments, we need to register for each client
            self.client_register_handler(executorId, info)
            if len(self.registered_executor_info) == len(self.executors):
                self.round_completion_handler()

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
            # TODO determine how to calculate self.tasks_round
            self.model_weights = list(
                map(lambda x: x/self.tasks_round, self.model_weights))
            self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))

    def _is_last_result_in_round(self):
        return self.model_in_update == self.tasks_round

    def client_register(self):
        """Register the executor information to the aggregator
        """
        start_time = time.time()
        for stub in self.client_communicator.stubs:
            while time.time() - start_time < 180:
                try:
                    response = stub.CLIENT_REGISTER(
                        job_api_pb2.RegisterRequest(
                            client_id=self.executor_id,
                            executor_id=self.executor_id,
                            executor_info=self.serialize_response(
                                self.report_executor_info_handler())
                        )
                    )
                    self.dispatch_worker_events(response)
                    break
                except Exception as e:
                    logging.warning(
                        f"Failed to connect to aggregator {e}. Will retry in 5 sec.")
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
        response_data = response_msg = commons.GL_ACK_RESPONSE
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
        self.dispatch_receive_events(request)

        event = commons.GL_ACK
        response_data = response_msg = commons.GL_ACK_RESPONSE
        response = job_api_pb2.ServerResponse(event=event,
                                              meta=response_msg, data=response_data)

        return response


if __name__ == "__main__":
    executor = Executor(parser.args)
    executor.run()
