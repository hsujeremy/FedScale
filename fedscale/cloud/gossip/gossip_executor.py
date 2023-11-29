import collections
import copy
import math
import random
import threading
import time
import grpc

from concurrent import futures
from random import Random

import numpy as np
import torch
import wandb

import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.client_manager import ClientManager
import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
from fedscale.cloud.gossip.gossip_channel_context import ClientConnections
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.fllibs import *
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset

"""
Make a server for each client 
"""
MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB

# TODO: implement CLIENT_REGISTER, CLIENT_PING, CLIENT_EXECUTE_COMPLETION

class Executor(job_api_pb2_grpc.JobServiceServicer):
    def __init__(self, args, num_iterations=100, client_id=0, ports=10):
        self.num_iterations = num_iterations
        self.model = None
        self.client_id = client_id

        self.training_sets = self.test_dataset = None

        # init model weights here? training config contains model weights under "model"
        # model weights are stored under self.model_adapter.get_weights
        self.client = TorchClient(args)
        self.model_adapter = self.client.get_model_adapter(init_model())

        self.args = args
        self.num_executors = args.num_executors

        # ======== Event Queue =======
        self.individual_client_events = {}  # Unicast
        self.events_queue = collections.deque()

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

    def select_neighbors(self, cur_time, min_replies, buffer_factor=2):
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
        train_res = self.client.train(
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

        self.training_sets, self.testing_sets = self.init_data()
        train_res = None

        for i in range(self.num_iterations):
            if i % 10 == 0:
                while True:
                    # check queue
                    if len(self.events_queue) > 0:
                        for i in range(len(self.events_queue)):
                            client_id, current_event, meta, data = self.events_queue.popleft()

                            # process event
                            if current_event == commons.UPLOAD_MODEL and train_res:
                                self.client_upload_model_handler(client_id, train_res)
                        break
                    ## else, ping client
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

    def client_upload_model_handler(self, client_id, train_res):
        """Uploads model to client at client_id.

        Args:
            client_id (int): The client id.
            train_res (dictionary): The results from training.

        """
        # Send model to client
        future_call = self.client_communicator.stubs[client_id].CLIENT_EXECUTE_COMPLETION.future(
            job_api_pb2.CompleteRequest(client_id=str(self.client_id), executor_id=self.executor_id,
                                        event=commons.UPLOAD_MODEL, status=True, msg=None,
                                        meta_result=None, data_result=self.serialize_response(train_res)
                                        ))
        future_call.add_done_callback(
            lambda _response: self.dispatch_worker_events(_response.result()))

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

    def client_completion_handler(self, results):
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
        self.update_weight_aggregation(results)

        self.update_lock.release()

    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.

        Args:
            results (dict): The results from a client.        
        """
        update_weights = results['update_weight']
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]
        if self._is_first_result_in_round():
            self.model_weights = update_weights
        else:
            self.model_weights = [weight + update_weights[i]
                                  for i, weight in enumerate(self.model_weights)]
        if self._is_last_result_in_round():
            self.model_weights = [
                np.divide(weight, self.tasks_round) for weight in self.model_weights]
            self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))

    def _is_first_result_in_round(self):
        return self.model_in_update == 1

    def _is_last_result_in_round(self):
        return self.model_in_update == self.tasks_round
    
    def CLIENT_REGISTER(self, request, context):
        """FL TorchClient register to the aggregator

        Args:
            request (RegisterRequest): Registeration request info from executor.

        Returns:
            ServerResponse: Server response to registeration request

        """

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
    
    def CLIENT_PING(self, request, context):
        """Handle client ping requests

        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            current_event = self.individual_client_events[executor_id].popleft()
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    executor_id)
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN)
            elif current_event == commons.MODEL_TEST:
                response_msg = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                response_data = self.model_wrapper.get_weights()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(event=current_event,
                                              meta=response_msg, data=response_data)
        if current_event != commons.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")

        return response

    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task.

        Args:
            request (CompleteRequest): Complete request info from executor.

        Returns:
            ServerResponse: Server response to job completion request

        """

        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event == commons.UPLOAD_MODEL:
            # TODO prob not right, need to fix by adding to a queue or something
            self.client_completion_handler(data_result)
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")

        return self.CLIENT_PING(request, context)


if __name__ == "__main__":
    executor = Executor(parser.args)
    executor.run()