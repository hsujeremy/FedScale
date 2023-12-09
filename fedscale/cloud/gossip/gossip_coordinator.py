import collections
import logging
import os
import pickle
import time
from concurrent import futures

import grpc
import torch
import wandb

import fedscale.cloud.gossip.job_api_pb2 as job_api_pb2
import fedscale.cloud.gossip.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.logger.aggregator_logging as logger
from fedscale.cloud import commons
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.resource_manager import ResourceManager
from fedscale.cloud.gossip.gossip_channel_context import GossipClientConnections

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


class GossipCoordinator(job_api_pb2_grpc.JobServiceServicer):
    def __init__(self, args):
        # init aggregator loger
        logger.initiate_aggregator_setting()
        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu'
        )

        self.experiment_mode = args.experiment_mode

        self.resource_manager = ResourceManager(commons.SIMULATION_MODE)
        self.client_manager = ClientManager(args.sample_mode, args=args)

        # ======== Channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None
        print('Coordinator expecting {} executors'.format(args.num_executors))
        self.client_communicator = GossipClientConnections(
            args.ps_ip, -1, is_coordinator=True)

        # ======== Event Queue ========
        self.individual_client_events = {}  # Unicast
        self.server_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque()  # Broadcast

        # TODO determine if its useful for aggregating training/testing results
        # self.client_training_results = {}

        # ======== Runtime Information ========
        self.registered_executor_info = set()
        self.num_of_clients = 0
        self.model_update_size = 0

        # ======== Wandb ========
        if args.wandb_token != "":
            os.environ['WANDB_API_KEY'] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(project=f'fedscale-{args.job_name}',
                                name=f'aggregator{args.this_rank}-{args.time_stamp}',
                                group=f'{args.time_stamp}')
                self.wandb.config.update({
                    "num_participants": args.num_participants,
                    "data_set": args.data_set,
                    "model": args.model,
                    "gradient_policy": args.gradient_policy,
                    "eval_interval": args.eval_interval,
                    "rounds": args.rounds,
                    "batch_size": args.batch_size,
                    "use_cuda": args.use_cuda
                })
            else:
                logging.error("Warning: wandb has already been initialized")
            # self.wandb.run.name = f'{args.job_name}-{args.time_stamp}'
        else:
            self.wandb = None

    def run(self):
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file)
        self.init_control_communication()
        self.event_monitor()
        self.stop()

    def load_client_profile(self, file_path):
        """For Simulation Mode: load client profiles/traces

        Args:
            file_path (string): File path for the client profiles/traces

        Returns:
            dictionary: Return the client profiles/traces

        """
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {client_id: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def add_event_handler(self, client_id, event, meta, data):
        """ Due to the large volume of requests, we will put all events into a queue first.

        Args:
            client_id (int): The client id.
            event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
            meta (dictionary or string): Meta message for grpc communication, could be event.
            data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

        """
        self.server_events_queue.append((client_id, event, meta, data))

    def broadcast_aggregator_events(self, event):
        """Issue tasks (events) to aggregator worker processes by adding grpc request event
        (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.broadcast_events_queue.append(event)

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        logging.info(f"Initiating control plane communication ...")

        # TODO Check if we need the simulation mode that was in the original implmentation
        # num_of_executors = 0
        # for ip_numgpu in self.args.executor_configs.split("="):
        #     ip, numgpu = ip_numgpu.split(':')
        #     for numexe in numgpu.strip()[1:-1].split(','):
        #         for _ in range(int(numexe.strip())):
        #             num_of_executors += 1
        self.executors = list(range(self.args.num_executors))

        # initiate a server process
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

        logging.info(
            f'%%%%%%%%%% Opening aggregator server using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()
        self.client_communicator.connect_to_executors(self.args.num_executors)

    def client_register_handler(self, executor_id, info):
        """Triggered once receive new executor registration.

        Args:
            executor_id (int): Executor Id
            info (dictionary): Executor information

        """
        # TODO Figure out registering info with executors since now client = executor
        systemProfile = {'computation': 1.0, 'communication': 1.0}
        if self.client_profiles:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (executor_id + 1) % len(self.client_profiles)
            systemProfile = self.client_profiles.get(mapped_id, systemProfile)

        client_id = executor_id
        self.client_manager.register_client(
            executor_id, client_id, size=info['size'][0], speed=systemProfile)
        # self.client_manager.registerDuration(
        #     client_id,
        #     batch_size=self.args.batch_size,
        #     local_steps=self.args.local_steps,
        #     # upload_size=self.model_update_size,
        #     # download_size=self.model_update_size
        # )

        logging.info("Info of all feasible clients {}".format(
            self.client_manager.getDataInfo()))

    def executor_info_handler(self, executor_id, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.

        Args:
            executor_id (int): Executor Id
            info (dictionary): Executor information

        """
        self.registered_executor_info.add(executor_id)
        logging.info(
            f"Received executor {executor_id} information, {len(self.registered_executor_info)}/{len(self.executors)}")

        self.client_register_handler(executor_id, info)
        if len(self.registered_executor_info) == len(self.executors):
            logging.info("All clients received! Broadcasting start...")
            # TODO: also tell the executors how many executors in total there are
            self.broadcast_aggregator_events(commons.START_ROUND)

    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")
        try:
            while True:
                # Broadcast events to clients
                if len(self.broadcast_events_queue) > 0:
                    current_event = self.broadcast_events_queue.popleft()
                    self.dispatch_client_events(current_event)
                else:
                    # execute every 100 ms
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info(
                "KeyboardInterrupt: stopping the coordinator and all executors ...")
            self.dispatch_client_events(commons.SHUT_DOWN)

    def dispatch_client_events(self, event, clients=None):
        """Issue tasks (events) to clients

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
            clients (list of int): target client ids for event.

        """
        for i, stub in enumerate(self.client_communicator.stubs):
            try:
                response = stub.CLIENT_PING(job_api_pb2.PingRequest(
                    client_id=str(i),
                    executor_id=str(i),
                    event=event,
                    num_executors=str(self.args.num_executors)
                ))
            except:
                logging.info(f"Failed to ping client {i} with event {event}")

    def update_default_task_config(self):
        """Update the default task configuration after each round
        """
        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor, self.args.min_learning_rate)

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        last_round_avg_util = sum(
            self.stats_util_accumulator) / max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for client_id in self.round_stragglers:
            self.client_manager.register_feedback(client_id, last_round_avg_util,
                                                  time_stamp=self.round,
                                                  duration=self.virtual_client_clock[client_id]['computation'] +
                                                  self.virtual_client_clock[client_id]['communication'],
                                                  success=False)

        # update select participants
        self.sampled_participants = self.select_participants(
            cur_time=1, select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
        (clients_to_run, round_stragglers, virtual_client_clock, round_duration,
         flatten_client_duration) = self.tictak_client_tasks(
            self.sampled_participants, self.args.num_participants)

        logging.info(f"Selected participants to run: {clients_to_run}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clients_to_run)
        self.tasks_round = len(clients_to_run)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(
                self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants]
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.stats_util_accumulator = []
        self.update_default_task_config()

        # TODO Ping or send response to client that invoked round_completion_handler

    def select_participants(self, cur_time, select_num_participants, overcommitment=1.3):
        """Select clients for next round.

        Args:
            select_num_participants (int): Number of clients to select.
            overcommitment (float): Overcommit ratio for next round.

        Returns:
            list of int: The list of sampled clients id.

        """
        return sorted(self.client_manager.select_participants(
            int(select_num_participants * overcommitment),
            cur_time=cur_time),
        )

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            Tuple: (the List of clients to run, the List of stragglers in the round, a Dict of the virtual clock of each
            client, the duration of the aggregation round, and the durations of each client's task).

        """
        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            completionTimes = []
            completed_client_clock = {}
            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)

                exe_cost = self.client_manager.get_completion_time(client_to_run,
                                                                   batch_size=client_cfg.batch_size,
                                                                   local_steps=client_cfg.local_steps,
                                                                   upload_size=self.model_update_size,
                                                                   download_size=self.model_update_size)

                roundDuration = exe_cost['computation'] + \
                    exe_cost['communication']
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = exe_cost

            num_clients_to_collect = min(
                num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            workers_sorted_by_completion_time = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k])
            top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            stragglers = [sampledClientsReal[k]
                          for k in workers_sorted_by_completion_time[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            return (clients_to_run, stragglers,
                    completed_client_clock, round_duration,
                    completionTimes[:num_clients_to_collect])
        else:
            completed_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock,
                    1, completionTimes)

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

    def get_client_conf(self, client_id):
        """Training configurations that will be applied on clients,
        developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: TorchClient training config.

        """
        conf = {
            'learning_rate': self.args.learning_rate,
        }
        return conf

    def create_client_task(self, executor_id):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        # TODO FIgure out if executor/client distinction is still necessary
        train_config = None
        config = self.get_client_conf(executor_id)
        train_config = {'client_id': executor_id, 'task_config': config}
        return train_config, self.model_wrapper.get_weights()

    def get_test_config(self, client_id):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        return {'client_id': client_id}

    def get_shutdown_config(self, client_id):
        """Shutdown config for client, developers can further define personalized client config here.

        Args:
            client_id (int): TorchClient id.

        Returns:
            dictionary: Shutdown config for new task.

        """
        return {'client_id': client_id}

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
            logging.info(
                f"Detect new client: {executor_id}, executor info: {executor_info}")
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
        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        response_data = response_msg = commons.DUMMY_RESPONSE
        current_event = commons.DUMMY_EVENT

        if event == commons.GL_REQUEST_NEIGHBORS:
            # TODO Integrate get valid neighbors
            pass

        # client_event_queue = self.individual_client_events[executor_id]
        # if client_event_queue:
        #     current_event = client_event_queue.popleft()
        #     if current_event == commons.MODEL_TEST:
        #         response_msg = self.get_test_config(client_id)
        #     elif current_event == commons.SHUT_DOWN:
        #         response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        response = job_api_pb2.ServerResponse(event=current_event,
                                              meta=response_msg, data=response_data)
        if current_event != commons.DUMMY_EVENT:
            logging.info(
                f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")

        return response

    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        if self.wandb != None:
            self.wandb.finish()
        time.sleep(5)
