import numpy as np
import copy
import time
import threading
import collections
from random import Random
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.fllibs import *
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn

"""
Make a server for each client 
"""

class Executor(object):
    def __init__(self, num_iterations=100, client_id=0):
        self.num_iterations = num_iterations
        self.model = None
        self.client_id = client_id

        self.training_sets = self.test_dataset = None


        # ======== Event Queue =======
        self.individual_client_events = {}  # Unicast
        self.events_queue = collections.deque()

        # ======== Model and Data ========
        self.model_wrapper = None
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.model_weights = None
        self.temp_model_path = os.path.join(
            logger.logDir, 'model_'+str(args.this_rank)+".npy")
        self.last_saved_round = 0
    
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

        clients_online = ClientManager.getFeasibleClients(cur_time)
        rng = Random()
        rng.seed(233)
        rng.shuffle(clients_online)
        client_len = min(min_replies * buffer_factor, len(clients_online)-1)
        return clients_online[:client_len]

    def train(self, config): 
        client_id, train_config = config['client_id'], config['task_config']
        assert 'model' in config


        # use torchclient trainer 
        client = TorchClient()
        pass

    def training_handler(self, conf, model, client):
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
        train_res = client.train(
            client_data=client_data, model=self.model_adapter.get_model(), conf=conf)

        return train_res

    def run(self): 
        """
            after each training loop,
                check # of incoming requests: 
                    clients to send weights
        """

        for i in range(self.num_iterations):
            # check queue 
            if len(self.events_queue) > 0:
                client_id, current_event, meta, data = self.events_queue.popleft()

                # process event
                if current_event == commons.UPLOAD_MODEL:
                    self.client_upload_model_handler(client_id)
            self.train()

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

    # def event_listener(self):
    #     """Listen to events from other clients.
    #     """
    #     logging.info("Start monitoring events ...")

    #     while True:
    #         # Handle queued events
    #         if len(self.events_queue) > 0:
    #             client_id, current_event, meta, data = self.events_queue.popleft()

                
    #             if current_event == commons.UPLOAD_MODEL:
    #                 self.client_completion_handler(
    #                     self.deserialize_response(data))
    #                 if len(self.stats_util_accumulator) == self.tasks_round:
    #                     self.round_completion_handler()

    #             # TODO modify to accept testing data
    #             elif current_event == commons.MODEL_TEST:
    #                 self.testing_completion_handler(
    #                     client_id, self.deserialize_response(data))

    #             else:
    #                 logging.error(f"Event {current_event} is not defined")

    #         else:
    #             # execute every 100 ms
    #             time.sleep(0.1)

    def client_upload_model_handler(self, client_id):
        """Uploads model to client at client_id.

        Args:
            client_id (int): The client id.

        """
        # Get client config
        client_config = self.client_manager.get_client_config(client_id)

        # Get client model
        client_model = self.model_wrapper.get_weights()

        # Serialize model
        serialized_model = self.serialize_response(self.model_wrapper.get_weights())

        # Send model to client
        future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
            job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                        event=commons.UPLOAD_MODEL, status=True, msg=None,
                                        meta_result=None, data_result=self.serialize_response(train_res)
                                        ))
        future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))
    
    def client_completion_handler(self, results):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

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
                                                       self.virtual_client_clock[results['client_id']]['communication']
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
            self.model_weights = [weight + update_weights[i] for i, weight in enumerate(self.model_weights)]
        if self._is_last_result_in_round():
            self.model_weights = [np.divide(weight, self.tasks_round) for weight in self.model_weights]
            self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))

