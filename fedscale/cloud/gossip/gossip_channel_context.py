import logging

import grpc

import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc

MAX_MESSAGE_LENGTH = 1*1024*1024*1024  # 1GB


class ClientConnections(object):
    """"Clients build connections to the cloud aggregator."""

    def __init__(self, aggregator_address, client_id, ports, base_port=18888):
        self.client_id = client_id
        self.ports = ports
        self.base_port = base_port
        self.aggregator_address = aggregator_address
        self.channels = []
        self.stubs = []

    def connect_to_servers(self):
        """
        Initialize all server connections. Assume that this list is static for
        the duration of training.
        """
        for port in range(self.ports):
            logging.info('%%%%%%%%%% Opening grpc connection to ' +
                        self.aggregator_address + 'at port ' +  port + ' %%%%%%%%%%')
            channel = grpc.insecure_channel(
                '{}:{}'.format(self.aggregator_address, port),
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ]
            )

            self.channels.append(channel)
            self.stubs.append(job_api_pb2_grpc.JobServiceStub(self.channel))

    def close_sever_connection(self):
        logging.info(
            '%%%%%%%%%% Closing grpc connection to the aggregator %%%%%%%%%%')
        for channel in self.channels:
            channel.close()
