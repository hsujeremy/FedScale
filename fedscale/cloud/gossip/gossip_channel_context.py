import logging

import grpc

import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc

MAX_MESSAGE_LENGTH = 1*1024*1024*1024  # 1GB


class GossipClientConnections(object):
    """"Clients build connections to the cloud aggregator."""

    def __init__(self, aggregator_address, client_id, ports, base_port=18888):
        self.client_id = client_id
        self.ports = ports
        self.base_port = base_port
        self.aggregator_address = aggregator_address
        self.aggregator_channel = None
        self.aggregator_stub = None
        self.channels = []
        self.stubs = []

    def connect_to_servers(self):
        """
        Initialize all server connections. Assume that this list is static for
        the duration of training.
        """
        # TODO: what if one of the ports isn't open/used by an active client yet?
        # - Seems like the channel is created regardless, without throwing an error.
        # TODO: what if the clients have different host names? Need to store those too

        # Connect to aggregator/coordinator
        logging.info('%%%%%%%%%% Opening grpc connection to ' +
                     self.aggregator_address + ':29500 %%%%%%%%%%')
        channel = grpc.insecure_channel(
            '{}:{}'.format(self.aggregator_address, 29500),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ]
        )
        self.aggregator_channel = channel
        self.aggregator_stub = job_api_pb2_grpc.JobServiceStub(channel)

        # Connect to other clients
        for port in range(self.ports):
            if port == self.client_id:
                continue
            logging.info(
                f'%%%%%%%%%% Opening grpc connection to {self.aggregator_address} at port {port + 1} %%%%%%%%%%'
            )
            channel = grpc.insecure_channel(
                '{}:{}'.format(self.aggregator_address, port + 4001),
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ]
            )

            self.channels.append(channel)
            self.stubs.append(job_api_pb2_grpc.JobServiceStub(channel))

    def close_server_connection(self):
        logging.info(
            '%%%%%%%%%% Closing grpc connection to the aggregator %%%%%%%%%%')
        for channel in self.channels:
            channel.close()
