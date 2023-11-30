import grpc
import concurrent.futures as futures
import logging

def create_server(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    port = '[::]:{}'.format(port)

    print(f'%%%%%%%%%% Opening aggregator server using port {port} %%%%%%%%%%')

    server.add_insecure_port(port)
    server.start()
    server.wait_for_termination()

if __name__ == "__main__": 
    create_server(2)


