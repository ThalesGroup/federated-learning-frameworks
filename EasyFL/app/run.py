import easyfl
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.communication import grpc_wrapper
from easyfl.registry.vclient import VirtualClient

server_addr = "server:22999"
config = {
    "data": {"dataset": "cifar10"},
    "model": "lenet",
    "test_mode": "test_in_client",
}
# Initialize configurations.
easyfl.init(config, init_all=False)
# Initialize the model, using the configured 'lenet'
model = easyfl.init_model()

# Construct gRPC request 
stub = grpc_wrapper.init_stub(grpc_wrapper.TYPE_SERVER, server_addr)
request = server_pb.RunRequest(model=codec.marshal(model))
# The request contains clients' addresses for the server to communicate with the clients.
clients = [VirtualClient("1", "client1:23000", 0), VirtualClient("2", "client2:23000", 1)]
for c in clients:
    request.clients.append(server_pb.Client(client_id=c.id, index=c.index, address=c.address))
# Send request to trigger training.
response = stub.Run(request)
result = "Success" if response.status.code == common_pb.SC_OK else response
print(result)
