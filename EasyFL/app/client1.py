import easyfl

# Configurations for the remote client.
conf = {
    "is_remote": True,
    "local_port": 23000,
    "server_addr": "server:22999",
    "index": 0,
    'data': {'dataset':'cifar10', 'num_of_clients': 2}
}
# Initialize only the configuration.
easyfl.init(conf, init_all=False)
# Start remote client service.
# The remote client waits to be connected with the remote server.
easyfl.start_client()
