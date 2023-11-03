import easyfl

# Configurations for the remote server.
conf = {
    "is_remote": True, 
    "local_port": 22999, 
    "server":{"clients_per_round": 2, 'random_selection': False}
}
# Initialize only the configuration.
easyfl.init(conf, init_all=False)
# Start remote server service.
# The remote server waits to be connected with the remote client.
easyfl.start_server()
