This is my implementation of real-federated Mnist_CNN in pytorch based on flower framework.

We are using python3.12, with the requirements in flower_with_containers/requirements.txt

- We use an IID partition of datasets (train, valid (to test locally clients's models) and test (to test the global model via the server)).

- Run the server and clients each in a terminal:

        python server.py

        python client1.py

        python client2.py
