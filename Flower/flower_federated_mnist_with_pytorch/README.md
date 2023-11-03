This is my implementation of real-federated Mnist_CNN in pytorch based on flower framework.

python version: 3.9

        pip -r install requirements_flower_env3.9.txt

- We use an IID partition of datasets (train, valid (to test locally clients's models) and test (to test the global model via the server)).

- Run the server and clients each in a terminal:

        python server.py

        python client1.py

        python client2.py
