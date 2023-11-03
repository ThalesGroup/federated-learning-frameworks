## THis FLOWER-based implementation presents a simple Federated Learning application of an Intrusion Detection System with an MLP model for defending against Cyberattacks using distributed data.

The dataset used is part of the [IEC 60870-5-104 Intrusion Detection Dataset](http://zenodo.org/record/7108614#.YzGaDtJBwUE) -- DOI: [10.1109/TII.2021.3093905](http://doi.org/10.1109/TII.2021.3093905) -- published by [ITHACA – University of Western Macedonia](http://ithaca.ece.uowm.gr/).

We choosed to use the IDDataset/Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs/tests_cic_15 path which contains both train and test datasets (It's up to you to choose the part you want, but you'll need to adapt it with a custom preprocessing that you'll also have to implement.)

We are using two clients and a server, with the FedAvg strategy. 

    python server.py
    python client1.py
    python client2.py
