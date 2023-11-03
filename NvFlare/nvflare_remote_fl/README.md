# NVFlare_Remote_FL

This is a demonstration of using different machines to realize a communication between a server and 2 clients with NVFlare framework. Please make sure to follow all steps in detail. In this video, the environment is already prepared, we show you the way to run FL through docker containers. 

## Build the docker image
Create a DockerFile that containes all required packages, including the nvflare one, we have to choose a specific port to export in the docker file, here we set 22222, and the run:

    docker build -t nvflare-image .


Warning : If you are behind a corporate proxy, you will need to pass the proxy variables


    docker build --build-arg  http_proxy=$HTTP_PROXY --build-arg  https_proxy=$HTTPS_PROXY --build-arg no_proxy=$NO_PROXY -t nvflare-image .


## Using a docker network 

    docker network create nvflare_network

### Run docker containers
We run the following command(we need to do this in 3 different command lines in order to have 3 different containers):

    docker run -d -it --network nvflare_network --name server nvflare-image
    docker run -d -it --network nvflare_network --name client1 nvflare-image
    docker run -d -it --network nvflare_network --name client2 nvflare-image 



Warning : If you are behind a corporate proxy, you will need to pass the proxy variables 
to your containers so that they can access internet (to download a dataset for example).
This can be done by creating/setting the ~/.docker/config.json file

```
{
        "proxies":{
                "default":{
                        "httpsProxy": "your_https_proxy",
                        "httpProxy": "your_http_proxy",
                        "noProxy": "your_no_proxy,localhost,server"
                }
        }
}
```

However, you don't want the proxy to be used for the communication between the nodes,
so you will need to add "server" (or the server container IP) to the NO_PROXY variable (as shown above).


### Get the IP of each node and lunch interactive shells

Get the ip addresses of the containers by executing:

    docker network inspect nvflare_network  

Note the IP address of the server's container, for example: 172.19.66.2, which we must include in the .json configuration file for clients, server and admin.

Lunch an interactive shell session on the 3 already-running containers by running:

    docker exec -it container_name bash

The execution of "nvflare poc --prepare -n 2" commande written in the DockerFile will create a directory in /tmp directory named poc 

copy nvflare examples to the /tmp/nvflare/poc/admin/transfer directory of the server to be able to submit jobs for client and server execution


```
    docker exec -it server bash
    mkdir /tmp/nvflare/poc/admin/transfer
    cp -r /workspace/NVFlare/examples/ /tmp/nvflare/poc/admin/transfer/
```

### Run Server, Clients and admin 

Finally, we run each bash script (of the server and clients) in each container separately, for the admin one, we run it in the server container:

Server: 

```
    docker exec -it server bash
    cd /tmp/nvflare/poc/server/startup
    ./start.sh
```

Client1:

```
    docker exec -it client1 bash
    cd /tmp/nvflare/poc/site-1/startup
    ./start.sh server:8002:8003
```
    
Client2:

```
    docker exec -it client2 bash
    cd /tmp/nvflare/poc/site-2/startup
    ./start.sh server:8002:8003
```

Admin:


```
    docker exec -it server bash
    mkdir /tmp/nvflare/poc/admin/transfer
    cp -r /workspace/NVFlare/examples/ /tmp/nvflare/poc/admin/transfer/
    cd /tmp/nvflare/poc/admin/startup
    ./fl_admin.sh
```


Submit a nvflare job by specifying its path, for example we run the hello-pt example:

    submit_job /tmp/nvflare/poc/admin/transfer/examples/hello-world/hello-pt/jobs/hello-pt

Verify the status of the clients by running

    check_status client

now we can see that federated learning runs correctly on docker containers!

First each client will load the data from internet (you might have trouble here if the containers cannot access internet)

The next step is to make it works without using the docker network but using container IP to simulate different computers


## Using the containers IP to simulate different computers

```
    docker run -d -it --name server nvflare-image
```


get the server container IP

`docker inspect server | grep IPAddress`    

You need to have this IP put in the  "noProxy"
 of ~/.docker/config.json for the client to be able to communicate with th server

```
    docker run -d -it --name client1 nvflare-image
    docker run -d -it --name client2 nvflare-image 
```

Server: 

```
    docker exec -it server bash
    cd /tmp/nvflare/poc/server/startup
    ./start.sh
```

Client1:

```
    docker exec -it client1 bash
    cd /tmp/nvflare/poc/site-1/startup
    ./start.sh <SERVER_IP>:8002:8003
```
    
Client2:

```
    docker exec -it client2 bash
    cd /tmp/nvflare/poc/site-2/startup
    ./start.sh <SERVER_IP>:8002:8003
```

Admin:

```
    docker exec -it server bash
    mkdir /tmp/nvflare/poc/admin/transfer
    cp -r /workspace/NVFlare/examples/ /tmp/nvflare/poc/admin/transfer/
    cd /tmp/nvflare/poc/admin/startup
    ./fl_admin.sh
```


Submit a nvflare job by specifying its path, for example we run the hello-pt example:

    submit_job /tmp/nvflare/poc/admin/transfer/examples/hello-world/hello-pt/jobs/hello-pt

Verify the status of the clients by running

    check_status client

now we can see that federated learning runs correctly on docker containers using their IP directly!

