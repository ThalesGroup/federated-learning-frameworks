# EasyFL_Remote_FL
This is a demonstration of using different docker containers to realize a communication between a server and 2 clients with EasyFL framework.

More details are present in the EasyFL [github page](https://github.com/EasyFL-AI/EasyFL/blob/master/docs/en/tutorials/remote_training.md). This Readme only synthetises the launch of EasyFL using docker containers.

## Prepare server and client's code

In the app folder, we prepare our server and client's code in different files: server.py, client1.py and client2.py and also the run.py (which allows to link these components together with a grpc connection and build the queries.)

## Build the docker image
Create a DockerFile that containes all required packages, including the easyfl, and run:

    docker build -t easyfl-image .

Warning : If you are behind a corporate proxy, you will need to pass the proxy variables


    docker build --build-arg  http_proxy=$HTTP_PROXY --build-arg  https_proxy=$HTTPS_PROXY --build-arg no_proxy=$NO_PROXY -t easyfl-image .

## Run docker containers with a docker network
We copy the created image_id and run the following command:

    docker network create easyfl_network
    docker run -d -it --network easyfl_network --name server easyfl-image
    docker run -d -it --network easyfl_network --name client1 easyfl-image
    docker run -d -it --network easyfl_network --name client2 easyfl-image


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


## Run Server and Clients

Be carefull, the app folder is copied inside the docker image during the build.
So changes made on the app folder of the repo will not impact the app folder of the containers. You will need delete the containers, rebuild the image and relaunch the containers to see the modification inside the containers.
Another way could be to mount the app folder when running the container, for example:

        docker run -d -it --network easyfl_network -v ./app:/my_app/ --name server easyfl-image

Server: 

```
    docker exec -it server bash
    cd /app
    python server.py
```

Client1:

```
    docker exec -it client1 bash
    cd /app
    python client1.py
```
    
Client2:

```
    docker exec -it client2 bash
    cd /app
    python client2.py
```


Launch the run: 

```
    docker exec -it server bash
    cd /app
    python run.py
```

now we can see that federated learning runs correctly on different containers!

