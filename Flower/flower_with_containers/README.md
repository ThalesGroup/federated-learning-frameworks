This is a demonstration of using different docker containers to realize the communication between server and clients (we use 2 clients in this tuto) with Flower framework.

## 1- We prepare our server and client's code in different files: server.py, client.py and client2.py, and place them in the app folder
We have to choose a specific port to export in the docker file, here we set 22222.

## 2- We build a docker image that containes all required packages:
Create a docker file and run: 

        docker build -t flower-image .


Warning : If you are behind a corporate proxy, you will need to pass the proxy variables


        docker build --build-arg  http_proxy=$HTTP_PROXY --build-arg  https_proxy=$HTTPS_PROXY,server --build-arg no_proxy=$NO_PROXY,server -t flower-image .
    
## 3- We create a docker network and launch 3 containers on it : 

        docker network create flower_network
        docker run -d -it --network flower_network --name server flower-image
        docker run -d -it --network flower_network --name client1 flower-image
        docker run -d -it --network flower_network --name client2 flower-image


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
       
## 4- Finally, we run each python file of the folder app in each container separately, now we can see that federated learning runs correctly on isolated machines!

Be carefull, the app folder is copied inside the docker image during the build.
So changes made on the app folder of the repo will not impact the app folder of the containers. You will need delete the containers, rebuild the image and relaunch the containers to see the modification inside the containers.
Another way could be to mount the app folder when running the container

        docker run -d -it --network flower_network -v ./app:/my_app/ --name server flower-image

Server: 

```
    docker exec -it server bash
    source .venv/bin/activate
    cd /app
    python server.py
```

Client1:

```
    docker exec -it client1 bash
    source .venv/bin/activate
    cd /app
    python client1.py
```
    
Client2:

```
    docker exec -it client2 bash
    source .venv/bin/activate
    cd /app
    python client2.py
```
