FROM ubuntu:latest
RUN  apt-get update
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN pip3 install grpcio
RUN python3 -m pip install flwr
RUN pip install -U scikit-learn
RUN pip install tensorflow
RUN pip install seaborn

ADD app /app/

EXPOSE 22222

