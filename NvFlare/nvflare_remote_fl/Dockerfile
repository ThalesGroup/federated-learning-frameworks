#ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.02-py3
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${PYTORCH_IMAGE}

ARG NVF_VERSION=2.3
ENV NVF_BRANCH=${NVF_VERSION}


RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install nvflare
RUN apt-get update
RUN apt-get install -y iputils-ping
WORKDIR /workspace/
RUN git clone https://github.com/NVIDIA/NVFlare.git --branch ${NVF_BRANCH} --single-branch NVFlare
RUN echo "y" | nvflare poc --prepare -n 2
EXPOSE 22222 
