
FROM python:3.9

RUN python3.9 -m pip install --upgrade pip
RUN pip install grpcio torch torchvision
RUN pip install grpcio-tools numpy omegaconf requests tqdm

RUN pip install easyfl

ADD app /app/

RUN echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" >> ~/.bashrc

RUN echo "y" | apt install git

