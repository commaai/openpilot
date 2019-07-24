FROM ubuntu:16.04

RUN apt-get update && apt-get install -y make python python-pip
COPY tests/safety/requirements.txt /panda/tests/safety/requirements.txt
RUN pip install -r /panda/tests/safety/requirements.txt
COPY . /panda
