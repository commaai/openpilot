FROM ubuntu:16.04

RUN apt-get update && apt-get install -y gcc-arm-none-eabi libnewlib-arm-none-eabi python python-pip gcc g++

RUN pip install pycrypto==2.6.1

COPY . /panda

WORKDIR /panda
