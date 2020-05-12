from ubuntu:16.04

RUN apt-get update && apt-get install -y libzmq3-dev clang wget git autoconf libtool curl make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.7.3
RUN pyenv global 3.7.3
RUN pyenv rehash

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

ENV PYTHONPATH=/project

# TODO: Add tag to cereal
RUN git clone https://github.com/commaai/cereal.git /project/cereal
RUN /project/cereal/install_capnp.sh

WORKDIR /project

COPY SConstruct .
COPY . /project/opendbc

RUN scons -c && scons -j$(nproc)
