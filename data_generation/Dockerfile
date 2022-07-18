FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    wget vim \
    software-properties-common \
    locales \
    unzip \
    openjdk-11-jdk

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en

RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get install -y python3 \
    python3-pip \
    python-setuptools

RUN pip3 install --upgrade pip

# Install Rust for M1 Compatibility
RUN apt-get install -y curl
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY data_generation/requirements.txt /source/requirements.txt
RUN pip3 install -r /source/requirements.txt

COPY data_generation/ /source

WORKDIR /source
CMD PYTHONPATH=/shared python3 -u main.py