FROM python:slim

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install grpcio grpcio_tools

COPY builder/ /source

WORKDIR /source
# can run indexing from here if necessary as well if necessary
CMD bash download_files.sh && build_protos.sh