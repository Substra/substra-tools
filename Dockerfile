FROM nvidia/cuda:9.0-base

RUN apt-get update; apt-get install -y build-essential libssl-dev  python3 python3-dev python3-pip
RUN pip3 install pillow

ADD ./setup.py /tmp
ADD ./substratools /tmp/substratools
RUN cd /tmp && pip3 install -e .

RUN mkdir -p /sandbox/opener

WORKDIR /sandbox
