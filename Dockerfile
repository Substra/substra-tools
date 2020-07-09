FROM nvidia/cuda:9.2-base-ubuntu18.04

RUN apt-get update; apt-get install -y build-essential libssl-dev python3 python3-dev python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install pillow==6.1.0 pandas==0.24.2 numpy==1.16.4 scikit-learn==0.21.2 lifelines==0.22.1 scipy==1.2.1

ADD ./setup.py /tmp
ADD ./README.md /tmp
ADD ./substratools /tmp/substratools
RUN cd /tmp && pip install .

RUN mkdir -p /sandbox
ENV PYTHONPATH /sandbox

WORKDIR /sandbox
