FROM nvidia/cuda:9.0-base

RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev wget && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt && \
    wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz && \
    tar -xvf Python-3.6.9.tgz && rm Python-3.6.9.tgz
RUN cd /opt/Python-3.6.9 ; ./configure ; make ; make install

RUN pip3 install --upgrade pip
RUN pip3 install pillow pandas numpy sklearn lifelines

ADD ./setup.py /tmp
ADD ./substratools /tmp/substratools
RUN cd /tmp && pip install .

RUN mkdir -p /sandbox/opener

WORKDIR /sandbox
