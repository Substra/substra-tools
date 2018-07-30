FROM frolvlad/alpine-python-machinelearning

RUN apk add py3-pillow

ADD ./setup.py /tmp
ADD ./substratools /tmp/substratools
RUN pip install --upgrade pip && cd /tmp && pip install -e . 

RUN mkdir -p /sandbox/opener && touch /sandbox/opener/__init__.py


WORKDIR /sandbox
