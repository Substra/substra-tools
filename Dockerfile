# Modified by .github/workflows/publish_docker.yml
ARG CUDA_IMAGE=nvidia/cuda:9.2-base-ubuntu18.04
FROM $CUDA_IMAGE

# Modified by .github/workflows/publish_docker.yml
ARG PYTHON_VERSION=3.7

ARG USER_ID=1001
ARG GROUP_ID=1001
ARG USER_NAME=sandbox
ARG GROUP_NAME=sandbox
ARG HOME_DIR=/sandbox

COPY ./setup.py /tmp
COPY ./README.md /tmp
COPY ./substratools /tmp/substratools

# need the 3 first lines to get python${PYTHON_VERSION} on all ubuntu versions
RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
# needed for tzdata, installed by python3.9
 && export DEBIAN_FRONTEND=noninteractive \
 && export TZ=Europe/Moscow \
# install essential apps
 && apt-get install -y \
    build-essential \
    libssl-dev \
    python${PYTHON_VERSION} \
    libpython${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
# needed for pillow
 && apt-get install -y libjpeg-dev zlib1g-dev \
# clean the apt cache
 && rm -rf /var/lib/apt/lists/ \
 # link default python, python3, pip, pip3 to python${PYTHON_VERSION}
 && rm -f /usr/bin/python \
 && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
 && rm -f /usr/bin/python3 \
 && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
 && python --version \
 && python3 --version \
# install pip for python${PYTHON_VERSION}
 && python -m pip install --upgrade --no-cache-dir pip \
 && pip --version \
 && pip3 --version \
# needed for pillow
 && pip install --upgrade --no-cache-dir setuptools \
# install essential datascience libraries
# latest versions compatible with python>=3.7
 && pip install --no-cache-dir pillow==9.0.1 pandas==1.3.5 numpy==1.21.5 scikit-learn==1.0.2 lifelines==0.26.4 scipy==1.7.3 \
# install substratools
 && cd /tmp && pip install --no-cache-dir . \
# clean the apt cache
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create default user, group, and home directory
RUN addgroup --gid ${GROUP_ID} ${GROUP_NAME}
RUN adduser --disabled-password --gecos "" --uid ${USER_ID} --gid ${GROUP_ID} --home ${HOME_DIR} ${USER_NAME}

ENV PYTHONPATH ${HOME_DIR}
WORKDIR ${HOME_DIR}
