# Modified by .github/workflows/publish_docker.yml
ARG CUDA_IMAGE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM $CUDA_IMAGE

# Modified by .github/workflows/publish_docker.yml
ARG PYTHON_VERSION=3.10
# TODO: find a way to parse this from the CUDA_IMAGE
# with bash string manipulation: ${CUDA_IMAGE##*-}//.
ARG DISTRO=ubuntu2004

ARG USER_ID=1001
ARG GROUP_ID=1001
ARG USER_NAME=sandbox
ARG GROUP_NAME=sandbox
ARG HOME_DIR=/sandbox

COPY ./setup.py /tmp
COPY ./README.md /tmp
COPY ./substratools /tmp/substratools

# Update the CUDA Linux GPG Repository Key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80 \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/3bf863cc.pub

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
# install pip for python${PYTHON_VERSION}
 && python${PYTHON_VERSION} -m pip install --upgrade --no-cache-dir pip \
# needed for pillow
 && python${PYTHON_VERSION} -m pip install --upgrade --no-cache-dir setuptools \
# install essential datascience libraries
# latest versions compatible with python>=3.8
 && python${PYTHON_VERSION} -m pip install --no-cache-dir pillow==9.2.0 pandas==1.4.3 numpy==1.23.1 scikit-learn==1.1.1 lifelines==0.27.1 scipy==1.9.0 \
# install substratools
 && cd /tmp && python${PYTHON_VERSION} -m pip install --no-cache-dir . \
# clean the apt cache
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
# Link python and python 3 to python${PYTHON_VERSION}
 && if ! command -v python &> /dev/null; \
    then echo "python not found, creating alias" \
    && rm -f /usr/bin/python \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python; \
 fi \
 && if ! command -v python3 &> /dev/null; \
    then echo "python3 not found, creating alias" \
    && rm -f /usr/bin/python3 \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3; \
 fi \
 && if python --version | grep -q "Python ${PYTHON_VERSION}"; then echo "pyhton command is properly set"; else exit 1 ; fi \
 && if python3 --version | grep -q "Python ${PYTHON_VERSION}"; then echo "pyhton3 command is properly set"; else exit 1 ; fi \
 && if pip --version | grep -q "python ${PYTHON_VERSION}"; then echo "pip command is properly set"; else exit 1 ; fi \
 && if pip3 --version | grep -q "python ${PYTHON_VERSION}"; then echo "pip3 command is properly set"; else exit 1 ; fi

# Create default user, group, and home directory
RUN addgroup --gid ${GROUP_ID} ${GROUP_NAME}
RUN adduser --disabled-password --gecos "" --uid ${USER_ID} --gid ${GROUP_ID} --home ${HOME_DIR} ${USER_NAME}

ENV PYTHONPATH ${HOME_DIR}
WORKDIR ${HOME_DIR}
