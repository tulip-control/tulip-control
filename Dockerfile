# following example at https://github.com/binder-examples/minimal-dockerfile

FROM ubuntu:20.04

RUN apt-get -y update
RUN apt-get -y install \
    python3  \
    python3-pip \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libgmp-dev \
    libmpfr-dev \
    graphviz \
    libglpk-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-regex-dev \
    libboost-test-dev \
    libeigen3-dev \
    libginac-dev \
    z3 \
    libz3-dev \
    python3-z3 \
    libhwloc-dev

RUN pip3 install -U pip

RUN pip install --no-cache notebook jupyterlab

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}
