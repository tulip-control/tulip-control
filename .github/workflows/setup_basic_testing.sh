#!/usr/bin/env bash


# Prepare for testing `tulip`.


# install APT packages
sudo apt update
sudo apt install \
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
# install dependencies from PyPI
pip install \
    --ignore-installed \
    --upgrade \
        pip \
        setuptools \
        wheel
pip install \
    --upgrade \
    --only-binary=numpy,scipy \
        numpy \
        scipy
pip install dd \
    --no-binary dd
# install `tulip`
python setup.py sdist
pip install dist/tulip-*.tar.gz
# install test dependencies
pip install pytest
# diagnostics
dot -V
set -o posix
echo "Exported environment variables:"
export -p
