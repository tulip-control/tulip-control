# docker build -t tulip:latest .
#
# The tag `tulip:latest` can be changed or omitted. The main intent
# for this Dockefile is to provide a basis for other Dockerfiles that
# install TuLiP from source code. To this end, the design here
# deviates somewhat from the common Dockerfile pattern of using the
# local sourcetree in the Docker image. Instead, it clones the
# (remote) tulip-control repository.
#
# After the image is built, it can be used interactively, e.g.,
#
#   docker run -i -t tulip:latest
#   source PYt/bin/activate
#   python -c 'import tulip; print(tulip.__version__)'
FROM debian:stable

RUN apt-get update && apt-get -y install \
    curl \
    build-essential \
    git \
    python-yaml

WORKDIR /root
RUN git clone https://github.com/tulip-control/tulip-control.git
COPY tulip-in-a-box.py /root/tulip-control/contrib/

WORKDIR /root/tulip-control
RUN python contrib/tulip-in-a-box.py --no-sudo
