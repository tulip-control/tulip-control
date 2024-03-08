#!/bin/bash -e
#
# dependency support script for stormpy interface of TuLiP
#
# The following `apt` packages are needed for
# building the binaries for `carl` and `storm`
# by following the commands listed below:
#
# autoconf
# build-essential
# cmake
# doxygen
# git
# libboost-all-dev
# libeigen3-dev
# libginac-dev
# libtool
# python2
# texinfo
# z3
# libz3-dev
# python3-z3
#
#
# initially created 2020-10-05

## CArL https://smtrat.github.io/carl/
# if [ ! -f carl.tar.xz ]
# then
#     curl -L -sS -o carl.tar.xz https://sourceforge.net/projects/cudd-mirror/files/carl.tar.xz/download
# fi
# echo 'cdcc8726aca2d1d1486c5746b5f4b729de2267aaf1dedfa72db9c835e67daa36  carl.tar.xz' | \
#     shasum -a 256 -c -
# xz -d carl.tar.xz && tar xf carl.tar
# export PREPLACE=$(echo $PWD | tail -c +2)
# export PREPLACE=$(echo $PREPLACE | sed 's/\//\\\//g')
# sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" carl/carlConfig.cmake
# sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" carl/carlTargets.cmake
# export carl_DIR=${PWD}/carl

if [ ! -f carl.tar.gz ]
then
    curl -L -sS -o carl.tar.gz \
    https://github.com/tulip-control/data/releases/download/stormpy_dependencies/carl.tar.gz
fi
echo 'd3be70201b852c4cb717c162268ef5c74fdfe79f8c6ae49bd2fabd7542bf0418  carl.tar.gz' | \
    shasum -a 256 -c -
mkdir -p extern
tar -xzf carl.tar.gz -C extern
export PREPLACE=$(echo $PWD | tail -c +2)
export PREPLACE=$(echo $PREPLACE | sed 's/\//\\\//g')
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" extern/carl/build/carlConfig.cmake
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" extern/carl/build/carlTargets.cmake
export carl_DIR=${PWD}/extern/carl/build


## to build CArL from source:
# git clone -b master14 --depth 1 https://github.com/smtrat/carl.git
# pushd carl
# mkdir build
# cd build
# cmake -DUSE_CLN_NUMBERS=ON -DUSE_GINAC=ON -DTHREAD_SAFE=ON ..
# make lib_carl
# popd

# to create the tarball that is uploaded to GitHub:
#
# Edit the files:
#   carl/build/carlConfig.cmake
#   carl/build/carlTargets.cmake
# to replace the paths specific to the machine
# with the string `PREFIXTOREPLACE`, to be used
# by `sed` as above.
# Then:
#
# tar -czf carl.tar.gz \
#     carl/src \
#     carl/build/carlConfig.cmake \
#     carl/build/carlTargets.cmake \
#     carl/build/libcarl.so.14.20 \
#     carl/build/src


## pycarl https://github.com/moves-rwth/pycarl/
if [ ! -f pycarl.tgz ]
then
    curl -sSL -o pycarl.tgz https://github.com/moves-rwth/pycarl/archive/refs/tags/2.2.0.tar.gz
fi
echo '64885a0b0abf13aaed542a05ef8e590194b13626dcd07209ec55b41f788c6a56  pycarl.tgz' | \
    shasum -a 256 -c -
tar xzf pycarl.tgz
pushd pycarl-2.2.0
python3 setup.py develop
popd


## Storm https://www.stormchecker.org/
# if [ ! -f storm.tar.xz ]
# then
#     curl -L -sS -o storm.tar.xz https://sourceforge.net/projects/cudd-mirror/files/storm.tar.xz/download
# fi
# echo '25ce93853da67162922e49c37cccba9b2fcb73ac11facbab93b26690957ddc7d  storm.tar.xz' | \
#     shasum -a 256 -c -
# xz -d storm.tar.xz && tar xf storm.tar
# sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" storm/stormConfig.cmake
# sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" storm/stormTargets.cmake
# export storm_DIR=${PWD}/storm

if [ ! -f storm.tar.gz ]
then
    curl -L -sS -o storm.tar.gz \
    https://github.com/tulip-control/data/releases/download/stormpy_dependencies/storm.tar.gz
fi
echo '1bd6af73b5a833d4577340605f91a4d7c180954b030dc11dd5e51b0544db426e  storm.tar.gz' | \
    shasum -a 256 -c -
mkdir -p extern
tar -xzf storm.tar.gz -C extern
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" extern/storm/build/stormConfig.cmake
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" extern/storm/build/stormTargets.cmake
export storm_DIR=${PWD}/extern/storm/build


## to build Storm from source:
# git clone -b stable --depth 1 https://github.com/moves-rwth/storm.git
# pushd storm
# mkdir build
# cd build
# cmake ..
# make
# make check
# cd bin
# export PATH=`pwd`:$PATH
# popd

# to create the tarball that is uploaded to GitHub:
#
# Edit:
#   storm/build/stormConfig.cmake
#   storm/build/stormTargets.cmake
# to replace the paths specific to the machine
# with the string `PREFIXTOREPLACE`, to be used
# by `sed` as above.
# Then:
#
#
# tar -czf storm.tar.gz \
#     storm/resources/3rdparty/cpptemplate \
#     storm/resources/3rdparty/exprtk \
#     storm/resources/3rdparty/gmm-5.2/include \
#     storm/resources/3rdparty/l3pp \
#     storm/resources/3rdparty/modernjson/src \
#     storm/resources/3rdparty/sylvan/src \
#     storm/build/stormConfig.cmake \
#     storm/build/stormTargets.cmake \
#     storm/build/bin \
#     storm/build/lib \
#     storm/build/include \
#     storm/build/resources


## stormpy https://moves-rwth.github.io/stormpy/
if [ ! -f stormpy-stable.tgz ]
then
    curl -sSL -o stormpy-stable.tgz https://github.com/moves-rwth/stormpy/archive/refs/tags/1.8.0.tar.gz
fi
echo '3c59fb8bed69637e7a1e96b9372198a3428b305520108baa3df627a35940762d  stormpy-stable.tgz' | \
    shasum -a 256 -c -
tar xzf stormpy-stable.tgz
pushd stormpy-1.8.0
python3 setup.py develop
popd
