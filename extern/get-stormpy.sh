#!/bin/bash -e
#
# dependency support script for stormpy interface of TuLiP
# initially created 2020-10-05

## CArL https://smtrat.github.io/carl/
if [ ! -f carl.tar.xz ]
then
    curl -L -sS -o carl.tar.xz https://sourceforge.net/projects/cudd-mirror/files/carl.tar.xz/download
fi
echo 'cdcc8726aca2d1d1486c5746b5f4b729de2267aaf1dedfa72db9c835e67daa36  carl.tar.xz' | \
    shasum -a 256 -c -
xz -d carl.tar.xz && tar xf carl.tar
export PREPLACE=$(echo $PWD | tail -c +2)
export PREPLACE=$(echo $PREPLACE | sed 's/\//\\\//g')
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" carl/carlConfig.cmake
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" carl/carlTargets.cmake
export carl_DIR=${PWD}/carl


## to build CArL from source:
# git clone -b master14 --depth 1 https://github.com/smtrat/carl.git
# pushd carl
# mkdir build
# cd build
# cmake -DUSE_CLN_NUMBERS=ON -DUSE_GINAC=ON -DTHREAD_SAFE=ON ..
# make lib_carl
# popd


## pycarl https://github.com/moves-rwth/pycarl/
if [ ! -f pycarl.tgz ]
then
    curl -sSL -o pycarl.tgz https://github.com/moves-rwth/pycarl/archive/2.0.4.tar.gz
fi
echo '751debb79599d697046ed89638503f946a35f316864bf405acc743df15173947  pycarl.tgz' | \
    shasum -a 256 -c -
tar xzf pycarl.tgz
pushd pycarl-2.0.4
python3 setup.py develop
popd


## Storm https://www.stormchecker.org/
if [ ! -f storm.tar.xz ]
then
    curl -L -sS -o storm.tar.xz https://sourceforge.net/projects/cudd-mirror/files/storm.tar.xz/download
fi
echo '25ce93853da67162922e49c37cccba9b2fcb73ac11facbab93b26690957ddc7d  storm.tar.xz' | \
    shasum -a 256 -c -
xz -d storm.tar.xz && tar xf storm.tar
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" storm/stormConfig.cmake
sed -i "s/PREFIXTOREPLACE/$PREPLACE/g" storm/stormTargets.cmake
export storm_DIR=${PWD}/storm


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


## stormpy https://moves-rwth.github.io/stormpy/
if [ ! -f stormpy-stable.tgz ]
then
    curl -sSL -o stormpy-stable.tgz https://github.com/moves-rwth/stormpy/archive/1.6.2.tar.gz
fi
echo '78f94f5d367b69c438b0442c24e74ca62887e751ea067e69c0d98cf32a12219c  stormpy-stable.tgz' | \
    shasum -a 256 -c -
tar xzf stormpy-stable.tgz
pushd stormpy-1.6.2
python3 setup.py develop
popd
