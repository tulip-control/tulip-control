#!/usr/bin/env bash


# Prepare for running test collection "full".


# install `cvxopt.glpk`
export CVXOPT_BUILD_GLPK=1
#
python -c "import setup; setup.install_cvxopt()"
python -c "import cvxopt.glpk"
#
# install `dd.cudd`
pip uninstall --yes dd
pip download --no-dependencies dd --no-binary dd
tar xzf dd-*.tar.gz
cd dd-*/
export CUDD_VERSION=3.0.0
export CUDD_GZ=cudd-${CUDD_VERSION}.tar.gz
#
# download
curl -sSL https://sourceforge.net/projects/\
cudd-mirror/files/${CUDD_GZ}/download > ${CUDD_GZ}
#
# checksum
echo "b8e966b4562c96a03e7fbea23972958\
7d7b395d53cadcc39a7203b49cf7eeb69  ${CUDD_GZ}" | \
    shasum -a 256 -c -
#
# unpack
tar -xzf ${CUDD_GZ}
python -c 'from download import make_cudd; make_cudd()'
python setup.py install --cudd
cd tests/
python -c 'import dd.cudd'
cd ../..
#
# install optional Python packages
pip install gr1py
#
# install `gr1c`
export GR1C_VERSION=0.13.0
export GR1C_GZ=gr1c-${GR1C_VERSION}-Linux_x86-64.tar.gz
# download
curl -sSLO https://github.com/\
tulip-control/gr1c/releases/download/\
v${GR1C_VERSION}/${GR1C_GZ}
# checksum
echo "1d45ca69d6acbf84ae6170de60b6c69\
073dffd3a6130c6213419e401d9d5c470  ${GR1C_GZ}" | \
    shasum -a 256 -c -
# unpack
tar -xzf ${GR1C_GZ}
export PATH=`pwd`/gr1c-${GR1C_VERSION}-Linux_x86-64:$PATH
# store values to use in later steps for environment variables
# <https://docs.github.com/en/actions/reference/
#     workflow-commands-for-github-actions#
#     setting-an-environment-variable>
echo "`pwd`/gr1c-${GR1C_VERSION}-Linux_x86-64" >> $GITHUB_PATH
# diagnostic information
which gr1c
#
# install `lily`
./extern/get-lily.sh
which perl
perl --version
pwd
export PERL5LIB=`pwd`/Lily-1.0.2
export PATH=`pwd`/Lily-1.0.2:$PATH
# store values to use in later steps for environment variables
echo "PERL5LIB=`pwd`/Lily-1.0.2" >> $GITHUB_ENV
# <https://docs.github.com/en/actions/reference/
#     workflow-commands-for-github-actions#adding-a-system-path>
echo "`pwd`/Lily-1.0.2" >> $GITHUB_PATH
# diagnostic information
echo $PATH
which lily.pl
#
# install `slugs`
./extern/get-slugs.sh
export PATH=`pwd`/slugs-repo/src:$PATH
# store values to use in later steps for environment variables
echo "`pwd`/slugs-repo/src" >> $GITHUB_PATH
# diagnostic information
which slugs
#
# install Python requirements for
# development testing
pip install gitpython
#
# install `stormpy` and its dependencies
./extern/get-stormpy.sh
