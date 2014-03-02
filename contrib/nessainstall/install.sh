# install script for unix-like systems
# assumes git is installed on you machine

# edit these to your preference

# e.g.: ~/.bash_profile if using bash,
# or:   ~/.tcshrc if using csh
export CFG_FILE=./test.txt

# where the `libraries` directory will go,
# will contain python, ATLAS, LAPACK, glpk, gr1c
INSTALL_LOC=~/

#------------------------------------------------------------
# do not edit below unless you know what you are doing
TMPLIB=$INSTALL_LOC/libraries

# create libraries to install things
mkdir ~/libraries

# in bash use export,
# in csh use setenv
sed '$ a\
	export PATH="$TMPLIB"/bin:$PATH' $CFG_FILE
sed '$ a\
	export LD_LIBRARY_PATH="$TMPLIB"/lib' $CFG_FILE

source $CFG_FILE
cd $INSTALL_LOC

#------------------------------------------------------------
# install ATLAS with LAPACK
wget http://sourceforge.net/projects/math-atlas/files/Stable/3.10.1/atlas3.10.1.tar.bz2
wget http://www.netlib.org/lapack/lapack-3.5.0.tgz

tar xjf atlas3.10.1.tar.bz2 # unpack only ATLAS
cd ATLAS
mkdir LinuxBuild
cd LinuxBuild
../configure -b 64 --prefix=$TMPLIB --shared \
	--with-netlib-lapack-tarfile=../lapack-3.5.0.tgz
make build # this takes forever...
make check
make ptcheck # should be no errors for make check, ptcheck, time
make time
make install

#------------------------------------------------------------
# install python
wget http://www.python.org/ftp/python/2.7.6/Python-2.7.6.tgz
tar xzf Python-2.7.6.tgz
cd Python-2.7.6
./configure --prefix=$TMPLIB --enable-shared
make
make install

#------------------------------------------------------------
# install pip
wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py
python get-pip.py
source $CFG_FILE

#------------------------------------------------------------
# install python packages
pip install numpy
pip install scipy
pip install matplotlib
pip install ply
pip install virtualenvwrapper # optional for your convenience

# downgrade pyparsing
pip uninstall pyparsing
pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709

# install latest pydot version
pip install http://pydot.googlecode.com/files/pydot-1.0.28.tar.gz

pip install networkx

#------------------------------------------------------------
# install glpk

# cvxopt is incompatible with newer versions
wget http://ftp.gnu.org/gnu/glpk/glpk-4.48.tar.gz
tar xzf glpk-4.48.tar.gz
cd glpk-4.48
./configure --prefix=$TMPLIB
make
make check # should return no errors
make install

#------------------------------------------------------------
# install cvxopt
wget http://abel.ee.ucla.edu/src/cvxopt-1.1.6.tar.gz
tar xzf cvxopt-1.1.6.tar.gz

# rename archive
mv cvxopt-1.1.6.tar.gz cvxopt-orig.tar.gz
cd cvxopt-1.1.6

# hack: edit setup.py
sed '5 c\
	BLAS_LIB_DIR = '"$TMPLIB"/lib' setup.py
sed '8 c\
	BLAS_LIB = [''satlas'', ''tatlas'', ''atlas'']' setup.py
sed '9 c\
	LAPACK_LIB = []' setup.py
sed '10 c\
	BLAS_EXTRA_LINK_ARGS = [''-nostdlib'']' setup.py
sed '36 c\
	BUILD_GLPK = 1' setup.py
sed '39 c\
	GLPK_LIB_DIR = ''"$TMPLIB"/lib''' setup.py
sed '42 c\
	GLPK_INC_DIR = ''"$TMPLIB"/include''' setup.py

# tar the edited package and install
cd ..
tar czf cvxopt-1.1.6.tar.gz cvxopt-1.1.6/
pip install cvxopt-1.1.6.tar.gz

#------------------------------------------------------------
# install gr1c

# download requires Caltech IP Address (change this to building it)
wget http://vehicles.caltech.edu/private/snapshots/nessa/gr1c/gr1c-0.6.6-95ec9c6.tar.gz

# untar and copy all binaries to your bin folder
tar xzf gr1c-0.6.6-95ec9c6.tar.gz
cd gr1c-0.6.6-95ec9c6
cp gr1c rg grpatch $TMPLIB/bin 

#------------------------------------------------------------
# install tulip
git clone https://github.com/tulip-control/tulip-control.git 
cd tulip-control
python setup.py install

# use this instead if you want to edit code in the git repository
#python setup.py develop
