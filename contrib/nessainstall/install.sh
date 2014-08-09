# install script for unix-like systems
#
# dependencies:
#	git
#
# configure:
#   - $CFG_FILE: your shell sources this at startup
#   - $INSTALL_LOC: installation will be there
#
# on ubuntu you must install dependencies required by python 2.7.8
# with apt-get as outlined here:
#	http://askubuntu.com/questions/101591/how-do-i-install-python-2-7-2-on-ubuntu

# e.g.: ~/.bash_profile if using bash,
# or:   ~/.tcshrc if using csh
export CFG_FILE=~/.bash_profile

# location to create directory "libraries"
# will contain python, ATLAS, LAPACK, glpk, gr1c
INSTALL_LOC=~

install_atlas=0
#------------------------------------------------------------
# do not edit below unless you know what you are doing
TMPLIB=$INSTALL_LOC/libraries
TMPBIN=$TMPLIB/bin

# create libraries to install things
mkdir $TMPLIB

# check required commands exist
#
# snippet from:
#    http://wiki.bash-hackers.org/scripting/style
my_needed_commands="sed curl tar"
missing_counter=0
for needed_command in $my_needed_commands; do
  if ! hash "$needed_command" >/dev/null 2>&1; then
    printf "Command not found in PATH: %s\n" "$needed_command" >&2
    ((missing_counter++))
  fi
done

if ((missing_counter > 0)); then
  printf "Minimum %d commands are missing in PATH, aborting\n" "$missing_counter" >&2
  exit 1
fi

# "export" works in bash
# "setenv" works in csh
sed -i '$ a export PATH='"$TMPBIN"':$PATH' $CFG_FILE
sed -i '$ a export LD_LIBRARY_PATH='"$TMPLIB"'/lib' $CFG_FILE
source $CFG_FILE

cd $INSTALL_LOC

#------------------------------------------------------------
# install ATLAS with LAPACK
if [ -o install_atlas ]; then
	curl -O http://sourceforge.net/projects/math-atlas/files/Stable/3.10.1/atlas3.10.1.tar.bz2
	curl -O http://www.netlib.org/lapack/lapack-3.5.0.tgz
	
	tar xjf atlas3.10.1.tar.bz2 # unpack only ATLAS
	cd ATLAS
	mkdir LinuxBuild
	cd LinuxBuild
	../configure -b 64 --prefix=$TMPLIB --shared \
		--with-netlib-lapack-tarfile=../lapack-3.5.0.tgz
	cd ..
	make build # this takes forever...
	make check
	make ptcheck # should be no errors for make check, ptcheck, time
	make time
	make install
else
	echo "Skipping ATLAS installation, set install_atlas to enable this."
fi

#------------------------------------------------------------
# install python
if [ -f "$TMPBIN/python" ]; then
	echo "Python already installed, skipping"
else
	curl -O http://www.python.org/ftp/python/2.7.6/Python-2.7.6.tgz
	tar xzf Python-2.7.6.tgz
	cd Python-2.7.6
	./configure --prefix=$TMPLIB --enable-shared
	make
	make install
fi

#------------------------------------------------------------
# install pip
curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
python get-pip.py
source $CFG_FILE

#------------------------------------------------------------
# install python packages
pip install numpy
pip install scipy
pip install matplotlib
pip install ply

# pyparsing needed as pydot dependency
# downgrade pyparsing
/usr/bin/yes | pip uninstall pyparsing
pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709

# install latest pydot version
pip install http://pydot.googlecode.com/files/pydot-1.0.28.tar.gz

pip install networkx
#------------------------------------------------------------
# optional python installs
pip install ipython

pip install virtualenvwrapper
sed -i '$ a export VIRTUALENVWRAPPER_VIRTUALENV='"$TMPLIB"'/bin/virtualenv-2.7' $CFG_FILE
sed -i '$ a source '"$TMPLIB"'/bin/virtualenvwrapper.sh' $CFG_FILE
source $CFG_FILE

# downgrade pyparsing
#------------------------------------------------------------
# install glpk
if [ -f "$TMPLIB/bin/glpsol" ]; then
	echo "glpk installed: skipping installing it"
else
	# cvxopt is incompatible with newer versions
	curl -O http://ftp.gnu.org/gnu/glpk/glpk-4.48.tar.gz
	tar xzf glpk-4.48.tar.gz
	cd glpk-4.48
	./configure --prefix=$TMPLIB
	make
	make check # should return no errors
	make install
fi
#------------------------------------------------------------
# install cvxopt
curl -O http://abel.ee.ucla.edu/src/cvxopt-1.1.6.tar.gz
tar xzf cvxopt-1.1.6.tar.gz

# rename archive
mv cvxopt-1.1.6.tar.gz cvxopt-orig.tar.gz
cd cvxopt-1.1.6

# hack: edit setup.py
sed -i "5 c BLAS_LIB_DIR = '"$TMPLIB"/lib'" setup.py
sed -i "8 c BLAS_LIB = ['satlas', 'tatlas', 'atlas']" setup.py
sed -i "9 c LAPACK_LIB = []" setup.py
sed -i "10 c BLAS_EXTRA_LINK_ARGS = ['-nostdlib']" setup.py
sed -i "36 c BUILD_GLPK = 1" setup.py
sed -i "39 c GLPK_LIB_DIR = '"$TMPLIB"/lib'" setup.py
sed -i "42 c GLPK_INC_DIR = '"$TMPLIB"/include'" setup.py

# tar the edited package and install
cd ..
tar czf cvxopt-1.1.6.tar.gz cvxopt-1.1.6/
pip install cvxopt-1.1.6.tar.gz

#------------------------------------------------------------
# install gr1c
#
# http://slivingston.github.io/gr1c/md_installation.html

# download requires Caltech IP Address (change this to building it)
curl -O http://vehicles.caltech.edu/private/snapshots/nessa/gr1c/gr1c-0.7.3.tar.gz

# untar and copy all binaries to your bin folder
tar xzf gr1c-0.7.3.tar.gz
cd gr1c-0.7.3
cp gr1c rg grpatch $TMPLIB/bin 

#------------------------------------------------------------
# install tulip
git clone https://github.com/tulip-control/tulip-control.git 
cd tulip-control
python setup.py install

# use this instead if you want to edit code in the git repository
#python setup.py develop
