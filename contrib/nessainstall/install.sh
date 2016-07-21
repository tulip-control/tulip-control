# install script for unix-like systems
#
# dependencies:
#	git
#
# configure:
#   - $CFG_FILE: your shell sources this at startup
#   - $INSTALL_LOC: installation will be there
#
# on ubuntu you must install dependencies required by python
# with apt-get as outlined here:
#	http://askubuntu.com/questions/101591/how-do-i-install-python-2-7-2-on-ubuntu
#
# caution: may need to apply
#	chmod g-wx,o-wx ~/.python-eggs
# due to cvxopt behavior

# debian/ubuntu dependencies:
#
# sudo apt-get install \
#   build-essential libreadline-gplv2-dev libncursesw5-dev \
#   libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev \
#   libbz2-dev libatlas-base-dev libatlas-dev gfortran libpng-dev

# e.g.: ~/.bashrc if using bash,
# or:   ~/.tcshrc if using csh
CFG_FILE=~/.bashrc
PYTHON_VERSION=2.7.11

# location to create directory "libraries"
# will contain python, ATLAS, LAPACK, glpk, gr1c
INSTALL_LOC=~

install_graphviz=true
install_glpk=true
install_atlas=false
tulip_develop=true # if 1, then tulip installed in develop mode

#------------------------------------------------------------
# do not edit below unless you know what you are doing
DOWNLOAD_LOC=$INSTALL_LOC/temp_downloads
TMPLIB=$INSTALL_LOC/libraries
TMPBIN=$TMPLIB/bin

# exit at first error
set -e

# create libraries to install things
if [ -d "$TMPLIB" ]; then
	echo "$TMPLIB already exists"
else
	echo "$TMPLIB does not exist: mkdir"
	mkdir $TMPLIB
fi
if [ -d "$DOWNLOAD_LOC" ]; then
	echo "$DOWNLOAD_LOC already exists"
else
	echo "$DOWNLOAD_LOC does not exist: mkdir"
	mkdir $DOWNLOAD_LOC
fi

# check required commands exist
#
# snippet from:
#    http://wiki.bash-hackers.org/scripting/style
my_needed_commands="sed curl tar gcc gfortran bison flex"
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
if [ -f "$CFG_FILE" ]; then
	echo "$CFG_FILE already exists"
else
	# sed cannot edit files w/o any lines
	echo "creating a new $CFG_FILE"
	echo "# auto-created by tulip installation script" >> $CFG_FILE
fi

# export before non-interactive shell exists
# remove the pattern search if not needed
sed -i '/# If not running interactively,/i \
export PATH='"$TMPBIN"':$PATH \
export LD_LIBRARY_PATH='"$TMPLIB"'/lib' $CFG_FILE
source $CFG_FILE

#------------------------------------------------------------
# install ATLAS with LAPACK
if [ "$install_atlas" = "true" ]; then
	cd $DOWNLOAD_LOC

	curl -LO http://sourceforge.net/projects/math-atlas/files/Stable/3.10.1/atlas3.10.1.tar.bz2
	curl -LO http://www.netlib.org/lapack/lapack-3.5.0.tgz

	tar xjf atlas3.10.1.tar.bz2 # unpack only ATLAS
	cd ATLAS
	if [ -d "LinuxBuild" ]; then
		echo "LinuxBuild dir already exists"
	else
		mkdir LinuxBuild
	fi
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
# install glpk
if [ "$install_glpk" = "true" ]; then
	if [ -f "$TMPBIN/glpsol" ]; then
		echo "GLPK already installed locally: skip"
	else
		echo "GLPK not found locally: install"
		cd $DOWNLOAD_LOC

		# cvxopt is incompatible with newer versions
		curl -LO http://ftp.gnu.org/gnu/glpk/glpk-4.48.tar.gz
		tar xzf glpk-4.48.tar.gz
		cd glpk-4.48
		./configure --prefix=$TMPLIB
		make
		make check # should return no errors
		make install

		# make sure this glpsol is used by bash later
		hash glpsol
	fi
fi

#------------------------------------------------------------
# install graphviz dot (needed by pydot)
if [ "$install_graphviz" = "true" ]; then
	if [ -f "$TMPBIN/dot" ]; then
		echo "GraphViz already installed locally: skip"
	else
		echo "GraphViz not found locally: install"
		cd $DOWNLOAD_LOC

		# cvxopt is incompatible with newer versions
		git clone https://github.com/ellson/graphviz.git
		cd graphviz
		./autogen.sh
		./configure --prefix=$TMPLIB

		make
		make check
		make install
	fi
fi

#------------------------------------------------------------
# install gr1c
#
# https://tulip-control.github.io/gr1c/md_installation.html

if [ -f "$TMPBIN/gr1c" ]; then
	echo "GR1C already installed locally: skip"
else
	echo "GR1C not found locally: install"
	cd $DOWNLOAD_LOC

	if [ -d "gr1c" ]; then
		echo "gr1c already cloned"
	else
		echo "cloning gr1c"
		git clone https://github.com/tulip-control/gr1c.git
	fi
	cd gr1c

	# install CUDD
	if [ -d "extern" ]; then
		echo "directory 'extern' already exists"
	else
		mkdir extern
	fi
	cd extern
	curl -LO ftp://vlsi.colorado.edu/pub/cudd-2.5.0.tar.gz
	tar -xzf cudd-2.5.0.tar.gz
	cd cudd-2.5.0
	make

	# build and install gr1c
	cd ../..
	make all
	make check
	make install prefix=$TMPBIN # doesn't include: grpatch, grjit

	hash gr1c
fi

#------------------------------------------------------------
# install python
if [ -f "$TMPBIN/python" ]; then
	echo "Python already installed locally: skip"
else
	echo "Python not found locally: install"
	cd $DOWNLOAD_LOC

	curl -LO http://www.python.org/ftp/python/\
	$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
	tar xzf Python-$PYTHON_VERSION.tgz
	cd Python-$PYTHON_VERSION
	./configure --prefix=$TMPLIB --enable-shared
	make
	make install

	hash python
fi

# verify python is correct
if [ $(command -v python) != "$TMPBIN/python" ]; then
	echo "local python not found"
	exit 1
fi

#------------------------------------------------------------
# install pip
if [ -f "$TMPBIN/pip" ]; then
	echo "Pip already installed locally: skip"
else
	echo "Pip not found locally: install"
	cd $DOWNLOAD_LOC

	curl -LO https://bootstrap.pypa.io/get-pip.py
	python get-pip.py

	hash pip
fi

# verify python is correct
if [ $(command -v pip) != "$TMPBIN/pip" ]; then
	echo "local pip not found"
	exit 1
fi

#------------------------------------------------------------
# install python packages
pip install numpy
pip install scipy

#------------------------------------------------------------
# install cvxopt

# env vars for building cvxopt
if [ "$install_atlas" = "true" ]; then
	echo "config CVXOPT for local ATLAS in Mac"

	# https://github.com/cvxopt/cvxopt/blob/master/setup.py#L60
	export CVXOPT_BLAS_LIB="satlas,tatlas,atlas"
	export CVXOPT_BLAS_LIB_DIR=$TMPLIB/lib
	export CVXOPT_BLAS_EXTRA_LINK_ARGS="-nostdlib"
	export CVXOPT_LAPACK_LIB="[]"
else
	if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
		echo "config CVXOPT for global ATLAS in Linux"

		export CVXOPT_BLAS_LIB="f77blas,cblas,atlas,gfortran"
		export CVXOPT_BLAS_LIB_DIR="/usr/lib"
		#export CVXOPT_BLAS_EXTRA_LINK_ARGS="[]"
		export CVXOPT_LAPACK_LIB="lapack"
	fi
fi

if [ "$install_glpk" = "true" ]; then
	echo "config CVXOPT to build GLPK extension"

	export CVXOPT_BUILD_GLPK=1
	export CVXOPT_GLPK_LIB_DIR=$TMPLIB/lib
	export CVXOPT_GLPK_INC_DIR=$TMPLIB/include
fi

# tar the edited package and install
if $(python -c "import cvxopt.glpk" &> /dev/null); then
	echo "CVXOPT already installed with GLPK locally: skip"
else
	echo "CVXOPT not found locally: install"
	cd $DOWNLOAD_LOC

	if [ -d "cvxopt" ]; then
		echo "cvxopt already cloned"
	else
		echo "cloning cvxopt"
		git clone https://github.com/cvxopt/cvxopt.git
	fi
	cd cvxopt

	python setup.py install
fi

#------------------------------------------------------------
# install polytope
if $(python -c "import polytope" &> /dev/null); then
	echo "polytope already installed locally: skip"
else
	echo "polytope not found locally: install"
	cd $DOWNLOAD_LOC
	if [ -d "polytope" ]; then
		echo "polytope already cloned"
	else
		echo "cloning polytope"
		git clone https://github.com/tulip-control/polytope.git
	fi
	cd polytope
	python setup.py install
fi

#------------------------------------------------------------
# install tulip
cd $DOWNLOAD_LOC
if [ -d "tulip-control" ]; then
	echo "tulip already cloned"
else
	echo "clonign tulip"
	git clone https://github.com/tulip-control/tulip-control.git
fi
cd tulip-control

if [ "$tulip_develop" = "true" ]; then
	python setup.py develop
else
	python setup.py install
fi
python run_tests.py --fast

#------------------------------------------------------------
# optional
pip install matplotlib

# skip virtualenvwrapper: fragile to install
#pip install virtualenvwrapper
#sed -i '$ a export VIRTUALENVWRAPPER_VIRTUALENV='"$TMPBIN"'/virtualenv-2.7' $CFG_FILE
#sed -i '$ a source '"$TMPBIN"'/virtualenvwrapper.sh' $CFG_FILE
#source $CFG_FILE
