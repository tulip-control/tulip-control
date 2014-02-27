Installs Tulip on Nessa. Hopefully, this is still easier than installing Tulip
on Windows.



1) Create libraries folder and permanently add to path

	mkdir ~/libraries
	mkdir ~/libraries/bin
	mkdir ~/libraries/include
	mkdir ~/libraries/lib
	mkdir ~/libraries/share

   Add the lines below to ~/.tcshrc:

   	export PATH /home/username/libraries/bin:$PATH
	export LD_LIBRARY_PATH /home/username/libraries/lib

   Reload .tcshrc:

   	source ~/.tcshrc



2) Install ATLAS with LAPACK

	a. Download ATLAS 3.10.1: http://sourceforge.net/projects/math-atlas

	b. Download LAPACK 3.5: http://www.netlib.org/lapack/#_lapack_version_3_5_0

	c. Untar the ATLAS archive, but not the LAPACK archive:

			tar xjf atlas3.10.1.tar.bz2

	d. Enter the directory, compile, check, and install ATLAS

			cd ATLAS
			mkdir LinuxBuild
			cd LinuxBuild
			../configure -b 64 --prefix=/path/to/libraries --shared \
				--with-netlib-lapack-tarfile=/path/to/lapack-3.5.0.tgz
			make build      (this takes forever...)
			make check
			make ptcheck    (should be no errors for make check, ptcheck, time)
			make time
			make install



3) Install glpk

	a. Download glpk version 4.48 to your home directory. cvxopt is incompatible
	   with newer versions

	b. Untar glpk:
		
			tar xzf glpk-4.48.tar.gz

	c. Build and install glpk into your libraries directory:

			cd glpk-4.48
			./configure --prefix=/path/to/libraries
			make
			make check (should return no errors)
			make install



2) Download and install Python

	wget http://www.python.org/ftp/python/2.7.6/Python-2.7.6.tgz
	tar xzf Python-2.7.6.tgz
	cd Python-2.7.6
	./configure --prefix=/home/stsuei/libraries --enable-shared
	make
	make install



3) Install pip

	wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py
	python get-pip.py
	source ~/.tcshrc



4) Install numpy, scipy, matplotlib, ply

	pip install numpy
	pip install scipy
	pip install matplotlib
	pip install ply



5) Downgrade pyparsing and install pydot, networkx

	pip uninstall pyparsing
	pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709
	easy_install pydot (don't use pip)
	pip install networkx

   See this page for more detail: http://stackoverflow.com/questions/15951748/pydot-and-graphviz-error-couldnt-import-dot-parser-loading-of-dot-files-will



6) Install gr1c:

	a. Download gr1c-0.6.6-95ec9c6.tar.gz from 
	   http://vehicles.caltech.edu/private/snapshots/nessa/gr1c/
	   (Requires Caltech IP Address)

	b. Untar the package and copy all binaries to your bin folder:

		tar xzf gr1c-0.6.6-95ec9c6.tar.gz
		cd gr1c-0.6.6-95ec9c6
		cp gr1c rg grpatch /path/to/libraries/bin 



7) Install cvxopt:

	a. Download cvxopt 1.1.6: http://abel.ee.ucla.edu/src/cvxopt-1.1.6.tar.gz
	b. Untar and rename the archive:

			tar xzf cvxopt-1.1.6.tar.gz
			mv cvxopt-1.1.6.tar.gz cvxopt-orig.tar.gz
			cd cvxopt-1.1.6

	c. Edit setup file:

	   		-Line 5:  BLAS_LIB_DIR = '/path/to/libraries/lib'
			-Line 8:  BLAS_LIB = [ 'satlas', 'tatlas', 'atlas' ]
			-Line 9:  LAPACK_LIB = []
			-Line 10: BLAS_EXTRA_LINK_ARGS = [ '-nostdlib' ]
			-Line 36: BUILD_GLPK = 1
			-Line 39: GLPK_LIB_DIR = '/path/to/libraries/lib'
			-Line 42: GLPK_INC_DIR = '/path/to/libraries/include'


	  d. Tar the edited package and install with pip:

	  		cd ..
			tar czf cvxopt-1.1.6.tar.gz cvxopt-1.1.6/
			pip install cvxopt-1.1.6.tar.gz



8) Install tulip:

	git clone https://github.com/tulip-control/tulip-control.git 
	cd tulip-control
	python setup.py build
	python setup.py install
