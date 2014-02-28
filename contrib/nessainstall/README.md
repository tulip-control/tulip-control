## Install `tulip` under your `/home/username`
This directory contains instructions for installing Tulip in your home directory on `nessa.cds.caltech.edu`.
The instructions in [`instructions.md`](instructions.md) are also applicable to other Linux servers on which one does not have `sudo` powers.

A couple notes:

1. For interactive ploting with `matplotlib` you need to enable `X11`-forwarding over `ssh` (option `-X`).

2. For saving figures just use a different backend for `matplotlib` as [discussed here](http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined?lq=1), e.g.:

		# in your user script
		import matplotlib as mpl
		mpl.use('Agg')
		# before mpl.pyplot or mpl.pylab are imported
	
	This avoids the `X11` session that is fragile and introduces delay. More permanently this can be saved in your `matplotlibrc`.


2. After following the directions in [`install.sh`](install.sh), your default Python installation will be the version of Python installed in your home directory, not the version installed on the machine for all users.
