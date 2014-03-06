## Install `tulip` under your `/home/username`
This directory contains instructions for installing Tulip in your home directory on `nessa.cds.caltech.edu`.
The instructions in [`instructions.md`](instructions.md) are also applicable to other Linux servers on which one does not have `sudo` powers.

A couple notes:

1. For saving figures there are three solutions (ordered from temporary to permanent):
	1. For interactive ploting with `matplotlib` you need to enable `X11`-forwarding over `ssh` (option `-X`). The the `X11` session is fragile and introduces minor delay.
	2. use a different backend for `matplotlib` as [discussed here](http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined?lq=1), e.g.:

			# in your user script
			import matplotlib as mpl
			mpl.use('Agg')
			# before mpl.pyplot or mpl.pylab are imported
	
	3. configure your `matplotlib` installation to use a different backend, as per its [documentation](http://matplotlib.org/users/customizing.html). This is the proper solution, because this is a platform-dependent issue, not something that should affect your code, nor bother you with extra `ssh` flags on every login.

2. After following the directions in [`install.sh`](install.sh), your default Python installation will be the version of Python installed in your home directory, not the version installed on the machine for all users.
