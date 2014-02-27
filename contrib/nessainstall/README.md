## Install `tulip` under your `/home/username`
This directory contains instructions for installing Tulip in your home directory on `nessa.cds.caltech.edu`.
The instructions in `instructions.txt` are also
applicable to other Linux servers on which one does not have `sudo` powers.

A couple notes:

1. You need to enable `X11`-forwarding over ssh in order to use any modules that `import matplotlib`.

2. After following the directions in `install.sh`, your default Python installation will be the version of Python installed in your home directory, not the version installed on the machine for all users.
