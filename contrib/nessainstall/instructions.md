The script `install.sh` installs Tulip on a `*nix` system.

Tip for using [`sshfs`](https://en.wikipedia.org/wiki/SSHFS) on `Mac OS X`,
to work comfortably on a remote machine:

- install [`FUSE for OS X`](http://osxfuse.github.io/)
- install `sshfs` with `MacPorts`, or if that is broken, by downloading the `pkg` from [`github`](https://github.com/osxfuse/osxfuse/wiki/SSHFS).
- if in your remote `~/.tcsh` you start `bash`, then remove that
- Then `sshfs -p 22 user@nessa.cds.caltech.edu:/home/user/path/to/tulip-control/ /Users/user/path/to/local/nessa/ -oauto_cache,reconnect,defer_permissions,negative_vncache,volname=nessa`

You can now:

- use your favorite editor/IDE locally
- view the files (esp. images) locally in `Finder`
- use [`virtualenvwrapper`](http://virtualenvwrapper.readthedocs.org/en/latest/) to install the remote `tulip` locally in `develop` mode:

```
mkvirtualenv nessa
cd ~/nessa`
python setup.py develop
```

Then your editor (e.g. `spyder`) will see the docstrings and functions as you work on them
- run the files remotely by opening a separate `ssh` session:

```
ssh user@nessa.cds.caltech.edu
cd ~/tulip-control
```
