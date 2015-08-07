This file provides instructions for running TuLiP using a Vagrant-generated VM
as provided by VirtualBox on a UNIX-like platform in which only non-root
(non-superuser) capabilities are available. There are basically three steps:

1. build and install Ruby locally, i.e., within your home directory;
2. install Vagrant locally;
3. `vagrant up` using Vagrantfile from the tulip-control repository.

In particular, a C compiler with which Ruby can be built and
[VirtualBox](https://www.virtualbox.org) must already be available. You can test
for these using

    vboxmanage --version
    cc -v

(Usually `cc` is an alias for `gcc` or some other C compiler. Failing that, try
`gcc --version`.) Part of the installation of Vagrant requires building and
installing several Ruby packages (gems) on which Vagrant depends. The
requirements for these are not completely listed here, but note that at least
one of them requires `g++`.


## Ruby

First, get, build, and install [Ruby](https://www.ruby-lang.org) within your
home directory. The latest release is available from <https://www.ruby-lang.org/en/downloads/>.
At the time of writing, the release version was 2.2.2.

    cd ~
    mkdir -p tmp
    cd tmp
    curl -L -O https://cache.ruby-lang.org/pub/ruby/2.2/ruby-2.2.2.tar.gz

The SHA256 digest of ruby-2.2.2.tar.gz should be
5ffc0f317e429e6b29d4a98ac521c3ce65481bfd22a8cf845fa02a7b113d9b44.

    tar -xzf ruby-2.2.2.tar.gz
    cd ruby-2.2.2
    ./configure --prefix=$(echo ~)/opt
    make && make install

In the above, we install Ruby to a subdirectory named "opt" in your home
directory. This is referred to generically with `$(echo ~)/opt`. Your shell
should automatically expand `$(echo ~)` to the absolute path of your home
directory. Failing that, you can enter it explicitly, e.g., changing the above
line to `./configure --prefix=/home/frodo/opt`. Finally,

    export PATH=$(echo ~)/opt/bin:$PATH

At this stage, `ruby` should be on your shell path. To check for it, try

    ruby --version


## Vagrant

Second, get and install [Vagrant](https://www.vagrantup.com) using the sources
for the current release. At the time of writing, the version was 1.7.2.

    cd ~
    cd tmp
    curl -L -O https://github.com/mitchellh/vagrant/archive/v1.7.2.tar.gz

The SHA256 digest of v1.7.2.tar.gz should be
e7cc1fadc53619afc966a88bc015c89b6daf2218961fa7f79a42feeaeb70adb6.

    tar -xzf v1.7.2.tar.gz
    cd vagrant-1.7.2
    gem install --version '< 1.8.0' bundler
    bundle install
    rake install

At this stage, `vagrant` should be on your shell path. To check for it, try

    vagrant --version


## Creating the VM

Obtain a copy of the TuLiP sources, e.g., a release

    curl -L -O https://github.com/tulip-control/tulip-control/archive/v1.2.0.tar.gz

or the repository

    git clone https://github.com/tulip-control/tulip-control.git

Then, supposing that the root directory of the TuLiP sourcetree is named
"tulip-control",

    cd tulip-control/contrib
    vagrant up

In some cases, `vagrant up` may fail because a program named `bsdtar` is
missing. Without root (superuser) capabilities, the easiest way to continue is
to create a symbolic link in `~/opt/bin` to `/bin/tar` but call it `bsdtar`.
E.g.,

    pushd ~/opt/bin
    ln -s /bin/tar bsdtar
    popd

then try `vagrant up` again. This will create a VM with Ubuntu 14.04 and most
dependencies of TuLiP; several optional dependencies like `lily` are not
included. Once it has completed, you can log-in using

    vagrant ssh

On the VM, the directory `/vagrant` is synchronized with the host directory that
is one level up from "contrib", i.e., the root of the TuLiP sourcetree. The VM
is now ready to have `polytope` and `tulip` installed in the usual manner. E.g.,

    cd /vagrant
    sudo pip install -e .

Now, the common use-case is to open a second terminal on the host (not the VM)
and then work from the root of the TuLiP sourcetree. Since file synchronization
is automatic, you can easily switch to the first terminal (on the VM) and run a
program that you are developing using an editor on the host, e.g., Emacs.

To get graphical output from the VM,

    vagrant ssh -- -X


## Additional options and usage

By default, the VM may not be configured to use the hardware of the host system
to the full extent. Here we describe how to adjust the VM to use all of the
available cores and memory. First halt the VM created (if it is running),

    vagrant halt

From the command-line, `vboxmanage list vms` will list the current VMs. Find the
one that was just created, which will have a name like "contrib_default_1438".
Then, to provide for 8 cores and 40000 MB of RAM,

    vboxmanage modifyvm contrib_default_14389 --cpus 8
    vboxmanage modifyvm contrib_default_14389 --memory 40000

You can then start the VM using `vagrant up` again from the "contrib" directory.
