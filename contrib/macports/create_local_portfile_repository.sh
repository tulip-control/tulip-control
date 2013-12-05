#!/bin/bash

# create a local MacPorts portfiles repository, guide at:
#	http://guide.macports.org/#development.local-repositories

# root ?
if [[ $UID != 0 ]]; then
    echo "Please run this script with sudo:"
    echo "sudo $0 $*"
    exit 1
fi

# macports available ?
command -v portindex >/dev/null 2>&1 || {
	echo >&2 "Cannot find MacPorts portindex command.\
	Check MacPorts installation.";
	exit 1;
}

# create local portfiles repository
mkdir -p -v ~/ports/math/py-tulip
cp Portfile ~/ports/math/py-tulip/Portfile

# link MacPorts to above
sudo echo file:///Users/`logname`/ports [nosync] >> \
	/opt/local/etc/macports/sources.conf
cd ~/ports
portindex

# demonstrate
echo on
port search py-tulip
