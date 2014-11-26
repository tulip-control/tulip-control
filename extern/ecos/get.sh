#!/bin/sh -e
#
# Get and patch ECOS version 1.0.4
#
# Original Copies of patched files have names with suffix of ".orig"
# Consult README in this same directory for help and details.

wget https://github.com/embotech/ecos/archive/v1.0.4.tar.gz
MD5CHECKSUM=9846bfa7817d3669f3c9269bf5567811
FILECHECKSUM=`md5sum v1.0.4.tar.gz| sed 's/ .*//'`
if [ "$MD5CHECKSUM" = "$FILECHECKSUM" ]
then
	tar xzf v1.0.4.tar.gz
	cd ecos-1.0.4
	sed -i.orig '10s/-Wextra/-Wextra -fPIC/' ecos.mk
else
	echo "The checksums do not match.  Please consult the README."
	false  # Quit indicating error
fi
