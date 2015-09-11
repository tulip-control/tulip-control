#!/bin/sh -e
#
# If you want to use the JTLV-based solver, then before `pip install .`
# run this script from the root of the sourcetree,
#
#     extern/get-jtlv.sh
#
# It will fetch the JTLV-based solver and place it under tulip/interfaces/.
#
# Note that the JTLV-based solver depends on Java and is accessible via
# tulip.interfaces.jtlv and indirectly through other parts of TuLiP.
# Also note that it is not necessary for using TuLiP.

curl -L -O https://github.com/tulip-control/tulip-control/raw/f2c2d7203e795cbabcf9c75516f437475f9faa9e/tulip/interfaces/jtlv_grgame.jar
SHA256CHECKSUM=718d417a866096609fa220475c3b259f98f97ccb7e3e18a46aca61dc1b657ae6

if hash sha256sum >/dev/null 2>&1; then
        FILECHECKSUM=`sha256sum jtlv_grgame.jar| sed 's/ .*//'`
elif hash shasum >/dev/null 2>&1; then
        FILECHECKSUM=`shasum -a 256 jtlv_grgame.jar| sed 's/ .*//'`
else
	echo "neither `sha256sum` nor `shasum` found in the PATH."
        rm jtlv_grgame.jar
	exit 1
fi

# fetch and verify checksum
if [ "$SHA256CHECKSUM" = "$FILECHECKSUM" ]
then
	mv jtlv_grgame.jar tulip/interfaces/
else
	echo "The checksums do not match."
	false  # Quit indicating error
fi
