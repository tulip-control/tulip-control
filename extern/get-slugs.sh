#!/bin/sh -e
#
# Fetch and build `slugs`.
#
# After this script returns, the `slugs` executable is
# slugs-repo/src/slugs
# and the converter to slugsin format is
# slugs-repo/tools/StructuredSlugsParser/compiler.py

curl -L https://github.com/VerifiableRobotics/slugs/tarball/7ccc857bfe50dec3181a525be48f98f68a539816 -o slugs.tar.gz
echo '9aebf5ebcf7af6f29d2174d07a2c1222a69b948e0ec144191098080e95fef7534cd4bca9c68f34c539998a8271e0e4737e3c83cb05457891d3e63e4bf956ad98  slugs.tar.gz' | shasum -a 512 -c -
mkdir slugs-repo
tar xzf slugs.tar.gz -C slugs-repo --strip=1
cd slugs-repo/src
make -j2
