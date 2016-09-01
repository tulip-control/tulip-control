#!/bin/sh -e
#
# Fetch and build `slugs`.
#
# After this script returns, the `slugs` executable is
# slugs-repo/src/slugs
# and the converter to slugsin format is
# slugs-repo/tools/StructuredSlugsParser/compiler.py

curl -L https://github.com/VerifiableRobotics/slugs/tarball/ad0cf12c14131fc6a20fe29edfe04d9aefd7c6d4 -o slugs.tar.gz
echo '04c8de023bddc6579fe5f1fe495dea641d3e4cf25bd48ff3a6b0a781d968136a6a635e07c9c1fcf0c784df60b72c24aab8029f0c613c71fed9c104f0e9f046ce  slugs.tar.gz' | shasum -a 512 -c -
mkdir slugs-repo
tar xzf slugs.tar.gz -C slugs-repo --strip=1
cd slugs-repo/src
make -j2
