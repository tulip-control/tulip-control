#!/bin/sh -e
#
# Fetch and build `slugs`.
#
# After this script returns, the `slugs` executable is
# slugs-repo/src/slugs
# and the converter to slugsin format is
# slugs-repo/tools/StructuredSlugsParser/compiler.py

git clone --depth=10 https://github.com/LTLMoP/slugs.git slugs-repo
cd slugs-repo
git checkout ad0cf12c14131fc6a20fe29edfe04d9aefd7c6d4
cd src
make
