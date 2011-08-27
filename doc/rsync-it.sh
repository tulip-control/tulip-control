#!/bin/sh
#
# Usage: ./rsync-it.sh USERNAME

rsync -e ssh -przv _build/html/* $1@web.sourceforge.net:/home/project-web/tulip-control/htdocs/doc/
