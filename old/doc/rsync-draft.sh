#!/bin/sh
#
# Usage: ./rsync-draft.sh USERNAME

rsync -e ssh -przv _build/html/* $1@web.sourceforge.net:/home/project-web/tulip-control/htdocs/draft-doc/
