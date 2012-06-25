#!/bin/sh
#
# Usage: ./rsync-apidraft.sh USERNAME

rsync -e ssh -przv api_doc/* $1@web.sourceforge.net:/home/project-web/tulip-control/htdocs/draft-api/
