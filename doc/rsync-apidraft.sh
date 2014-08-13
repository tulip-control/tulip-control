#!/bin/sh
#
# Usage: ./rsync-apidraft.sh USERNAME

rsync -e ssh -przv _build/api_doc/* $1@web.sourceforge.net:/home/project-web/tulip-control/htdocs/api-doc-draft/
