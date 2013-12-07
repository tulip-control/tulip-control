#!/bin/sh
#
# Usage: ./rsync-it.sh USERNAME

echo "Copying User's and Developer's Guides..."
rsync -e ssh -przv _build/html/* $1@web.sourceforge.net:/home/project-web/tulip-control/htdocs/doc/

echo "Copying API documentation..."
rsync -e ssh -przv _build/api_doc/* $1@web.sourceforge.net:/home/project-web/tulip-control/htdocs/api-doc/
