#!/bin/sh
#
# Post summary webpage
#
# Usage: ./rsync-web.sh USERNAME

echo "Copying summary page..."
rsync -e ssh -przv website/*.html website/*.css $1@web.sourceforge.net:/home/project-web/tulip-control/htdocs/
