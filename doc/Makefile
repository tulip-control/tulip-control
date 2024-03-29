# Makefile for Sphinx and Epydoc-generated documentation
#
#
# NOTES: - By default, the "--graph=classtree" option is used when invoking
#          Epydoc, causing class inheritance to be shown in an image.  This
#          option requires the Graphviz (http://www.graphviz.org/) program dot
#          to be installed.

TULIP_REL     = ../tulip

# Epydoc options
EPYDOC        = python -c 'from epydoc.cli import cli; cli()'
EPYBUILDDIR   = api_doc
EPYOPTS       = -v --graph=classtree --no-sourcecode --no-private

# Sphinx options
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build/

.PHONY: api

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  api        to make API documentation using Epydoc"
	@echo "  clean      delete build output"

clean:
	-rm -rf $(BUILDDIR)*

html:
	@echo "Generating bibliography.rst..."
	./genbib.py bib.txt
	mkdir -p _static  # because it is mentioned in `conf.py`
	$(SPHINXBUILD) -b html $(SPHINXOPTS) . $(BUILDDIR)html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)html."

api:
	mkdir -p $(BUILDDIR)$(EPYBUILDDIR)
	$(EPYDOC) $(EPYOPTS) --html --name=TuLiP --url=http://tulip-control.org -o $(BUILDDIR)$(EPYBUILDDIR) $(TULIP_REL) --exclude=parsetab\|version\|graphics
