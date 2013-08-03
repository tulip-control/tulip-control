# Makefile for Sphinx and Epydoc-generated documentation
#
#
# NOTES: - By default, the "--graph=classtree" option is used when invoking
#          Epydoc, causing class inheritance to be shown in an image.  This
#          option requires the Graphviz (http://www.graphviz.org/) program dot
#          to be installed.

TULIP_REL     = ../tulip

# Epydoc options
EPYDOC        = epydoc
EPYBUILDDIR   = api_doc
EPYOPTS       = --graph=classtree --no-sourcecode

# Sphinx options
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

.PHONY: api

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  api        to make API documentation using Epydoc"
	@echo "  clean      delete build output"

clean:
	-rm -rf $(BUILDDIR)/* $(EPYBUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) . $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

api:
	$(EPYDOC) $(EPYOPTS) --html -v --name=TuLiP --url=http://tulip-control.sourceforge.net -o $(EPYBUILDDIR) $(TULIP_REL)/{__init__.py,spec.py,polytope/__init__.py,polytope/esp.py,polytope/plot.py,polytope/polytope.py,polytope/quickhull.py,transys.py,jtlvint.py,gr1cint.py,synth.py}
