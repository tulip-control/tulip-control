# Makefile for Epydoc-generated documentation
#
#
# NOTES: - By default, the "--graph=classtree" option is used when invoking
#          Epydoc, causing class inheritance to be shown in an image.  This
#          option requires the Graphviz (http://www.graphviz.org/) program dot
#          to be installed.

TULIP_REL     = ../tulip

EPYDOC        = epydoc
EPYBUILDDIR   = api_doc
EPYOPTS       = --graph=classtree

.PHONY: api

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  api        to make API documentation using Epydoc"
	@echo "  clean      delete build output"

clean:
	-rm -rf $(EPYBUILDDIR)/*

api:
	$(EPYDOC) $(EPYOPTS) --html -v --name=TuLiP --url=http://tulip-control.sourceforge.net -o $(EPYBUILDDIR) $(TULIP_REL)/{__init__.py,spec.py,polytope/__init__.py,polytope/esp.py,polytope/plot.py,polytope/polytope.py,polytope/quickhull.py,transys.py}
