#!/usr/bin/env python

from distutils.core import setup

setup(name = 'tulip',
      version = '0.2a',
      description = 'Temporal Logic Planning (TuLiP) Toobox',
      author = 'Caltech Control and Dynamical Systems',
      author_email = 'murray@cds.caltech.edu',
      url = 'http://tulip-control.sourceforge.net',
      requires = ['scipy', 'numpy', 'cvxopt'],
      packages = ['tulip'],
      package_dir = {'tulip' : 'tulip'},
      package_data={'tulip': ['matlab/*.m', 'jtlv_grgame.jar']},
      scripts = ['tools/aut2dot',]
      )
