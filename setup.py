#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

setup(name = 'tulip',
      version = '1.0a',
      description = 'Temporal Logic Planning (TuLiP) Toolbox',
      author = 'Caltech Control and Dynamical Systems',
      author_email = 'murray@cds.caltech.edu',
      url = 'http://tulip-control.sourceforge.net',
      license = 'BSD',
      requires = ['numpy', 'scipy', 'cvxopt', 'networkx'],
      packages = ['tulip'],
      package_dir = {'tulip' : 'tulip'},
      package_data={'tulip': ['jtlv_grgame.jar', 'polytope/*.py', 'abstract/*.py']},
      )
