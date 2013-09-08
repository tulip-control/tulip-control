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
      requires = ['numpy', 'scipy', 'cvxopt'],
      install_requires = ['networkx >= 1.6'],
      packages = ['tulip', 'tulip.transys', 'tulip.abstract', 
                  'tulip.polytope', 'tulip.spec'],
      package_dir = {'tulip' : 'tulip'},
      package_data={'tulip': ['jtlv_grgame.jar'], 
                    'tulip.transys' : ['d3.v3.min.js']},
)
