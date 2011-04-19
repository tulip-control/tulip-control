#!/usr/bin/env python

from distutils.core import setup

setup(name = 'tulip',
      version = '0.1a',
      description = 'Temporal Logic Planning (TuLiP) Toobox',
      author = 'Nok Wongpiromsarn',
      author_email = 'murray@cds.caltech.edu',
      url = 'http://tulip-control.sourceforge.net',
      requires = ['scipy'],
      package_dir = {'tulip' : 'tulip'},
      packages = ['tulip'],
     )
