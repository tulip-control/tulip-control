#!/usr/bin/env python

""" print.py --- Print an error and warning message

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010
"""

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[95m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
def printWarning(text):
    print bcolors.WARNING + text + bcolors.ENDC
def printError(text):
    print bcolors.FAIL + text + bcolors.ENDC
