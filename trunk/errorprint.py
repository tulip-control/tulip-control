#!/usr/bin/env python

""" print.py --- Print an error and warning message

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010
"""
import inspect

class bcolors:
    OKGREEN = '\033[1;92m'
    WARNING = '\033[1;95m'
    FAIL = '\033[1;91m'
    INFO = '\033[1;94m'
    ENDC = '\033[0m'

def printWarning(text, obj=None):
    tmp = "WARNING"
    modulename = str(inspect.getmodulename(inspect.getouterframes(inspect.currentframe())[1][1]))
    if (modulename != 'None'):
        tmp += " " + modulename + '.'
    if (obj is not None):
        tmp += obj.__class__.__name__ + '.'
    if (modulename != 'None'):
        tmp += str(inspect.getouterframes(inspect.currentframe())[1][3])
    tmp += ": " + text 
    print bcolors.WARNING + tmp + bcolors.ENDC

def printError(text, obj=None):
    tmp = "ERROR"
    modulename = str(inspect.getmodulename(inspect.getouterframes(inspect.currentframe())[1][1]))
    if (modulename != 'None'):
        tmp += " " + modulename + '.'
    if (obj is not None):
        tmp += obj.__class__.__name__ + '.'
    if (modulename != 'None'):
        tmp += str(inspect.getouterframes(inspect.currentframe())[1][3])
    tmp += ": " + text 
    print bcolors.FAIL + tmp + bcolors.ENDC

def printInfo(text):
    print bcolors.INFO + text + bcolors.ENDC
