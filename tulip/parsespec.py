#!/usr/bin/env python
#
# Copyright (c) 2011 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# 
# $Id$

""" parseSpec.py --- parser module

doc-string goes here
"""

import re, copy
from numpy import array
from errorprint import printWarning, printError
from polytope_computations import Polytope

##############################################################
def parseMatrix(matstr):
    all_rows = matstr.split(';')
    mat = []
    num_col = -1
    for i, row in enumerate(all_rows):
        row = re.findall('[-+]?'+r'\d+\.?\d*', row)
        row = [float(num) for num in row]
        if (num_col < 0):
            num_col = len(row)
        elif(len(row) != num_col):
            printError('ERROR: The number of columns in row ' + str(i+1) + \
                           ' is not the same as the previous row')
            exit(1)            
        mat.append(row)
    return array(mat)
##############################################################
def parseLTL(file, vars):
    assumption = ''
    guarantee = ''
    guarantee_begin = False
    while 1:
        line = file.readline()
        if (line.strip() == 'env vars'):
            nexttype = 0
            break
        elif (line.strip() == 'sys disc vars'):
            nexttype = 1
            break
        elif (line.strip() == 'disc prop'):
            nexttype = 2
            break
        elif (line.strip() == 'sys cont vars'):
            nexttype = 3
            break
        elif (line.strip() == 'cont prop'):
            nexttype = 4
            break
        elif (line.strip() == 'sys dyn'):
            nexttype = 5
            break
        elif (line.strip() == 'spec'):
            nexttype = 6
            break
        elif not line: # End of File
            nexttype = -1
            break 
        elif (line.isspace()):
            pass  
        else:
            if (not guarantee_begin):
                tmpstr = line.partition(';')
                vars[0] += tmpstr[0].strip()
                vars[1] += tmpstr[2].strip()
                if (len(tmpstr[1]) > 0):
                    guarantee_begin = True
            else:
                vars[1] += line.strip()
    return nexttype

##############################################################
def parseContRange(list):
    numvars = len(list)
    A = []
    b = []
    for i in range(0,len(list)):
        minmax = re.findall('([-+]?\d+)\s*,\s*([-+]?\d+)', list[i])
        if (len(minmax) != 1):
            printError("ERROR: Unknown")
            exit(1)
        else:
            minmax = minmax[0]
            A = A + [[0. for j in range(0,i)] + [1.] + [0. for j in range(0, numvars-i-1)]]
            A = A + [[0. for j in range(0,i)] + [-1.] + [0. for j in range(0, numvars-i-1)]]
            b = b + [[float(minmax[1])]]
            b = b + [[float(minmax[0])]]
    A = array(A)
    b = array(b)
    return Polytope(A,b)

##############################################################
def parseCont(list):
    cont_prop = {}
    for sym, prop in list.iteritems():
        cont_part= prop.partition(',')
        contA = parseMatrix(cont_part[0])
        contB = parseMatrix(cont_part[2])
        cont_prop[sym] = Polytope(contA, contB)
    return cont_prop
    
##############################################################
def parseDict(file, vars):
    """Takes line and puts keys/vars into dictionary"""
    while 1:
        line = file.readline()
        if (len(line) == 0 or line.isspace()):
            pass
        elif (line.strip() == 'env vars'):
            nexttype = 0
            break
        elif (line.strip() == 'sys disc vars'):
            nexttype = 1
            break
        elif (line.strip() == 'disc prop'):
            nexttype = 2
            break
        elif (line.strip() == 'sys cont vars'):
            nexttype = 3
            break
        elif (line.strip() == 'cont prop'):
            nexttype = 4
            break
        elif (line.strip() == 'sys dyn'):
            nexttype = 5
            break
        elif (line.strip() == 'spec'):
            nexttype = 6
            break
        else:
            a = line.partition(':')
            name = a[0].strip()
            value = a[2].strip()
            vars[name] = value
    return nexttype


##############################################################
def parseSpec(spec_file):
    """Parses specifications from file

    - spec_file is the name of the text file containing the specs.
    """
    nexttype = -1
    file = open(spec_file, 'r')
    while 1:
        line = file.readline()
        if (line.strip() == 'env vars'):
            nexttype = 0
            break
        elif (line.strip() == 'sys disc vars'):
            nexttype = 1
            break
        elif (line.strip() == 'disc prop'):
            nexttype = 2
            break
        elif (line.strip() == 'sys cont vars'):
            nexttype = 3
            break
        elif (line.strip() == 'cont prop'):
            nexttype = 4
            break
        elif (line.strip() == 'sys dyn'):
            nexttype = 5
            break
        elif (line.strip() == 'spec'):
            nexttype = 6
            break
        elif (line.lstrip() != '#'):
            printError("No variables match inputs needed")
        elif not line:
            print('End of File')
            break   

    env_vars, sys_disc_vars, disc_prop, sys_cont_vars, cont_prop, sys_dyn = {}, {}, {}, {}, {}, {}
    spec = ['', '']
    while 1:
        if (nexttype == 0):
            nexttype = parseDict(file, env_vars)
        elif (nexttype == 1):
            nexttype = parseDict(file, sys_disc_vars)
        elif (nexttype == 2):
            nexttype = parseDict(file, disc_prop)
        elif (nexttype == 3):
            nexttype = parseDict(file, sys_cont_vars)
        elif (nexttype == 4):
            nexttype = parseDict(file, cont_prop)
        elif (nexttype == 5):
            nexttype = parseDict(file, sys_dyn)
        elif (nexttype == 6):
             nexttype = parseLTL(file, spec)
        else:
            break

    
    cont_vars = sys_cont_vars.keys()
    cont_vars_range = sys_cont_vars.values()
    cont_range_polyB = parseContRange(cont_vars_range)
    cont_prop = parseCont(cont_prop)
   

    return env_vars, sys_disc_vars, disc_prop, cont_vars, cont_range_polyB, cont_prop, sys_dyn, spec

########################################################

if __name__ == "__main__":
    (env_vars, sys_disc_vars, disc_prop, cont_vars, cont_range_polyB, cont_prop, sys_dyn, spec) = parseSpec(spec_file='specs/specFileEx.spc')

########################################################
