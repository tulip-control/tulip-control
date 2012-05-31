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

""" 
-------------------------------------
Rhtlputil Module --- Helper Functions
-------------------------------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 25, 2010
:Version: 0.1.0
"""

import re, copy, os
from itertools import product
from errorprint import printError, printWarning

def evalExpr(expr='', vardict={}, verbose=0):
    vardict = copy.deepcopy(vardict)
    imp_ind = expr.rfind('->')
    if (imp_ind < 0):
        vardict['True'] = True
        vardict['False'] = False
        expr = expr.replace('!', 'not ')
        expr = expr.replace('&', ' and ')
        expr = expr.replace('|', ' or ')
        expr = re.sub(r'\b\s*=', '==', expr)
        ret = eval(expr, {"__builtins__":None}, vardict)
        return ret
    else:
        right_expr = expr[imp_ind+2:]
        if (verbose > 1):
            print 'right_expr = ' + right_expr
        num_paren = 0
        right_expr_end_ind = 0
        while (num_paren < 1 and right_expr_end_ind < len(right_expr)):
            if (right_expr[right_expr_end_ind] == '('):
                num_paren -= 1
            elif (right_expr[right_expr_end_ind] == ')'):
                num_paren += 1
            right_expr_end_ind += 1
        if (num_paren == 0):
            left_expr = expr[:imp_ind]
            if (verbose > 1):
                print 'left_expr = ' + left_expr
            right_expr_eval = evalExpr(right_expr, vardict, verbose)
            left_expr_eval = evalExpr(left_expr, vardict, verbose)
            if (not left_expr_eval or right_expr_eval):
                return True
            else:
                return False
        elif (num_paren == 1):
            right_expr_end_ind -= 1
            guarantee_formula = right_expr[:right_expr_end_ind]
            if (verbose > 1):
                print 'guarantee_formula = ' + guarantee_formula
            guarantee_eval = evalExpr(guarantee_formula, vardict, verbose)
            ass_start_ind = imp_ind
            num_paren = 0
            while (num_paren > -1 and ass_start_ind > 0):
                ass_start_ind -= 1
                if (expr[ass_start_ind] == '('):
                    num_paren -= 1
                elif (expr[ass_start_ind] == ')'):
                    num_paren += 1
            if (num_paren != -1):
                printError('ERROR rhtlputil.evalExpr: ' + \
                           'Invalid formula. Unbalanced parenthesis in ' + \
                               expr[:right_expr_end_ind+1])
                raise Exception("Invalid formula")
            assumption_formula = expr[ass_start_ind+1:imp_ind]
            if (verbose > 1):
                print 'assumption_formula = ' + assumption_formula
            assumption_eval = evalExpr(assumption_formula, vardict, verbose)
            imp_res = 'False'
            if (not assumption_eval or guarantee_eval):
                imp_res = 'True'
            if (verbose > 1):
                print 'implication result = ' + imp_res
            reduced_expr = expr[:ass_start_ind] + imp_res + right_expr[right_expr_end_ind+1:]
            if (verbose > 1):
                print 'reduced_expr = ' + reduced_expr
            ret = evalExpr(reduced_expr, vardict, verbose)
            return ret
        else:
            printError('ERROR rhtlputil.evalExpr: ' + \
                           "Invalid formula. Too many '(' in " + right_expr)
            raise Exception("Invalid formula")


###################################################################
def yicesSolveSat(expr='', allvars={}, ysfile='', verbose=0):
    import subprocess, sys
    cmd = subprocess.Popen(["which", "yices"], stdout=subprocess.PIPE, close_fds=True)
    yices_exist = False
    for line in cmd.stdout:
        if (verbose > 0):
            print line
        if ('yices' in line):
            yices_exist = True

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if (not yices_exist):
        return None

    toYices(expr=expr, allvars=allvars, ysfile=ysfile, verbose=verbose)
    cmd = subprocess.Popen(["yices", "-e", ysfile], \
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    sat = False
    ce = ''
    for line in cmd.stdout:
        if (verbose > 1):
            print line,
        if ('Error' in line):
            printError("yices error: ")
            print line
            return None
        if ('unsat' in line):
            sat = False
            ce = ''
            return sat, ce
        elif ('sat' in line):
            sat = True
            continue

        if (sat and not line.isspace()): # Examples
            ce += line
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return sat, ce

###################################################################
def parseYices(ysfile, verbose=0):
    sat = False
    ce = ''
    f = open(ysfile, 'r')
    for line in f:
        if ('unsat' in line):
            sat = False
            ce = ''
            return sat, ce
        elif ('sat' in line):
            sat = True
            continue

        if (sat): # Examples
            ce += line + '\n'
    f.close()

###################################################################

def toYices(expr='', allvars={}, ysfile='', verbose=0):
    # Temporary variable for defining subtype
    stvar = 'n'
    while (stvar in allvars.keys()):
        stvar += 'n'

    if not os.path.exists(os.path.abspath(os.path.dirname(ysfile))):
        if verbose > 0:
            printWarning("Folder for ysfile " + ysfile + " does not exist. Creating...", obj=None)
        os.mkdir(os.path.abspath(os.path.dirname(ysfile)))
    f = open(ysfile, 'w')

    # Declare the variables 
    for var, val in allvars.iteritems():
        if ('boolean' in val):
            f.write('(define ' + var + '::bool)\n')
        else:
            if (isinstance(val, str)):
                tmp = re.findall('[-+]?\d+', val)
            elif (isinstance(val, list)):
                tmp = tmp = [str(i) for i in val]
            else:
                printError('Unknown type of variable ' + var)
                raise TypeError("Invalid variable type.")

            tmpstr = ''
            for tmpval in tmp:
                tmpstr += ' (= ' + stvar + ' ' + tmpval + ')'

            f.write('(define ' + var + '::(subtype (' + stvar + '::int) (or' + \
                        tmpstr + ')))\n')

    # Formula
    f.write('(assert ' + expr2ysstr(expr=expr, verbose=verbose) + ')\n')
    f.write('(check)\n')
    f.close()


###################################################################

def expr2ysstr(expr='', verbose=0):
    expr = re.sub(r'\b'+'True'+r'\b', 'true', expr)
    expr = re.sub(r'\b'+'False'+r'\b', 'false', expr)

    expr = expr.strip()
    if (verbose > 3):
        print('expr = ' +  expr)

    if (len(expr) == 0):
        retstr = ''
        return retstr

    numoparen = expr.count('(')
    numcparen = expr.count(')')
    if (numoparen != numcparen):
        printError("Unbalanced parenthesis. " + str(numoparen) + " of '(' and " + \
                       str(numcparen) + " of ')'")
        raise Exception("Invalid formula")

    # Deal with the parentheses
    paren_tmp = {}
    if (numoparen > 0):
        oparen_ind = []
        cparen_ind = []
        num_paren = 0
        for ind in xrange(0, len(expr)):
            if (expr[ind] == '('):
                    num_paren += 1
                    if (num_paren == 1):
                        oparen_ind.append(ind)
            elif (expr[ind] == ')'):
                if (num_paren <= 0):
                    printError("Unbalanced parenthesis. Extra ')' at position " + str(ind))
                    raise Exception("Invalid formula")
                num_paren -= 1 
                if (num_paren == 0):
                    cparen_ind.append(ind)

        if (len(oparen_ind) != len(cparen_ind)):
            printError("Unbalanced parenthesis.")
            raise Exception("Invalid formula")
        if (verbose > 3):
            print('oparen_ind = ' + str(oparen_ind))
            print('cparen_ind = ' + str(cparen_ind))

        if (len(oparen_ind) == 1 and oparen_ind[0] == 0 and \
                len(cparen_ind) == 1 and cparen_ind[0] == len(expr)-1):
            return expr2ysstr(expr=expr[1:-1], verbose = verbose)

        paren_tmp_sym = 'paren_tmp_sym'
        while (paren_tmp_sym in expr):
            paren_tmp_sym += 'm'

        for ind in xrange(len(cparen_ind)-1, -1, -1):
            id = str(len(paren_tmp))
            paren_tmp[paren_tmp_sym + id] = \
                expr[oparen_ind[ind]:cparen_ind[ind]+1]
            expr = expr[:oparen_ind[ind]] + ' ' + paren_tmp_sym + id + \
                ' ' + expr[cparen_ind[ind]+1:]

        if (verbose > 3):
            print("After processing parentheses, expr = " + expr)
            print(paren_tmp)
            
    # First split the formula by the temporal operator
    sexpr = expr.rsplit('->', 1)
    if (len(sexpr) > 1):
        (leftexpr, rightexpr) = __toYicesSepExpr(sexpr[0].strip(), sexpr[1].strip(), \
                                                     paren_tmp, isbinary=True, \
                                                     checkleft=True, checkright=True, \
                                                     verbose=verbose)
        retstr = '(=> ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
            expr2ysstr(rightexpr, verbose=verbose) + ')'
        return retstr

    sexpr = expr.rsplit('|', 1)
    if (len(sexpr) > 1):
        (leftexpr, rightexpr) = __toYicesSepExpr(sexpr[0].strip(), sexpr[1].strip(), \
                                                     paren_tmp, isbinary=True,  \
                                                     checkleft=True, checkright=True, \
                                                     verbose=verbose)
        retstr = '(or ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
            expr2ysstr(rightexpr, verbose=verbose) + ')'
        return retstr

    sexpr = expr.rsplit('&', 1)
    if (len(sexpr) > 1):
        (leftexpr, rightexpr) = __toYicesSepExpr(sexpr[0].strip(), sexpr[1].strip(), \
                                                     paren_tmp, isbinary=True, \
                                                     checkleft=True, checkright=True, \
                                                     verbose=verbose)
        retstr = '(and ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
            expr2ysstr(rightexpr, verbose=verbose) + ')'
        return retstr

    sexpr = expr.rsplit('!', 1)
    if (len(sexpr) > 1):
        (leftexpr, rightexpr) = __toYicesSepExpr(sexpr[0].strip(), sexpr[1].strip(), \
                                                     paren_tmp, isbinary=False, \
                                                     checkleft=True, checkright=True, \
                                                     verbose=verbose)
        retstr = '(not ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
            expr2ysstr(rightexpr, verbose=verbose) + ')'
        return retstr

    # Next split the formula by =, >=, >, <=, <
    indeq = expr.find('=')
    indgr = expr.find('>')
    indle = expr.find('<')
    if (indeq > 0 or indgr > 0 or indle > 0):
        numeq = expr.count('=')
        if (numeq > 1):
            printError("Too many '=' in " + expr)
            raise Exception("Invalid formula")
        numgr = expr.count('>')
        if (numgr > 1):
            printError("Too many '>' in " + expr)
            raise Exception("Invalid formula")
        numle = expr.count('<')
        if (numle > 1):
            printError("Too many '<' in " + expr)
            raise Exception("Invalid formula")
        if (numle == 1 and numgr == 1):
            printError("The subformula " + expr + " contains both > and <")
            raise Exception("Invalid formula")
        if (numeq == 1 and numgr == 1 and indeq != indgr + 1):
            printError("The subformula " + expr + " contains both > and =")
            raise Exception("Invalid formula")
        if (numeq == 1 and numle == 1 and indeq != indle + 1):
            printError("The subformula " + expr + " contains both < and =")
            raise Exception("Invalid formula")

        if (numeq == 1 and numgr == 1): # >=
            (leftexpr, rightexpr) = __toYicesSepExpr(expr[:indgr].strip(), \
                                                         expr[indeq+1:].strip(), \
                                                         paren_tmp, isbinary=True, \
                                                         checkleft=True, checkright=True, \
                                                         verbose=verbose)
            retstr = '(>= ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
                expr2ysstr(rightexpr, verbose=verbose) + ')'
            return retstr
        if (numeq == 1 and numle == 1): # <=
            (leftexpr, rightexpr) = __toYicesSepExpr(expr[:indle].strip(), \
                                                         expr[indeq+1:].strip(), \
                                                         paren_tmp, isbinary=True, \
                                                         checkleft=True, checkright=True, \
                                                         verbose=verbose)
            retstr = '(<= ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
                expr2ysstr(rightexpr, verbose=verbose) + ')'
            return retstr
        if (numeq == 1): # =
            (leftexpr, rightexpr) = __toYicesSepExpr(expr[:indeq].strip(), \
                                                         expr[indeq+1:].strip(), \
                                                         paren_tmp, isbinary=True, \
                                                         checkleft=True, checkright=True, \
                                                         verbose=verbose)
            retstr = '(= ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
                expr2ysstr(rightexpr, verbose=verbose) + ')'
            return retstr
        if (numgr == 1): # >
            (leftexpr, rightexpr) = __toYicesSepExpr(expr[:indgr].strip(), \
                                                         expr[indgr+1:].strip(), \
                                                         paren_tmp, isbinary=True, \
                                                         checkleft=True, checkright=True, \
                                                         verbose=verbose)
            retstr = '(> ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
                expr2ysstr(rightexpr, verbose=verbose) + ')'
            return retstr
        if (numle == 1): # <
            (leftexpr, rightexpr) = __toYicesSepExpr(expr[:indle].strip(), \
                                                         expr[indle+1:].strip(), \
                                                         paren_tmp, isbinary=True, \
                                                         checkleft=True, checkright=True, \
                                                         verbose=verbose)
            retstr = '(< ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
                expr2ysstr(rightexpr, verbose=verbose) + ')'
            return retstr

    # Next split the formula by +, -, * and /
    indpl = expr.rfind('+')
    indmi = expr.rfind('-')
    indmu = expr.rfind('*')
    inddi = expr.rfind('/')

    if (indpl > 0 or indmi > 0):
        if (indpl > indmi):
            op = '+'
        else:
            op = '-'
        sexpr = expr.rsplit(op, 1)
        (leftexpr, rightexpr) = __toYicesSepExpr(sexpr[0].strip(), sexpr[1].strip(), \
                                                     paren_tmp, isbinary=True, \
                                                     checkleft=False, checkright=True, \
                                                     verbose=verbose)
        if (len(leftexpr) == 0):
            leftexpr = '0'
        if (not (leftexpr[-1].isalnum() or leftexpr[-1] == ')')):
            printError("Invalid subformula " + leftexpr)
            raise Exception("Invalid formula")
        retstr = '(' + op + ' ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
            expr2ysstr(rightexpr, verbose=verbose) + ')'
        return retstr
    if (indmu > 0 or inddi > 0):
        if (indmu > inddi):
            op = '*'
        else:
            op = '/'
        sexpr = expr.rsplit(op, 1)
        (leftexpr, rightexpr) = __toYicesSepExpr(sexpr[0].strip(), sexpr[1].strip(), \
                                                     paren_tmp, isbinary=True, \
                                                     checkleft=True, checkright=True, \
                                                     verbose=verbose)
        retstr = '(' + op + ' ' + expr2ysstr(leftexpr, verbose=verbose) + ' ' + \
            expr2ysstr(rightexpr, verbose=verbose) + ')'
        return retstr

    # If none of the operators are found, then just return the expression

    return expr.strip()



###################################################################

def __toYicesSepExpr(leftexpr, rightexpr, paren_tmp, isbinary=True, checkleft=True, \
                         checkright=True, verbose=0):
    for k, v in paren_tmp.iteritems():
        leftexpr = leftexpr.replace(k, v)
        rightexpr = rightexpr.replace(k, v)

    leftexpr = leftexpr.strip()
    if (checkleft):
        if (isbinary):
            if (len(leftexpr) == 0 or not (leftexpr[-1].isalnum() or leftexpr[-1] == ')')):
                printError("Invalid subformula " + leftexpr)
                raise Exception("Invalid formula")
        elif (len(leftexpr) > 0): # The case where the operator is !
            if((leftexpr[-1] != '(' and leftexpr[-1] != '&' and \
                   leftexpr[-1] != '|') or (len(leftexpr) > 1 and leftexpr[-2:] != '->')):
                printError("Invalid subformula " + leftexpr)
                raise Exception("Invalid formula")

    rightexpr = rightexpr.strip()
    if (checkright):
        if (len(rightexpr) == 0 or not (rightexpr[0].isalnum() or rightexpr[0] == '(' \
                                            or rightexpr[0] == '!')):
            printError("Invalid subformula " + rightexpr)
            raise Exception("Invalid formula")

    return leftexpr, rightexpr


###################################################################

def findCycle(graph, W0ind=[], verbose=0):
# Find a cycle that does not go through any node in W0ind
    for root in xrange(0, len(graph)):
        if (root in W0ind):
            continue

        to_visit = [root]
        to_visit_path = [[root]]
        visited = []

        while (not (len(to_visit) == 0)):
            v = to_visit.pop()
            visited.append(v)
            path_to_v = to_visit_path.pop()
            neighbors = graph[v]
            tmpneighbors = []
            for ne in neighbors:
                if (not ne in W0ind):
                    tmpneighbors.append(ne)
            neighbors = tmpneighbors

            if (verbose > 2):
                print
                print 'root = ', root
                print 'to_visit = ', to_visit
                print 'to_visit_path = ', to_visit_path
                print 'v = ', v
                print 'path_to_v = ', path_to_v
                print 'neighbors = ', neighbors
                print 'visited = ', visited
            for i in xrange(0,len(neighbors)):
                if (neighbors[i] == root):
                    path = path_to_v[:]
                    path.append(neighbors[i])
                    return path
                elif (neighbors[i] in path_to_v):
                    ind = path_to_v.index(neighbors[i])
                    path = path_to_v[ind:]
                    path.append(neighbors[i])
                    return path
                elif (not neighbors[i] in visited and \
                          not neighbors[i] in to_visit):
                    to_visit.append(neighbors[i])
                    tmp = path_to_v[:]
                    tmp.append(neighbors[i])
                    to_visit_path.append(tmp)
                    if (verbose > 2):
                        print 'path_to_v = ', path_to_v
                        print 'to_visit = ', to_visit
                        print 'to_visit_path = ', to_visit_path

    return []


###################################################################

def findPath(graph, root, goal, verbose=0):
    if (root == goal):
        return [root]

    to_visit = [root]
    to_visit_path = [[root]]
    visited = []

    while (not (len(to_visit) == 0)):
        v = to_visit.pop()
        visited.append(v)
        path_to_v = to_visit_path.pop()
        neighbors = graph[v]
        if (verbose > 1):
            print
            print 'to_visit = ', to_visit
            print 'to_visit_path = ', to_visit_path
            print 'v = ', v
            print 'path_to_v = ', path_to_v
            print 'neighbors = ', neighbors
            print 'visited = ', visited
        for i in xrange(0,len(neighbors)):
            if (neighbors[i] == goal):
                path = path_to_v[:]
                path.append(neighbors[i])
                return path
            elif (not neighbors[i] in visited and \
                      not neighbors[i] in to_visit):
                to_visit.append(neighbors[i])
                tmp = path_to_v[:]
                tmp.append(neighbors[i])
                to_visit_path.append(tmp)
                if (verbose > 2):
                    print 'path_to_v = ', path_to_v
                    print 'to_visit = ', to_visit
                    print 'to_visit_path = ', to_visit_path

    return []
