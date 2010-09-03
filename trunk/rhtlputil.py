#!/usr/bin/env python

""" 
-------------------------------------
Rhtlputil Module --- Helper Functions
-------------------------------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 25, 2010
:Version: 0.1.0
"""

import re, copy

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

def product(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

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

            if (verbose > 1):
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



###################################################################

# Test
if __name__ == "__main__":
    print('Testing evalExpr')
    vardict = {'x':1, 'y':0, 'z':1, 'w':0}
    expr = 'x=0 -> y < 1 & z >= 2 -> w'
    ret = evalExpr(expr=expr, vardict=vardict, verbose=3)
    print ret, " Should be False"
    print
    expr = '(x=0 -> y < 1) & (z >= 2 -> w)'
    ret = evalExpr(expr=expr, vardict=vardict, verbose=3)
    print ret, " Should be True"
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing findCycle')
    graph = [[1,2,3], [2], [], [2]]
    path = findCycle(graph=graph, verbose=3)
    print "path", path
    print
    graph = [[2,3], [0], [0,1], [2]]
    path = findCycle(graph=graph, verbose=3)
    print "path", path
    path = findCycle(graph=graph, W0ind=[0], verbose=3)
    print "path", path
