#!/usr/bin/env python

""" 
-------------------------------------
Rhtlputil Module --- Helper Functions
-------------------------------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 25, 2010
:Version: 0.1.0
"""

def evalExpr(expr='', vardict={}, verbose=0):
    imp_ind = expr.rfind('->')
    if (imp_ind < 0):
        vardict['True'] = True
        vardict['False'] = False
        expr = expr.replace('!', 'not ')
        expr = expr.replace('&', ' and ')
        expr = expr.replace('|', ' or ')
        expr = expr.replace('=', '==')
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
            right_expr_eval = evalexpr(right_expr, vardict, verbose)
            left_expr_eval = evalexpr(left_expr, vardict, verbose)
            if (not left_expr_eval or right_expr_eval):
                return True
            else:
                return False
        elif (num_paren == 1):
            right_expr_end_ind -= 1
            guarantee_formula = right_expr[:right_expr_end_ind]
            if (verbose > 1):
                print 'guarantee_formula = ' + guarantee_formula
            guarantee_eval = evalexpr(guarantee_formula, vardict, verbose)
            ass_start_ind = imp_ind
            num_paren = 0
            while (num_paren > -1 and ass_start_ind > 0):
                ass_start_ind -= 1
                if (expr[ass_start_ind] == '('):
                    num_paren -= 1
                elif (expr[ass_start_ind] == ')'):
                    num_paren += 1
            if (num_paren != -1):
                printError('ERROR proplogic.evalexpr: ' + \
                           'Invalid formula. Unbalanced parenthesis in ' + \
                               expr[:right_expr_end_ind+1])
                raise Exception("Invalid formula")
            assumption_formula = expr[ass_start_ind+1:imp_ind]
            if (verbose > 1):
                print 'assumption_formula = ' + assumption_formula
            assumption_eval = evalexpr(assumption_formula, vardict, verbose)
            imp_res = 'False'
            if (not assumption_eval or guarantee_eval):
                imp_res = 'True'
            if (verbose > 1):
                print 'implication result = ' + imp_res
            reduced_expr = expr[:ass_start_ind] + imp_res + right_expr[right_expr_end_ind+1:]
            if (verbose > 1):
                print 'reduced_expr = ' + reduced_expr
            ret = evalexpr(reduced_expr, vardict, verbose)
            return ret
        else:
            printError('ERROR proplogic.evalexpr: ' + \
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

def findCycle(graph=[], root, verbose=0):
