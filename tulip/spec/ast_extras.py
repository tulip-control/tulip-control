from tulip.spec.ast import *

def simplify(node):
    if isinstance(node, ASTBool):
        return node
    if isinstance(node, ASTVar):
        return node
    if isinstance(node, ASTAnd):
        l = simplify(node.op_l)
        r = simplify(node.op_r)
        if isinstance(l, ASTBool):
            if l.val:
                return r
            #Return False
            return l
        if isinstance(r, ASTBool):
            if r.val:
                return l
            #Return False
            return r
        return ASTAnd.new(l, r)
    if isinstance(node, ASTOr):
        l = simplify(node.op_l)
        r = simplify(node.op_r)
        if isinstance(l, ASTBool):
            if l.val:
                return l
            return r
        if isinstance(r, ASTBool):
            if r.val:
                return r
            return l
        return ASTOr.new(l, r)
    if isinstance(node, ASTNot):
        a = simplify(node.operand)
        if isinstance(a, ASTBool):
            if a.val:
                return ASTBool(None, None, ["FALSE"])
            else:
                return ASTBool(None, None, ["TRUE"])
        return node
    if isinstance(node, ASTUnTempOp):
        a = simplify(node.operand)
        return ASTUnTempOp.new(a, node.operator)
    raise ValueError

def conj_reduce(node, invars):
    if isinstance(node, ASTAnd):
        lnode = conj_reduce(node.op_l, invars)
        if isinstance(lnode, ASTBool) and not lnode.val:
            #conjunction contained contradiction
            return lnode
        rnode = conj_reduce(node.op_r, invars)
        if isinstance(rnode, ASTBool) and not rnode.val:
            #conjunction contained contradiction
            return rnode
        return ASTAnd.new(lnode, rnode)
    if isinstance(node, ASTBool):
        return node
    if isinstance(node, ASTVar):
        if node.val in invars:
            if invars[node.val] == 1:
                return ASTBool(None, None, ["TRUE"])
            else:
                return ASTBool(None, None, ["FALSE"])
        else:
            invars.update({node.val: 1})
            return node
    if isinstance(node, ASTNot):
        if node.operand.val in invars:
            if invars[node.operand.val] == 0:
                return ASTBool(None, None, ["TRUE"])
            else:
                return ASTBool(None, None, ["FALSE"])
        else:
            invars.update({node.operand.val: 0})
            return node

    raise ValueError

def disj_reduce(node, invars):
    if isinstance(node, ASTOr):
        lnode = disj_reduce(node.op_l, invars)
        if isinstance(lnode, ASTBool) and lnode.val:
            #disjunction is true
            return lnode
        rnode = disj_reduce(node.op_r, invars)
        if isinstance(rnode, ASTBool) and rnode.val:
            #disjunction is true
            return rnode
        return ASTOr.new(lnode, rnode)
    if isinstance(node, ASTBool):
        return node
    if isinstance(node, ASTVar):
        if node.val in invars:
            if invars[node.val] == 1:
                return ASTBool(None, None, ["TRUE"])
            else:
                return ASTBool(None, None, ["FALSE"])
        else:
            invars.update({node.val: 1})
            return node
    if isinstance(node, ASTNot):
        if node.operand.val in invars:
            if invars[node.operand.val] == 0:
                return ASTBool(None, None, ["TRUE"])
            else:
                return ASTBool(None, None, ["FALSE"])
        else:
            invars.update({node.operand.val: 0})
            return node

    raise ValueError

def nnf_op_to_var(node):
    #Assuming simplified nnf
    #Replaces temperal operators with variable: operator+" "+variable
    if isinstance(node, ASTAnd):
        return ASTAnd.new(nnf_op_to_var(node.op_l), nnf_op_to_var(node.op_r))
    if isinstance(node, ASTOr):
        return ASTOr.new(nnf_op_to_var(node.op_l), nnf_op_to_var(node.op_r))
    if isinstance(node, ASTNot):
        return ASTNot.new(nnf_op_to_var(node.operand))
    if isinstance(node, ASTUnTempOp):
        if isinstance(node.operand, ASTVar):
            var_name = node.operator+" "+node.operand.val
            return ASTVar(None, None, [var_name])
    return node

def DNF_reduce(node, all_vars):
    if isinstance(node, ASTOr):
        return ASTOr.new(simplify(DNF_reduce(node.op_l, all_vars)),
                         simplify(DNF_reduce(node.op_r, all_vars)))
    if isinstance(node, ASTAnd):
        invars = {}
        node =  conj_reduce(node, invars)
        all_vars.append(invars)
        print invars
        return simplify(node)
    else:
        #Single node (temporal, negation, var, bool)
        invars = {}
        node = conj_reduce(node, invars)
        all_vars.append(invars)
        print invars
        return simplify(node)


def to_NNF(node):
    if isinstance(node, ASTVar):
        if node.val.upper() == "TRUE":
            return ASTBool(None, None, ["TRUE"])
        if node.val.upper() == "FALSE":
            return ASTBool(None, None, ["FALSE"])
        return node

    if isinstance(node, ASTBool):
        return node

    if isinstance(node, ASTBinary):
        l = node.op_l
        r = node.op_r
        if isinstance(node, ASTXor):
            return simplify(to_NNF(ASTOr.new(ASTAnd.new(l, ASTNot.new(r)),
                                ASTAnd.new(ASTNot.new(l), r))))
        if isinstance(node, ASTImp):
            return simplify(to_NNF(ASTOr.new(ASTNot.new(l), r)))
        if isinstance(node, ASTBiImp):
            return simplify(to_NNF(ASTOr.new(ASTAnd.new(l, r),
                                             ASTAnd.new(ASTNot.new(l),
                                                        ASTNot.new(r)))))
        if isinstance(node, ASTBiTempOp):
            raise NotImplementedError
        if isinstance(node, ASTAnd):
            return simplify(ASTAnd.new(to_NNF(node.op_l), to_NNF(node.op_r)))
        if isinstance(node, ASTOr):
            return simplify(ASTOr.new(to_NNF(node.op_l), to_NNF(node.op_r)))

    if isinstance(node, ASTUnary):
        if isinstance(node, ASTNot):
            if isinstance(node.operand, ASTNot):
                return simplify(to_NNF(node.operand.operand))
            if isinstance(node.operand, ASTAnd):
                return simplify(ASTOr.new(to_NNF(ASTNot.new(node.operand.op_l)),
                                          to_NNF(ASTNot.new(node.operand.op_r))))
            if isinstance(node.operand, ASTOr):
                return simplify(ASTAnd.new(to_NNF(ASTNot.new(node.operand.op_l)),
                                           to_NNF(ASTNot.new(node.operand.op_r))))
            if isinstance(node.operand, ASTBool):
                if node.operand.val:
                    return ASTBool(None, None, ["FALSE"])
                else:
                    return ASTBool(None, None, ["TRUE"])
            if isinstance(node.operand, ASTVar):
                if node.operand.val.upper() == "TRUE":
                    return ASTBool(None, None, ["FALSE"])
                if node.operand.val.upper() == "FALSE":
                    return ASTBool(None, None, ["TRUE"])
                return node
            if isinstance(node.operand, ASTUnTempOp):
                temp = to_NNF(node.operand)
                if isinstance(temp, ASTUnTempOp):
                    return ASTNot.new(temp)
        if isinstance(node, ASTUnTempOp):
            operand = to_NNF(node.operand)
            #Move operand inwards and repeat
            if isinstance(operand, ASTNot):
                return to_NNF(ASTNot.new(to_NNF(ASTUnTempOp.new(operand.operand, node.operator))))
            if isinstance(operand, ASTAnd):
                return ASTAnd.new(to_NNF(ASTUnTempOp.new(operand.op_l, node.operator)),
                                  to_NNF(ASTUnTempOp.new(operand.op_r, node.operator)))
            if isinstance(operand, ASTOr):
                return ASTOr.new(to_NNF(ASTUnTempOp.new(operand.op_l, node.operator)),
                                 to_NNF(ASTUnTempOp.new(operand.op_r, node.operator)))
            if isinstance(operand, ASTBool):
                return ASTUnTempOp.new(operand, node.operator)
            if isinstance(operand, ASTVar):
                return ASTUnTempOp.new(operand, node.operator)

    raise ValueError

def NNF_to_DNF(node):
    if isinstance(node, ASTVar):
        return node
    if isinstance(node, ASTBool):
        return node
    if isinstance(node, ASTNot):
        return simplify(node)
    if isinstance(node, ASTUnTempOp):
        return simplify(node)
    if isinstance(node, ASTOr):
        l = NNF_to_DNF(node.op_l)
        r = NNF_to_DNF(node.op_r)
        return simplify(ASTOr.new(NNF_to_DNF(node.op_l),NNF_to_DNF(node.op_r)))
    if isinstance(node, ASTAnd):
        l = NNF_to_DNF(node.op_l)
        r = NNF_to_DNF(node.op_r)
        if isinstance(l, ASTOr):
            return simplify(ASTOr.new(NNF_to_DNF(ASTAnd.new(l.op_l, r)), \
                                      NNF_to_DNF(ASTAnd.new(l.op_r, r))))
        if isinstance(r, ASTOr):
            return simplify(ASTOr.new(NNF_to_DNF(ASTAnd.new(l, r.op_l)), \
                                      NNF_to_DNF(ASTAnd.new(l, r.op_r))))
        #Must be "And" or "Val/Bool/Neg"
        return conj_reduce(nnf_op_to_var(ASTAnd.new(l, r)),{})

    raise ValueError

def NNF_to_CNF(node):
    if isinstance(node, ASTVar):
        return node
    if isinstance(node, ASTBool):
        return node
    if isinstance(node, ASTNot):
        return simplify(node)
    if isinstance(node, ASTUnTempOp):
        return simplify(node)
    if isinstance(node, ASTAnd):
        l = NNF_to_CNF(node.op_l)
        r = NNF_to_CNF(node.op_r)
        return simplify(ASTAnd.new(NNF_to_CNF(node.op_l),NNF_to_CNF(node.op_r)))
    if isinstance(node, ASTOr):
        l = NNF_to_CNF(node.op_l)
        r = NNF_to_CNF(node.op_r)
        if isinstance(l, ASTAnd):
            return simplify(ASTAnd.new(simplify(NNF_to_CNF(ASTOr.new(l.op_l, r))), \
                                       simplify(NNF_to_CNF(ASTOr.new(l.op_r, r)))))
        if isinstance(r, ASTAnd):
            return simplify(ASTAnd.new(NNF_to_CNF(ASTOr.new(l, r.op_l)), \
                                       NNF_to_CNF(ASTOr.new(l, r.op_r))))
        #Must be "Or" or "Val/Bool/Neg"
        return disj_reduce(nnf_op_to_var(ASTOr.new(l, r)), {})

    raise ValueError