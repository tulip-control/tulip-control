import numpy


def matrix2str(A):
    """Convert a matrix A into a string

    @param A: a numpy.array matrix
    @rtype string
    """
    s = ""
    for x in numpy.nditer(A, order='F'):
        s = s + str(x) + ","

    return s


def polytope2str(p, name):
    """Convert a polytope into C code

    @param p: a polytope.polytope.Polytope
    @param name: the name of the polytope in C code
    @rtype string
    """
    k = p.A.shape[0]
    l = p.A.shape[1]
    # pik=k
    s = "idxint "+name+"k = "+str(k)+";\n"
    # pil=l
    s = s+"idxint "+name+"l = "+str(l)+";\n"
    # piA = [A11,A21,...,Ak1,A12,...,Ak2,...,A1l,...,Akl];
    s = s+"pfloat "+name+"A[] = {"+matrix2str(-1*p.A)+"};\n"
    # pib = [b1,b2,...,bk];
    s = s+"pfloat "+name+"b[] = {"+matrix2str(-1*p.b)+"};\n"
    # picenter = [p.chebXc1,...,p.chebXcl];
    s = s+"pfloat "+name+"center[] = {"+matrix2str(p.chebXc)+"};\n"
    return s
