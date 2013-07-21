#!/usr/bin/env python
"""
SCL; 31 Dec 2011.
"""

from tulip.conxml import *


# This test should be broken into units.
def conxml_test():
    A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float64)
    b = np.asarray(range(3))
    assert tagmatrix("A", A) == '<A type="matrix" r="3" c="3">1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0</A>'
    assert tagmatrix("b", b) == '<b type="matrix" r="3" c="1">0 1 2</b>'

    P = pc.Polytope(A, b, normalize=False)
    assert tagpolytope("P", P) == '<P type="polytope"><H type="matrix" r="3" c="3">1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0</H><K type="matrix" r="3" c="1">0.0 1.0 2.0</K></P>'

    li = range(7)
    assert taglist("trans", li) == '<trans>0 1 2 3 4 5 6</trans>'

    (li_out_name, li_out) = untaglist(taglist("trans", li))
    assert (li_out_name == "trans") and (li_out == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    li = ["hello, kitty", "it's GO TIME!"]
    assert taglist("arblist", li) == '<arblist><litem value="hello, kitty" /><litem value="it\'s GO TIME!" /></arblist>'

    (li_out_name, li_out) = untaglist(taglist("arblist", li), cast_f=None,
                                      namespace=None)
    assert (li_out_name == "arblist") and (li_out == ["hello, kitty", "it's GO TIME!"])

    di = {1:2, 3:4, 'X0':'boolean'}
    assert tagdict("env_vars", di) == '<env_vars><item key="1" value="2" /><item key="X0" value="boolean" /><item key="3" value="4" /></env_vars>'

    (di_out_name, di_out) = untagdict(tagdict("sys_vars", di),
                                      namespace=None)
    assert (di_out_name == "sys_vars") and (di_out == {'1': '2', 'X0': 'boolean', '3': '4'})

    (A_out_name, A_out) = untagmatrix(tagmatrix("A", A))
    assert (A_out_name == "A") and (np.all(np.all(A_out == A)))

    (b_out_name, b_out) = untagmatrix(tagmatrix("b", b),
                                      np_type=int)
    assert (b_out_name == "b") and (np.all(np.squeeze(b_out) == b))

    (P_out_name, P_out) = untagpolytope(tagpolytope("P", P),
                                        namespace=None)
    assert (P_out_name == "P") and (np.all(np.all(P_out.A == P.A))) and (np.all(np.squeeze(P_out.b) == np.squeeze(P.b)))

    list_prop  = range(10)
    list_poly = [pc.Polytope(A*(k+1), b*(k+1), normalize=False) for k in range(10)]
    R = pc.Region(list_prop=list_prop, list_poly=list_poly)
    assert tagregion(R) == '<region><list_prop>0 1 2 3 4 5 6 7 8 9</list_prop><reg_item type="polytope"><H type="matrix" r="3" c="3">1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0</H><K type="matrix" r="3" c="1">0.0 1.0 2.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0</H><K type="matrix" r="3" c="1">0.0 2.0 4.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">3.0 6.0 9.0 12.0 15.0 18.0 21.0 24.0 27.0</H><K type="matrix" r="3" c="1">0.0 3.0 6.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">4.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0 36.0</H><K type="matrix" r="3" c="1">0.0 4.0 8.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">5.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0 45.0</H><K type="matrix" r="3" c="1">0.0 5.0 10.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">6.0 12.0 18.0 24.0 30.0 36.0 42.0 48.0 54.0</H><K type="matrix" r="3" c="1">0.0 6.0 12.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">7.0 14.0 21.0 28.0 35.0 42.0 49.0 56.0 63.0</H><K type="matrix" r="3" c="1">0.0 7.0 14.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">8.0 16.0 24.0 32.0 40.0 48.0 56.0 64.0 72.0</H><K type="matrix" r="3" c="1">0.0 8.0 16.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">9.0 18.0 27.0 36.0 45.0 54.0 63.0 72.0 81.0</H><K type="matrix" r="3" c="1">0.0 9.0 18.0</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0</H><K type="matrix" r="3" c="1">0.0 10.0 20.0</K></reg_item></region>'

    (R_out_name, R_out) = untagregion(tagregion(R), cast_f_list=int, namespace=None)
    assert R_out.list_prop == R.list_prop
    for (ind, P) in enumerate(R_out.list_poly):
        R.list_poly[ind].b = np.asarray(R.list_poly[ind].b, dtype=np.float64)
        assert str(R.list_poly[ind]) == str(P)

    # Check fringe cases
    assert tagmatrix("omfg", []) == '<omfg type="matrix" r="0" c="0"></omfg>'
    assert tagpolytope("rtfm", None) == '<rtfm type="polytope"><H type="matrix" r="0" c="0"></H><K type="matrix" r="0" c="0"></K></rtfm>'
    assert untaglist(taglist("good_times", None)) == ("good_times", [])
