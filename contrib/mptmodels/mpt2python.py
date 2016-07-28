from tulip import hybrid
import polytope
import scipy.io

"""
Contains functions that will read a .mat file exported by the MATLAB function
mpt2python and import it to either a PwaSysDyn or a LtiSysDyn.

Stephanie Tsuei, June 2014
"""


def load(filename):
    data = scipy.io.loadmat(filename)
    islti = bool(data['islti'][0][0])
    ispwa = bool(data['ispwa'][0][0])

    if islti:
        sys = load_lti(data['A'], data['B'], data['domainA'],
                       data['domainB'], data['UsetA'], data['UsetB'])

    elif ispwa:
        nlti = len(data['A'][0])
        lti_systems = []

        for i in xrange(nlti):
            A = data['A'][0][i]
            B = data['B'][0][i]
            K = data['K'][0][i]
            domainA = data['domainA'][0][i]
            domainB = data['domainB'][0][i]
            UsetA = data['UsetA'][0][i]
            UsetB = data['UsetB'][0][i]

            ltisys = load_lti(A, B, K, domainA, domainB, UsetA, UsetB)
            lti_systems.append(ltisys)

        cts_ss = polytope.Polytope(data['ctsA'], data['ctsB'])
        sys = hybrid.PwaSysDyn(list_subsys=lti_systems, domain=cts_ss)

    return sys



def load_lti(A, B, K, domainA, domainB, UsetA, UsetB):
    domain = polytope.Polytope(domainA, domainB)
    Uset = polytope.Polytope(UsetA, UsetB)

    lti = hybrid.LtiSysDyn(A=A, B=B, K=K, domain=domain, Uset=Uset)
    return lti
