# Copyright (c) 2011, 2013 by California Institute of Technology
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
"""
Several functions that help in working with tulipcon XML files.

This module includes a few functions for parsing YAML files, providing
an easier way to specify transition systems.  See rsimple_example.yaml
in the "examples" directory for reference.
"""

import re
import xml.etree.ElementTree as ET
import numpy as np

from hybrid import LtiSysDyn
import automaton
from abstract.prop2part import prop2part, PropPreservingProposition
import polytope as pc
#import jtlvint  # XXX: not yet overhauled
import errorprint as ep
from spec import GRSpec

from StringIO import StringIO
try:
    import yaml
except ImportError:
    yaml = None  # Thus calling a function that depends on PyYAML will
                 # lead to an exception.


#################
# Problem Types #
#################
INCOMPLETE_PROB = -1
NONE_PROB = 0
SYNTH_PROB = 1
RHTLP_PROB = 2


###############
# XML Globals #
###############
DEFAULT_NAMESPACE = "http://tulip-control.sourceforge.net/ns/0"


def readXMLfile(fname, verbose=0):
    """Read tulipcon XML string directly from a file.

    Returns 3-tuple, as described in *loadXML* docstring.  (This is a
    convenience method, mostly only wrapping *loadXML*.)
    """
    with open(fname, "r") as f:
        x = f.read()
    return loadXML(x, verbose=verbose)

def loadXML(x, verbose=0, namespace=DEFAULT_NAMESPACE):
    """Return ({SynthesisProb,PropPreservingPartition}, CtsSysDyn, Automaton).

    The first element of the returned tuple depends on the problem
    type detected. If type "none", then it is an instance of
    PropPreservingPartition, e.g., as returned by the function
    discretize in module discretize. If of type "synth" or "rhtlp",
    then an instance of SynthesisProb or a child class is returned.

    Any empty or missing items are set to None, or an exception is
    raised if the missing item is required.

    The argument x can also be an instance of
    xml.etree.ElementTree._ElementInterface ; this is mainly for
    internal use, e.g. by the function untagpolytope and some
    load/dumpXML methods elsewhere.

    To easily read and process this string from a file, instead call the method
    *readXMLfile*
    """
    if not isinstance(x, str) and not isinstance(x, ET._ElementInterface):
        raise TypeError("tag to be parsed must be given as a string or ElementTree._ElementInterface.")

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x

    if (namespace is None) or (len(namespace) == 0):
        ns_prefix = ""
    else:
        ns_prefix = "{"+namespace+"}"
        
    if elem.tag != ns_prefix+"tulipcon":
        raise TypeError("root tag should be tulipcon.")
    if ("version" not in elem.attrib.keys()):
        raise ValueError("unversioned tulipcon XML string.")
    if int(elem.attrib["version"]) != 0:
        raise ValueError("unsupported tulipcon XML version: "+str(elem.attrib["version"]))

    ptype_tag = elem.find(ns_prefix+"prob_type")
    if ptype_tag is None:
        ptype = INCOMPLETE_PROB
    elif ptype_tag.text == "none":
        ptype = NONE_PROB
    elif ptype_tag.text == "synth":
        ptype = SYNTH_PROB
    elif ptype_tag.text == "rhtlp":
        ptype = RHTLP_PROB
    else:
        raise ValueError("Unrecognized prob_type: "+str(ptype_tag.text))
    
    # Build CtsSysDyn, or set to None
    c_dyn = elem.find(ns_prefix+"c_dyn")
    if c_dyn is None:
        sys_dyn = None
    else:
        (tag_name, A) = untagmatrix(c_dyn.find(ns_prefix+"A"))
        (tag_name, B) = untagmatrix(c_dyn.find(ns_prefix+"B"))
        (tag_name, E) = untagmatrix(c_dyn.find(ns_prefix+"E"))
        (tag_name, K) = untagmatrix(c_dyn.find(ns_prefix+"K"))
        (tag_name, Uset) = untagpolytope(c_dyn.find(ns_prefix+"U_set"))
        (tag_name, Wset) = untagpolytope(c_dyn.find(ns_prefix+"W_set"))
        sys_dyn = LtiSysDyn(A, B, E, K, Uset, Wset)

    # Extract LTL specification
    s_elem = elem.find(ns_prefix+"spec")
    if s_elem.find(ns_prefix+"env_init") is not None:  # instance of GRSpec style
        spec = GRSpec()
        for spec_tag in ["env_init", "env_safety", "env_prog",
                         "sys_init", "sys_safety", "sys_prog"]:
            if s_elem.find(ns_prefix+spec_tag) is None:
                raise ValueError("invalid specification in tulipcon XML string.")
            (tag_name, li) = untaglist(s_elem.find(ns_prefix+spec_tag),
                                       cast_f=str, namespace=namespace)
            li = [v.replace("&lt;", "<") for v in li]
            li = [v.replace("&gt;", ">") for v in li]
            li = [v.replace("&amp;", "&") for v in li]
            setattr(spec, spec_tag, li)

    else:  # assume, guarantee strings style
        spec = ["", ""]
        if s_elem.find(ns_prefix+"assume") is not None:
            spec[0] = s_elem.find(ns_prefix+"assume").text
        if s_elem.find(ns_prefix+"guarantee") is not None:
            spec[1] = s_elem.find(ns_prefix+"guarantee").text
        for k in [0, 1]:  # Undo special character encoding
            if spec[k] is None:
                spec[k] = ""
            else:
                spec[k] = spec[k].replace("&lt;", "<")
                spec[k] = spec[k].replace("&gt;", ">")
                spec[k] = spec[k].replace("&amp;", "&")

    # Build Automaton
    aut_elem = elem.find(ns_prefix+"aut")
    if aut_elem is None \
            or ((aut_elem.text is None) and len(aut_elem.getchildren()) == 0):
        aut = None
    else:
        aut = automaton.Automaton()
        if not aut.loadXML(aut_elem, namespace=DEFAULT_NAMESPACE):
            raise ValueError("failed to read Automaton from given tulipcon XML string.")
            aut = None

    # Discrete dynamics, if available
    d_dyn = elem.find(ns_prefix+"d_dyn")
    if d_dyn is None:
        prob = None
    else:
        (tag_name, env_vars) = untagdict(elem.find(ns_prefix+"env_vars"))
        (tag_name, sys_disc_vars) = untagdict(elem.find(ns_prefix+"sys_vars"))
        if ((d_dyn.find(ns_prefix+"domain") is None)
            and (d_dyn.find(ns_prefix+"trans") is None)
            and (d_dyn.find(ns_prefix+"prop_symbols") is None)):
            disc_dynamics = None
        else:
            (tag_name, domain) = untagpolytope(d_dyn.find(ns_prefix+"domain"))
            (tag_name, trans) = untagmatrix(d_dyn.find(ns_prefix+"trans"), np_type=np.uint8)
            (tag_name, prop_symbols) = untaglist(d_dyn.find(ns_prefix+"prop_symbols"), cast_f=str)
            region_elem = d_dyn.find(ns_prefix+"regions")
            list_region = []
            if region_elem is not None:
                region_items = region_elem.findall(ns_prefix+"region")
                if region_items is not None and len(region_items) > 0:
                    for region_item in region_items:
                        (tag_name, R) = untagregion(region_item, cast_f_list=int,
                                                    np_type_P=np.float64)
                        list_region.append(R)

            orig_map_elem = d_dyn.find(ns_prefix+"orig_map")
            orig_part_elem = d_dyn.find(ns_prefix+"orig_partition")
            if (orig_map_elem is None) or (orig_part_elem is None):
                orig_list_region = None
                orig = None
            else:
                (tag_name, orig) = untaglist(orig_map_elem, cast_f=int)
                cell_items = orig_part_elem.findall(ns_prefix+"cell")
                orig_list_region = []
                if cell_items is not None and len(cell_items) > 0:
                    for cell_item in cell_items:
                        (tag_name, P) = untagpolytope(cell_item)
                        orig_list_region.append(P)
                        
            disc_dynamics = PropPreservingPartition(domain=domain,
                                                    num_prop=len(prop_symbols),
                                                    list_region=list_region,
                                                    num_regions=len(list_region),
                                                    adj=0,
                                                    trans=trans,
                                                    list_prop_symbol=prop_symbols,
                                                    orig_list_region=orig_list_region,
                                                    orig=orig)

        # Build appropriate ``problem'' instance
        if ptype == SYNTH_PROB:
            raise Exception("conxml.loadXML: SYNTH_PROB type is defunct, possibly temporarily")
            # prob = jtlvint.generateJTLVInput(env_vars=env_vars,
            #                                  sys_disc_vars=sys_disc_vars,
            #                                  spec=spec,
            #                                  disc_props={},
            #                                  disc_dynamics=disc_dynamics,
            #                                  smv_file="", spc_file="", verbose=2)
        elif ptype == RHTLP_PROB:
            raise Exception("conxml.loadXML: RHTLP_PROB type is defunct, possibly temporarily")
            # if disc_dynamics is not None:
            #     prob = rhtlp.RHTLPProb(shprobs=[], Phi="True", discretize=False,
            #                            env_vars=env_vars,
            #                            sys_disc_vars=sys_disc_vars,
            #                            disc_dynamics=disc_dynamics,
            #                            #cont_props=cont_props,
            #                            spec=spec)
            # else:
            #     prob = rhtlp.RHTLPProb(shprobs=[], Phi="True", discretize=False,
            #                            env_vars=env_vars,
            #                            sys_disc_vars=sys_disc_vars,
            #                            #cont_props=cont_props,
            #                            spec=spec)
        elif ptype == NONE_PROB:
            prob = disc_dynamics
        else: #ptype == INCOMPLETE_PROB
            prob = None
    return (prob, sys_dyn, aut)


def dumpXML(prob=None, spec=['',''], sys_dyn=None, aut=None,
            disc_dynamics=None,
            synthesize_aut=False, verbose=0, pretty=False):
    """Return tulipcon XML string.

    prob is an instance of SynthesisProb or a child class, such as
    RHTLPProb (both defined in rhtlp.py).  sys_dyn is an instance of
    CtsSysDyn (defined in discretize.py), or None. If None, then
    continuous dynamics are considered empty (i.e., there are none).

    disc_dynamics is an instance of PropPreservingPartition, such as
    returned by the function discretize in module discretize.  This
    argument is supported to permit entire avoidance of SynthesisProb
    or related classes.

    If pretty is True, then use indentation and newlines to make the
    resulting XML string more visually appealing.
    
    aut is an instance of Automaton.  If None (rather than an
    instance), then an empty <aut> tag is written.

    spec may be an instance of GRSpec or a list.  If of GRSpec, then
    it is formed as expected.  If spec is a list, then first element
    of "assume" string, and second element of "guarantee" string.
    spec=None is also accepted, in which case the specification is
    considered empty, but note that this could cause problems later
    unless some content is introduced.

    ** synthesize_aut flag is IGNORED **
    If synthesize_aut is True, then if prob.__realizable is not None,
    use its value to determine whether a previously computed \*.aut
    file should be read.  Else (if prob.__realizable is None or
    False), then try to compute automaton (saving to file
    prob.__jtlvfile + '.aut', as usual). If synthesize_aut is False,
    then just save an empty <aut></aut>.

    ** verbose flag is IGNORED (but that will change...) **

    To easily save the result here to a file, instead call the method
    *writeXMLfile*
    """
    if prob is not None:
        raise Exception("conxml.dumpXML: non-empty prob types are not supported, possibly temporarily")
    # if prob is not None and not isinstance(prob, (rhtlp.SynthesisProb, rhtlp.RHTLPProb)):
    #     raise TypeError("prob should be an instance (or child) of rhtlp.SynthesisProb.")
    if sys_dyn is not None and not isinstance(sys_dyn, LtiSysDyn):
        raise TypeError("sys_dyn should be an instance of discretizeM.CtsSysDyn or None.")
    if disc_dynamics is not None and not isinstance(disc_dynamics, PropPreservingPartition):
        raise TypeError("disc_dynamics should be an instance of PropPreservingPartition or None.")
    if aut is not None and not isinstance(aut, automaton.Automaton):
        raise TypeError("aut should be an instance of Automaton or None.")
    if spec is not None and not isinstance(spec, (list, GRSpec)):
        raise TypeError("spec should be an instance of list or GRSpec")

    if pretty:
        nl = "\n"  # Newline
        idt = "  "  # Indentation
    else:
        nl = ""
        idt = ""
    idt_level = 0

    output = '<?xml version="1.0" encoding="UTF-8"?>'+nl
    output += '<tulipcon xmlns="http://tulip-control.sourceforge.net/ns/0" version="0">'+nl
    idt_level += 1
    if sys_dyn is not None:
        output += idt*idt_level+'<prob_type>'
        if prob is None:
            output += 'none'
        else:
            raise Exception("conxml.dumpXML: non-empty prob types are not supported, possibly temporarily")
        # if isinstance(prob, rhtlp.RHTLPProb):  # Beware of order and inheritance
        #     output += 'rhtlp'
        # elif isinstance(prob, rhtlp.SynthesisProb):
        #     output += 'synth'
        # else: # prob is None
        #     output += 'none'
        output += '</prob_type>'+nl

        output += idt*idt_level+'<c_dyn>'+nl
        idt_level += 1
        output += idt*idt_level+tagmatrix("A",sys_dyn.A)+nl
        output += idt*idt_level+tagmatrix("B",sys_dyn.B)+nl
        output += idt*idt_level+tagmatrix("E",sys_dyn.E)+nl
        output += idt*idt_level+tagmatrix("K",sys_dyn.K)+nl

        # Need facility for setting sample period; maybe as new attribute in CtsSysDyn class?
        output += idt*idt_level+'<sample_period>1</sample_period>'+nl

        output += idt*idt_level+tagpolytope("U_set", sys_dyn.Uset)+nl
        output += idt*idt_level+tagpolytope("W_set", sys_dyn.Wset)+nl
        idt_level -= 1
        output += idt*idt_level+'</c_dyn>'+nl

        if prob is not None:
            output += tagdict("env_vars", prob.getEnvVars(), pretty=pretty, idt_level=idt_level)
            output += tagdict("sys_vars", prob.getSysVars(), pretty=pretty, idt_level=idt_level)
        else:
            output += tagdict("env_vars", dict([(v,"boolean") for v in spec.env_vars]), pretty=pretty, idt_level=idt_level)
            output += tagdict("sys_vars", dict([(v,"boolean") for v in spec.sys_vars]), pretty=pretty, idt_level=idt_level)

    if spec is None:
        output += idt*idt_level+'<spec><assume></assume><guarantee></guarantee></spec>'+nl
    elif isinstance(spec, GRSpec):
        # Copy out copies of GRSpec attributes
        spec_dict = {"env_init": spec.env_init, "env_safety": spec.env_safety,
                     "env_prog": spec.env_prog,
                     "sys_init": spec.sys_init, "sys_safety": spec.sys_safety,
                     "sys_prog": spec.sys_prog}

        # Map special XML characters to safe forms
        for k in spec_dict.keys():
            if isinstance(spec_dict[k], str):
                spec_dict[k] = spec_dict[k].replace("<", "&lt;")
                spec_dict[k] = spec_dict[k].replace(">", "&gt;")
                spec_dict[k] = spec_dict[k].replace("&", "&amp;")
            else:
                spec_dict[k] = [v.replace("<", "&lt;") for v in spec_dict[k]]
                spec_dict[k] = [v.replace(">", "&gt;") for v in spec_dict[k]]
                spec_dict[k] = [v.replace("&", "&amp;") for v in spec_dict[k]]
        
        # Finally, dump XML tags
        output += idt*idt_level+'<spec>'+nl
        for (k, v) in spec_dict.items():
            if isinstance(v, str):
                v = [v,]
            output += idt*(idt_level+1)+ taglist(k, v) +nl
        output += idt*idt_level+'</spec>'+nl

    else:
        for k in [0,1]:
            spec[k] = spec[k].replace("<", "&lt;")
            spec[k] = spec[k].replace(">", "&gt;")
            spec[k] = spec[k].replace("&", "&amp;")
        output += idt*idt_level+'<spec><assume>'+spec[0]+'</assume>'+nl
        output += idt*(idt_level+1)+'<guarantee>'+spec[1]+'</guarantee></spec>'+nl

    if (disc_dynamics is None) and (prob is not None):
        disc_dynamics = prob.getDiscretizedDynamics()

    if disc_dynamics is not None:
        output += idt*idt_level+'<d_dyn>'+nl
        idt_level += 1

        output += idt*idt_level+tagpolytope("domain", disc_dynamics.domain)+nl
        output += idt*idt_level+tagmatrix("trans", disc_dynamics.trans)+nl
        output += idt*idt_level+taglist("prop_symbols",
                                        disc_dynamics.list_prop_symbol)+nl
        output += idt*idt_level+'<regions>'+nl
        idt_level += 1
        if disc_dynamics.list_region is not None and len(disc_dynamics.list_region) > 0:
            for R in disc_dynamics.list_region:
                output += tagregion(R, pretty=pretty, idt_level=idt_level)
        idt_level -= 1
        output += idt*idt_level+'</regions>'+nl
        if disc_dynamics.orig_list_region is not None:
            output += idt*idt_level+taglist("orig_map", disc_dynamics.orig)+nl
            output += idt*idt_level+'<orig_partition>'+nl
            idt_level += 1
            for P in disc_dynamics.orig_list_region:
                output += idt*idt_level+tagpolytope("cell", P)+nl
            idt_level -= 1
            output += idt*idt_level+'</orig_partition>'+nl
        idt_level -= 1
        output += idt*idt_level+'</d_dyn>'+nl

    if aut is None:
        output += idt*idt_level+'<aut></aut>'+nl
    else:
        output += aut.dumpXML(pretty=pretty, idt_level=idt_level)+nl

    output += idt_level*idt+'<extra></extra>'+nl

    idt_level -= 1
    assert idt_level == 0
    output += '</tulipcon>'+nl
    return output

def writeXMLfile(fname, prob=None, spec=['',''], sys_dyn=None, aut=None,
                 disc_dynamics=None,
                 synthesize_aut=False, verbose=0, pretty=False):
    """Write tulipcon XML string directly to a file.

    Returns nothing.  (This is a convenience method, mostly only
    wrapping *dumpXML*.)
    """
    with open(fname, "w") as f:
        f.write(dumpXML(prob=prob, spec=spec, sys_dyn=sys_dyn, aut=aut,
                        disc_dynamics=disc_dynamics,
                        synthesize_aut=synthesize_aut, verbose=verbose,
                        pretty=pretty))
    return


def loadXMLtrans(x, namespace=DEFAULT_NAMESPACE):
    """Read only the continuous transition system from a tulipcon XML string.

    I.e., the continuous dynamics (A, B, etc.), and the
    proposition-preserving partition, along with its reachability
    data.

    Return (sys_dyn, disc_dynamics, horizon).

    Raise exception if critical error.
    """
    if not isinstance(x, str) and not isinstance(x, ET._ElementInterface):
        raise TypeError("tag to be parsed must be given as a string or ElementTree._ElementInterface.")

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x

    if (namespace is None) or (len(namespace) == 0):
        ns_prefix = ""
    else:
        ns_prefix = "{"+namespace+"}"

    if elem.tag != ns_prefix+"tulipcon":
        raise TypeError("root tag should be tulipcon.")
    if ("version" not in elem.attrib.keys()):
        raise ValueError("unversioned tulipcon XML string.")
    if int(elem.attrib["version"]) != 0:
        raise ValueError("unsupported tulipcon XML version: "+str(elem.attrib["version"]))

    # Build CtsSysDyn, or set to None
    c_dyn = elem.find(ns_prefix+"c_dyn")
    if c_dyn is None:
        sys_dyn = None
    else:
        (tag_name, A) = untagmatrix(c_dyn.find(ns_prefix+"A"))
        (tag_name, B) = untagmatrix(c_dyn.find(ns_prefix+"B"))
        (tag_name, E) = untagmatrix(c_dyn.find(ns_prefix+"E"))
        (tag_name, Uset) = untagpolytope(c_dyn.find(ns_prefix+"U_set"))
        (tag_name, Wset) = untagpolytope(c_dyn.find(ns_prefix+"W_set"))
        sys_dyn = LtiSysDyn(A, B, E, [], Uset, Wset)

    # Discrete dynamics, if available
    d_dyn = elem.find(ns_prefix+"d_dyn")
    if d_dyn is None:
        horizon = None
        disc_dynamics = None
    else:
        if not d_dyn.attrib.has_key("horizon"):
            raise ValueError("missing horizon length used for reachability computation.")
        horizon = int(d_dyn.attrib["horizon"])
        if (d_dyn.find(ns_prefix+"domain") is None) \
                and (d_dyn.find(ns_prefix+"trans") is None) \
                and (d_dyn.find(ns_prefix+"prop_symbols") is None):
            disc_dynamics = None
        else:
            (tag_name, domain) = untagpolytope(d_dyn.find(ns_prefix+"domain"))
            (tag_name, trans) = untagmatrix(d_dyn.find(ns_prefix+"trans"), np_type=np.uint8)
            (tag_name, prop_symbols) = untaglist(d_dyn.find(ns_prefix+"prop_symbols"), cast_f=str)
            region_elem = d_dyn.find(ns_prefix+"regions")
            list_region = []
            if region_elem is not None:
                region_items = d_dyn.find(ns_prefix+"regions").findall(ns_prefix+"item")
                if region_items is not None and len(region_items) > 0:
                    for region_item in region_items:
                        (tag_name, R) = untagregion(region_item, cast_f_list=int,
                                                    np_type_P=np.float64)
                        list_region.append(R)

            disc_dynamics = PropPreservingPartition(domain=domain,
                                                    num_prop=len(prop_symbols),
                                                    list_region=list_region,
                                                    num_regions=len(list_region),
                                                    adj=0,
                                                    trans=trans,
                                                    list_prop_symbol=prop_symbols)

    return (sys_dyn, disc_dynamics, horizon)


def dumpXMLtrans(sys_dyn, disc_dynamics, horizon, extra="",
                 pretty=False):
    """Return tulipcon XML containing only a continuous transition system.

    The argument extra (as a string) is copied verbatim into the
    <extra> element.

    The "pretty" flag has the same meaning as elsewhere (e.g., see
    docstring for dumpXML function).
    """
    if not isinstance(sys_dyn, LtiSysDyn):
        raise TypeError("sys_dyn must be an instance of discretizeM.CtsSysDyn")
    if not isinstance(disc_dynamics, PropPreservingPartition):
        raise TypeError("disc_dynamics must be an instance of PropPreservingPartition")
    
    if pretty:
        nl = "\n"  # Newline
        idt = "  "  # Indentation
    else:
        nl = ""
        idt = ""
    idt_level = 0

    output = '<?xml version="1.0" encoding="UTF-8"?>'+nl
    output += '<tulipcon xmlns="http://tulip-control.sourceforge.net/ns/0" version="0">'+nl
    idt_level += 1
    output += idt*idt_level+'<c_dyn>'+nl
    idt_level += 1
    output += idt*idt_level+tagmatrix("A",sys_dyn.A)+nl
    output += idt*idt_level+tagmatrix("B",sys_dyn.B)+nl
    output += idt*idt_level+tagmatrix("E",sys_dyn.E)+nl

    # Need facility for setting sample period; maybe as new attribute in CtsSysDyn class?
    output += idt*idt_level+'<sample_period>1</sample_period>'+nl

    output += idt*idt_level+tagpolytope("U_set", sys_dyn.Uset)+nl
    output += idt*idt_level+tagpolytope("W_set", sys_dyn.Wset)+nl
    idt_level -= 1
    output += idt*idt_level+'</c_dyn>'+nl

    output += idt*idt_level+'<d_dyn horizon="'+str(horizon)+'">'+nl
    idt_level += 1

    output += idt*idt_level+tagpolytope("domain", disc_dynamics.domain)+nl
    output += idt*idt_level+tagmatrix("trans", disc_dynamics.trans)+nl
    output += idt*idt_level+taglist("prop_symbols",
                                    disc_dynamics.list_prop_symbol)+nl
    output += idt*idt_level+'<regions>'+nl
    idt_level += 1
    if disc_dynamics.list_region is not None and len(disc_dynamics.list_region) > 0:
        for R in disc_dynamics.list_region:
            output += tagregion(R, pretty=pretty, idt_level=idt_level)
    idt_level -= 1
    output += idt*idt_level+'</regions>'+nl
    if disc_dynamics.orig_list_region is not None:
        output += idt*idt_level+taglist("orig_map", disc_dynamics.orig)+nl
        output += idt*idt_level+'<orig_partition>'+nl
        idt_level += 1
        for P in disc_dynamics.orig_list_region:
            output += idt*idt_level+tagpolytope("cell", P)+nl
        idt_level -= 1
        output += idt*idt_level+'</orig_partition>'+nl
    idt_level -= 1
    output += idt*idt_level+'</d_dyn>'+nl

    output += idt_level*idt+'<extra>'+extra+'</extra>'+nl

    idt_level -= 1
    assert idt_level == 0
    output += '</tulipcon>'+nl
    return output


def untaglist(x, cast_f=float,
              namespace=DEFAULT_NAMESPACE):
    """Extract list from given tulipcon XML tag (string).

    Use function cast_f for type-casting extracting element strings.
    The default is float, but another common case is cast_f=int (for
    "integer").  If cast_f is set to None, then items are left as
    extracted, i.e. as strings.

    The argument x can also be an instance of
    xml.etree.ElementTree._ElementInterface ; this is mainly for
    internal use, e.g. by the function untagpolytope and some
    load/dumpXML methods elsewhere.

    Return result as 2-tuple, containing name of the tag (as a string)
    and the list obtained from it.
    """
    if not isinstance(x, str) and not isinstance(x, ET._ElementInterface):
        raise TypeError("tag to be parsed must be given as a string or ElementTree._ElementInterface.")

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x

    if (namespace is None) or (len(namespace) == 0):
        ns_prefix = ""
    else:
        ns_prefix = "{"+namespace+"}"

    # Extract list
    if cast_f is None:
        cast_f = str
    litems = elem.findall(ns_prefix+'litem')
    if len(litems) > 0:
        li = [cast_f(k.attrib['value']) for k in litems]
    elif elem.text is None:
        li = []
    else:
        li = [cast_f(k) for k in elem.text.split()]
        
    return (elem.tag, li)

def untagdict(x, cast_f_keys=None, cast_f_values=None,
              namespace=DEFAULT_NAMESPACE):
    """Extract list from given tulipcon XML tag (string).

    Use functions cast_f_keys and cast_f_values for type-casting
    extracting key and value strings, respectively, or None.  The
    default is None, which means the extracted keys (resp., values)
    are left untouched (as strings), but another common case is
    cast_f_values=int (for "integer") or cast_f_values=float (for
    "floating-point numbers"), while leaving cast_f_keys=None to
    indicate dictionary keys are strings.

    The argument x can also be an instance of
    xml.etree.ElementTree._ElementInterface ; this is mainly for
    internal use, e.g. by the function untagpolytope and some
    load/dumpXML methods elsewhere.

    Return result as 2-tuple, containing name of the tag (as a string)
    and the dictionary obtained from it.
    """
    if not isinstance(x, str) and not isinstance(x, ET._ElementInterface):
        raise TypeError("tag to be parsed must be given as a string or ElementTree._ElementInterface.")

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x

    if (namespace is None) or (len(namespace) == 0):
        ns_prefix = ""
    else:
        ns_prefix = "{"+namespace+"}"

    # Extract dictionary
    items_li = elem.findall(ns_prefix+'item')
    if cast_f_keys is None:
        cast_f_keys = str
    if cast_f_values is None:
        cast_f_values = str
    di = dict()
    for item in items_li:
        # N.B., we will overwrite duplicate keys without warning!
        di[cast_f_keys(item.attrib['key'])] = cast_f_values(item.attrib['value'])
    return (elem.tag, di)

def untagmatrix(x, np_type=np.float64):
    """Extract matrix from given tulipcon XML tag (string).

    np_type is the NumPy type into which to cast read string.  The
    default is the most common, 64-bit floating point.

    The argument x can also be an instance of
    xml.etree.ElementTree._ElementInterface ; this is mainly for
    internal use, e.g. by the function untagpolytope and some
    load/dumpXML methods elsewhere.

    Return result as 2-tuple, containing name of the tag (as a string)
    and the NumPy ndarray obtained from it.
    """
    if not isinstance(x, str) and not isinstance(x, ET._ElementInterface):
        raise TypeError("tag to be parsed must be given as a string or ElementTree._ElementInterface.")

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x  # N.B., just passing the reference.
    if elem.attrib['type'] != "matrix":
        raise ValueError("tag should indicate type of ``matrix''.")
    
    if elem.text is None:
        x_mat = np.array([])  # Handle special empty case.
    else:
        num_rows = int(elem.attrib['r'])
        num_cols = int(elem.attrib['c'])
        x_mat = np.array([k for k in elem.text.split()], dtype=np_type)
        x_mat = x_mat.reshape(num_rows, num_cols)
    return (elem.tag, x_mat)

def untagpolytope(x, np_type=np.float64, namespace=DEFAULT_NAMESPACE):
    """Extract polytope from given tuilpcon XML tag (string).

    Essentially calls untagmatrix and gathers results into Polytope
    instance.
    
    The argument x can also be an instance of
    xml.etree.ElementTree._ElementInterface ; this is mainly for
    internal use, e.g. by the function untagpolytope and some
    load/dumpXML methods elsewhere.

    Return result as 2-tuple, containing name of the tag (as a string)
    and the Polytope (as defined in tulip.polytope_computations).
    """
    if not isinstance(x, str) and not isinstance(x, ET._ElementInterface):
        raise TypeError("tag to be parsed must be given as a string or ElementTree._ElementInterface.")

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x
    if elem.attrib['type'] != "polytope":
        raise ValueError("tag should indicate type of ``polytope''.")

    if (namespace is None) or (len(namespace) == 0):
        ns_prefix = ""
    else:
        ns_prefix = "{"+DEFAULT_NAMESPACE+"}"

    h_tag = elem.find(ns_prefix+'H')
    k_tag = elem.find(ns_prefix+'K')

    (H_out_name, H_out) = untagmatrix(h_tag, np_type=np_type)
    (K_out_name, K_out) = untagmatrix(k_tag, np_type=np_type)

    return (elem.tag, pc.Polytope(H_out, K_out, normalize=False))

def untagregion(x, cast_f_list=str, np_type_P=np.float64,
                namespace=DEFAULT_NAMESPACE):
    """Extract region from given tulipcon XML tag (string).

    The argument x can also be an instance of
    xml.etree.ElementTree._ElementInterface ; this is mainly for
    internal use, e.g. by the function untagpolytope and some
    load/dumpXML methods elsewhere.

    Type-casting is handled as described by the untaglist and
    untagpolytope functions, which are passed the references
    cast_f_list and np_type_P, respectively.

    Return the result as 2-tuple, containing name of the tag (as a string)
    and the Region (as defined in tulip.polytope_computations).
    """
    if not isinstance(x, str) and not isinstance(x, ET._ElementInterface):
        raise TypeError("tag to be parsed must be given as a string or ElementTree._ElementInterface.")

    if (namespace is None) or (len(namespace) == 0):
        ns_prefix = ""
    else:
        ns_prefix = "{"+DEFAULT_NAMESPACE+"}"

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x
    if elem.tag != ns_prefix+"region":
        raise ValueError("tag must be an instance of ``region''.")

    (tag_name, list_prop) = untaglist(elem.find(ns_prefix+"list_prop"), cast_f=cast_f_list)
    if list_prop is None:
        list_prop = []

    poly_tags = elem.findall(ns_prefix+"reg_item")
    list_poly = []
    if poly_tags is not None and len(poly_tags) > 0:
        for P_elem in poly_tags:
            (tag_name, P) = untagpolytope(P_elem, np_type=np_type_P,
                                          namespace=namespace)
            list_poly.append(P)

    return (elem.tag, pc.Region(list_poly=list_poly, list_prop=list_prop))

def tagdict(name, di, pretty=False, idt_level=0):
    """Create tag that basically stores a dictionary object.

    If pretty is True, then use indentation and newlines to make the
    resulting XML string more visually appealing.  idt_level is the
    base indentation level on which to create automaton string.  This
    level is only relevant if pretty=True.

    N.B., all keys and values are treated as strings (and frequently
    wrapped in str() to force this behavior).

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if not isinstance(name, str):
        raise TypeError("tag name must be a string.")

    if pretty:
        nl = "\n"  # Newline
        idt = "  "  # Indentation
    else:
        nl = ""
        idt = ""
    
    output = idt*idt_level+'<'+name+'>'+nl
    for (k, v) in di.items():
        output += idt*(idt_level+1)+'<item key="' + str(k) \
            + '" value="' + str(v) + '" />'+nl
    output += idt*idt_level+'</'+name+'>'+nl
    return output

def tagpolytope(name, P):
    """Create tag of type "Polytope", with given name.

    Polytope is as defined in tulip.polytope_computations module.

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if not isinstance(name, str):
        raise TypeError("tag name must be a string.")
    # Handle nil polytope case
    if P is None or P == []:
        P = pc.Polytope(np.array([]), np.array([]), normalize=False)
    output = '<'+name+' type="polytope">'
    output += tagmatrix("H", P.A)
    output += tagmatrix("K", P.b)
    output += '</'+name+'>'
    return output

def taglist(name, li, no_litem=False):
    """Create tag that basically stores a list object.

    N.B., all list elements are treated as strings (and wrapped in
    str() to force this behavior).

    By default, if an element in given list is of type string, then
    each element is placed in its own <litem> tag. This allows
    elements to be arbitrary (printable) strings in an XML file. To
    disable this, invoke taglist with no_litem=True.

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if not isinstance(name, str):
        raise TypeError("tag name must be a string.")

    output = '<'+name+'>'
    if li is not None:
        has_str = False
        for k in li:
            if isinstance(k, str):
                has_str = True
                break
        if has_str and not no_litem:
            for k in li:
                output += '<litem value="'+str(k)+'" />'
        else:
            output += ' '.join([str(k) for k in li])
    output += '</'+name+'>'
    return output

def tagregion(R, pretty=False, idt_level=0):
    """Create tag of type "Region."

    Region is as defined in tulip.polytope_computations module.

    If pretty is True, then use indentation and newlines to make the
    resulting XML string more visually appealing.  idt_level is the
    base indentation level on which to create automaton string.  This
    level is only relevant if pretty=True.

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if pretty:
        nl = "\n"  # Newline
        idt = "  "  # Indentation
    else:
        nl = ""
        idt = ""

    # Handle nil Region case
    if R is None or R == []:
        R = pc.Region(list_poly=[], list_prop=[])
    output = idt*idt_level+'<region>'+nl
    idt_level += 1
    output += idt*idt_level+taglist("list_prop", R.list_prop)+nl
    if R.list_poly is not None and len(R.list_poly) > 0:
        for P in R.list_poly:
            output += idt*idt_level+tagpolytope("reg_item", P)+nl
    output += idt*(idt_level-1)+'</region>'+nl
    return output

def tagmatrix(name, A):
    """Create tag of type "matrix", with given name.

    Given matrix, A, should be an ndarray.  If it is a vector, rather
    than a matrix (i.e. if len(A.shape) = 1), then it is regarded as a
    column vector, i.e., we set r="m" c="1", where A has m
    elements.

    <name type="matrix" r="n" c="m">...</name>

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if not isinstance(name, str):
        raise TypeError("tag name must be a string.")

    # Handle empty matrix case.
    if A is None or A == []:
        A = np.array([])

    if len(A.shape) == 1 and A.shape[0] == 0:  # Empty matrix?
        output = '<'+name+' type="matrix" r="0" c="0"></'+name+'>'
    elif len(A.shape) == 1:  # Column vector?
        output = '<'+name+' type="matrix" r="'+str(A.shape[0]) \
            +'" c="1">'
        for i in range(A.shape[0]-1):
            output += str(A[i]) + ' '
        output += str(A[-1]) + '</'+name+'>'

    else:  # Otherwise, treat as matrix
        output = '<'+name+' type="matrix" r="'+str(A.shape[0]) \
            +'" c="'+str(A.shape[1])+'">'
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == A.shape[0]-1 and j == A.shape[1]-1:
                    break  # ...since last element is not followed by a comma.
                output += str(A[i][j]) + ' '
        output += str(A[-1][-1]) + '</'+name+'>'

    return output


def yaml_polytope(x):
    """Given dictionary from YAML data file, return polytope."""
    if x.has_key("V"):
        tmp_V = np.loadtxt(StringIO(x["V"]))
        return pc.qhull(tmp_V)
    else:
        tmp_H = np.loadtxt(StringIO(x["H"]))
        tmp_K = np.loadtxt(StringIO(x["K"]))
        if len(tmp_K.shape) == 1:
            tmp_K = np.reshape(tmp_K, (tmp_K.shape[0], 1))
        return pc.Polytope(tmp_H, tmp_K)

def readYAMLfile(fname, verbose=0):
    """Wrap loadYAML function."""
    with open(fname, "r") as f:
        x = f.read()
    return loadYAML(x, verbose=verbose)

def loadYAML(x, verbose=0):
    """Read transition system specified using YAML in given string.

    Return (sys_dyn, initial_partition, N), where:

    - sys_dyn is the system dynamics (instance of CtsSysDyn),
    - cont_prop is the continuous propositions
      of the space (instance of PropPreservingPartition), and
    - N is the horizon length (default is 10).

    To easily read and process this string from a file, instead call
    the method *readYAMLfile*.

    Raise exception if critical error.
    """
    if yaml is None:
        raise ImportError("PyYAML package not found.\nTo read/write YAML, you will need to install PyYAML; see http://pyyaml.org/")

    dumped_data = yaml.load(x)

    # Sanity check
    if (("A" not in dumped_data.keys())
        or ("B" not in dumped_data.keys())
        or ("U" not in dumped_data.keys())
        or ("H" not in dumped_data["U"].keys())
        or ("K" not in dumped_data["U"].keys())
        or ("cont_prop" not in dumped_data.keys())
        or ("X" not in dumped_data.keys())
        or ("assumption" not in dumped_data.keys())
        or ("guarantee" not in dumped_data.keys())):
        raise ValueError("Missing required data.")
    
    # Parse dynamics related strings
    A = np.loadtxt(StringIO(dumped_data["A"]))
    B = np.loadtxt(StringIO(dumped_data["B"]))
    U = yaml_polytope(dumped_data["U"])
    X = yaml_polytope(dumped_data["X"])
    if dumped_data.has_key("E"):
        E = np.loadtxt(StringIO(dumped_data["E"]))
    else:
        E = []  # No disturbances indicated by empty list.
    if dumped_data.has_key("W"):
        W = yaml_polytope(dumped_data["W"])
    else:
        W = []  # No disturbances
    if dumped_data.has_key("horizon"):
        N = dumped_data["horizon"]
    else:
        N = 10  # Default to 10, as in function discretize.discretize
        
    env_vars = dict()
    if dumped_data.has_key("env_vars"):
        env_vars = dumped_data["env_vars"]
        
    sys_disc_vars = dict()
    if dumped_data.has_key("sys_disc_vars"):
        sys_disc_vars = dumped_data["sys_disc_vars"]

    # Parse initial partition
    cont_prop = dict()
    for (k, v) in dumped_data["cont_prop"].items():
        cont_prop[k] = yaml_polytope(v)
        
    # Spec
    assumption = dumped_data["assumption"]
    guarantee = dumped_data["guarantee"]

    if verbose > 0:
        print "A =\n", A
        print "B =\n", B
        print "E =\n", E
        print "X =", X
        print "U =", U
        print "W =", W
        print "horizon (N) =", N
        for (k, v) in cont_prop.items():
            print k+" =\n", v

    # Build transition system
    sys_dyn = LtiSysDyn(A, B, E, [], U, W)
    initial_partition = prop2part(X, cont_prop)

    return (sys_dyn, initial_partition, N, assumption, guarantee, env_vars, sys_disc_vars)
