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
Several functions that help in writing tulipcon XML files.

Note that many of these functions are just convenience wrappers around
NumPy and Python's pickling routines, if the use_pickling flag is set.
However pickling is largely untested!  For now you should go with
use_pickling=False (the default, so just call these functions without
any pickling flags).
"""

import pickle
import re
import xml.etree.ElementTree as ET
import numpy as np

import rhtlp
import discretize
import automaton
import prop2part
import polytope as pc
import jtlvint
import errorprint as ep
from spec import GRSpec


#################
# Problem Types #
#################
SYNTH_PROB = 0
RHTLP_PROB = 1


def readXMLfile(fname, verbose=0, use_pickling=False):
    """Read tulipcon XML string directly from a file.

    Returns 3-tuple, as described in *loadXML* docstring.  (This is a
    convenience method, mostly only wrapping *loadXML*.)
    """
    with open(fname, "r") as f:
        x = f.read()
    return loadXML(x, verbose=verbose, use_pickling=use_pickling)

def loadXML(x, verbose=0, use_pickling=False):
    """Return 3-tuple of, in terms of classes, (SynthesisProb, CtsSysDyn, Automaton).

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
        
    if elem.tag != "tulipcon":
        raise TypeError("root tag should be tulipcon.")
    if ("version" not in elem.attrib.keys()):
        raise ValueError("unversioned tulipcon XML string.")
    if int(elem.attrib["version"]) != 0:
        raise ValueError("unsupported tulipcon XML version: "+str(elem.attrib["version"]))

    ptype_tag = elem.find("prob_type")
    if ptype_tag is None:
        ep.printWarning('tulipcon XML string is missing <prob_type> tag;\nassuming SynthesisProb')
        ptype = SYNTH_PROB
    elif ptype_tag.text == "synth":
        ptype = SYNTH_PROB
    elif ptype_tag.text == "rhtlp":
        ptype = RHTLP_PROB
    else:
        raise ValueError("Unrecognized prob_type: "+str(ptype_tag.text))
    
    # Build CtsSysDyn, or set to None
    c_dyn = elem.find("c_dyn")
    if c_dyn is None:
        sys_dyn = None
    else:
        (tag_name, A) = untagmatrix(c_dyn.find("A"))
        (tag_name, B) = untagmatrix(c_dyn.find("B"))
        (tag_name, E) = untagmatrix(c_dyn.find("E"))
        (tag_name, Uset) = untagpolytope(c_dyn.find("U_set"))
        (tag_name, Wset) = untagpolytope(c_dyn.find("W_set"))
        sys_dyn = discretize.CtsSysDyn(A, B, E, [], Uset, Wset)

    # Extract LTL specification
    s_elem = elem.find("spec")
    if s_elem.find("env_init") is not None:  # instance of GRSpec style
        spec = GRSpec()
        for spec_tag in ["env_init", "env_safety", "env_prog",
                         "sys_init", "sys_safety", "sys_prog"]:
            if (s_elem.find(spec_tag) is not None) \
                    and (s_elem.find(spec_tag).text is not None):
                setattr(spec, spec_tag, s_elem.find(spec_tag).text)
                setattr(spec, spec_tag,
                        getattr(spec, spec_tag).replace("&lt;", "<"))
                setattr(spec, spec_tag,
                        getattr(spec, spec_tag).replace("&gt;", ">"))
                setattr(spec, spec_tag,
                        getattr(spec, spec_tag).replace("&amp;", "&"))
    else:  # assume, guarantee strings style
        spec = ["", ""]
        if s_elem.find("assume") is not None:
            spec[0] = s_elem.find("assume").text
        if s_elem.find("guarantee") is not None:
            spec[1] = s_elem.find("guarantee").text
        for k in [0, 1]:  # Undo special character encoding
            spec[k] = spec[k].replace("&lt;", "<")
            spec[k] = spec[k].replace("&gt;", ">")
            spec[k] = spec[k].replace("&amp;", "&")

    # ``Continuous propositions'', if available
    cp_tag = elem.find("cont_props")
    if cp_tag is not None:
        cont_props = dict()
        for sym_tag in cp_tag.findall("item"):
            if "key" not in sym_tag.attrib.keys():
                ep.printWarning("mal-formed <cont_props> tag in given tulipcon XML string.")
                cont_props = {}
                break  # Give-up on this <cont_props> tag
            sym_poly_tag = sym_tag.find("cont_prop_poly")
            if sym_poly_tag is None:
                cont_props[sym_tag.attrib["key"]] = None
            else:
                (tag_name, cont_props[sym_tag.attrib["key"]]) = untagpolytope(sym_poly_tag)
    else:
        cont_props = {}
    
    # Discrete dynamics, if available
    d_dyn = elem.find("d_dyn")
    if d_dyn is None:
        prob = None
    else:
        (tag_name, env_vars) = untagdict(elem.find("env_vars"))
        (tag_name, sys_disc_vars) = untagdict(elem.find("con_vars"))
        if (d_dyn.find("domain") is None) \
                and (d_dyn.find("trans") is None) \
                and (d_dyn.find("prop_symbols") is None):
            disc_dynamics = None
        else:
            (tag_name, domain) = untagpolytope(d_dyn.find("domain"))
            (tag_name, trans) = untagmatrix(d_dyn.find("trans"), np_type=np.uint8)
            (tag_name, prop_symbols) = untaglist(d_dyn.find("prop_symbols"), cast_f=str)
            region_elem = d_dyn.find("regions")
            list_region = []
            if region_elem is not None:
                region_items = d_dyn.find("regions").findall("item")
                if region_items is not None and len(region_items) > 0:
                    for region_item in region_items:
                        (tag_name, R) = untagregion(region_item, cast_f_list=int,
                                                    np_type_P=np.float64)
                        list_region.append(R)

            disc_dynamics = prop2part.PropPreservingPartition(domain=domain,
                                                              num_prop=len(prop_symbols),
                                                              list_region=list_region,
                                                              num_regions=len(list_region),
                                                              adj=0,
                                                              trans=trans,
                                                              list_prop_symbol=prop_symbols)

        # Build appropriate ``problem'' instance
        if ptype == SYNTH_PROB:
            prob = jtlvint.generateJTLVInput(env_vars=env_vars,
                                             sys_disc_vars=sys_disc_vars,
                                             spec=spec,
                                             disc_props={},
                                             disc_dynamics=disc_dynamics,
                                             smv_file="", spc_file="", verbose=2)
        else:  # ptype == RHTLP_PROB
            if disc_dynamics is not None:
                prob = rhtlp.RHTLPProb(shprobs=[], Phi="True", discretize=False,
                                       env_vars=env_vars,
                                       sys_disc_vars=sys_disc_vars,
                                       disc_dynamics=disc_dynamics,
                                       cont_props=cont_props,
                                       spec=spec)
            else:
                prob = rhtlp.RHTLPProb(shprobs=[], Phi="True", discretize=False,
                                       env_vars=env_vars,
                                       sys_disc_vars=sys_disc_vars,
                                       cont_props=cont_props,
                                       spec=spec)

    # Build Automaton
    aut_elem = elem.find("aut")
    if aut_elem is None \
            or ((aut_elem.text is None) and len(aut_elem.getchildren()) == 0):
        aut = None
    else:
        aut = automaton.Automaton()
        if not aut.loadXML(aut_elem):
            ep.printError("failed to read Automaton from given tulipcon XML string.")
            aut = None

    return (prob, sys_dyn, aut)
                               
def dumpXML(prob, spec=['',''], sys_dyn=None, aut=None,
            synthesize_aut=False, verbose=0, pretty=False, use_pickling=False):
    """Return tulipcon XML string.

    prob is an instance of SynthesisProb or a child class, such as
    RHTLPProb (both defined in rhtlp.py).  sys_dyn is an instance of
    CtsSysDyn (defined in discretize.py), or None. If None, then
    continuous dynamics are considered empty (i.e. there are none);
    ** NOT IMPLEMENTED YET (where sys_dyn = None) **

    If pretty is True, then use indentation and newlines to make the
    resulting XML string more visually appealing.
    
    aut is an instance of Automaton.  If None (rather than an
    instance), then an empty <aut> tag is written.

    spec may be an instance of GRSpec or a list.  If of GRSPec, then
    it is formed as expected.  If spec is a list, then first element
    of ``assume'' string, and second element of ``guarantee'' string.
    spec=None is also accepted, in which case the specification is
    considered empty, but note that this could cause problems later
    unless some content is introduced.

    ** synthesize_aut flag is IGNORED **
    If synthesize_aut is True, then if prob.__realizable is not None,
    use its value to determine whether a previously computed *.aut
    file should be read.  Else (if prob.__realizable is None or
    False), then try to compute automaton (saving to file
    prob.__jtlvfile + '.aut', as usual). If synthesize_aut is False,
    then just save an empty <aut></aut>.

    ** verbose flag is IGNORED (but that will change...) **

    To easily save the result here to a file, instead call the method
    *writeXMLfile*
    """
    if not isinstance(prob, (rhtlp.SynthesisProb, rhtlp.RHTLPProb)):
        raise TypeError("prob should be an instance (or child) of rhtlp.SynthesisProb.")
    if sys_dyn is not None and not isinstance(sys_dyn, discretize.CtsSysDyn):
        raise TypeError("sys_dyn should be an instance of discretizeM.CtsSysDyn or None.")
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

    output = '<tulipcon version="0">'+nl
    idt_level += 1
    output += idt*idt_level+'<prob_type>'
    if isinstance(prob, rhtlp.RHTLPProb):  # Beware of order and inheritance
        output += 'rhtlp'
    elif isinstance(prob, rhtlp.SynthesisProb):
        output += 'synth'
    output += '</prob_type>'+nl

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

    output += tagdict("env_vars", prob.getEnvVars(), pretty=pretty, idt_level=idt_level)
    output += tagdict("con_vars", prob.getSysVars(), pretty=pretty, idt_level=idt_level)

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
            spec_dict[k] = spec_dict[k].replace("<", "&lt;")
            spec_dict[k] = spec_dict[k].replace(">", "&gt;")
            spec_dict[k] = spec_dict[k].replace("&", "&amp;")
        
        # Finally, dump XML tags
        output += idt*idt_level+'<spec>'+nl
        for (k, v) in spec_dict.items():
            output += idt*(idt_level+1)+ '<'+k+'>' + v + '</'+k+'>'+nl
        output += idt*idt_level+'</spec>'+nl

    else:
        for k in [0,1]:
            spec[k] = spec[k].replace("<", "&lt;")
            spec[k] = spec[k].replace(">", "&gt;")
            spec[k] = spec[k].replace("&", "&amp;")
        output += idt*idt_level+'<spec><assume>'+spec[0]+'</assume>'+nl
        output += idt*(idt_level+1)+'<guarantee>'+spec[1]+'</guarantee></spec>'+nl

    # Perhaps there is a cleaner way to do this, rather than by
    # checking for method getContProps?
    if hasattr(prob, "getContProps"):
        output += idt*idt_level+'<cont_props>'+nl
        idt_level += 1
        for (prop_sym, prop_poly) in prob.getContProps().items():
            output += idt*idt_level+'<item key="'+prop_sym+'">'+nl
            output += idt*(idt_level+1)+tagpolytope("cont_prop_poly", prop_poly)+nl
            output += idt*idt_level+'</item>'+nl
        idt_level -= 1
        output += idt*idt_level+'</cont_props>'+nl

    disc_dynamics = prob.getDiscretizedDynamics()
    if disc_dynamics is None:
        output += idt*idt_level+'<d_dyn></d_dyn>'+nl
    else:
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
                output += tagregion("item", R, pretty=pretty, idt_level=idt_level)
        idt_level -= 1
        output += idt*idt_level+'</regions>'+nl
        idt_level -= 1
        output += idt*idt_level+'</d_dyn>'+nl

    if aut is None:
        output += idt*idt_level+'<aut></aut>'+nl
    else:
        output += aut.dumpXML(pretty=pretty, idt_level=idt_level)+nl

    output += idt_level*idt+'<smv_file></smv_file><spec_file></spec_file>'+nl
    output += idt_level*idt+'<extra></extra>'+nl

    idt_level -= 1
    assert idt_level == 0
    output += '</tulipcon>'+nl
    return output

def writeXMLfile(fname, prob, spec, sys_dyn=None, aut=None,
                 synthesize_aut=False, verbose=0, pretty=False,
                 use_pickling=False):
    """Write tulipcon XML string directly to a file.

    Returns nothing.  (This is a convenience method, mostly only
    wrapping *dumpXML*.)
    """
    with open(fname, "w") as f:
        f.write(dumpXML(prob=prob, spec=spec, sys_dyn=sys_dyn, aut=aut,
                        synthesize_aut=synthesize_aut, verbose=verbose,
                        pretty=pretty, use_pickling=use_pickling))
    return

def untaglist(x, cast_f=float, use_pickling=False):
    """Extract list from given tulipcon XML tag (string).

    Use function cast_f for type-casting extracting element strings.
    The default is float, but another common case is cast_f=int (for
    ``integer'').  If cast_f is set to None, then items are left as
    extracted, i.e. as strings.  Note that pickling restores the type
    automatically, so cast_f is ignored if use_pickling is true.

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

    # Extract list
    if use_pickling:
        li = pickle.loads(elem.text)
    elif elem.text is None:
        li = []
    else:
        if cast_f is None:
            cast_f = str
        li = [cast_f(k) for k in elem.text.split(",")]
    return (elem.tag, li)

def untagdict(x, cast_f_keys=None, cast_f_values=None,
              use_pickling=False):
    """Extract list from given tulipcon XML tag (string).

    Use functions cast_f_keys and cast_f_values for type-casting
    extracting key and value strings, respectively, or None.  The
    default is None, which means the extracted keys (resp., values)
    are left untouched (as strings), but another common case is
    cast_f_values=int (for ``integer'') or cast_f_values=float (for
    ``floating-point numbers''), while leaving cast_f_keys=None to
    indicate dictionary keys are strings.  Note that pickling restores
    the type automatically, so cast_f_keys and cast_f_values are
    ignored if use_pickling is true.

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

    # Extract dictionary
    if use_pickling:
        di = pickle.loads(elem.text)
    else:
        items_li = elem.findall('item')
        if cast_f_keys is None:
            cast_f_keys = str
        if cast_f_values is None:
            cast_f_values = str
        di = dict()
        for item in items_li:
            # N.B., we will overwrite duplicate keys without warning!
            di[cast_f_keys(item.attrib['key'])] = cast_f_values(item.attrib['value'])
    return (elem.tag, di)

def untagmatrix(x, np_type=np.float64, use_pickling=False):
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
    elif use_pickling:
        x_mat = np.loads(elem.text)
    else:
        num_rows = int(elem.attrib['r'])
        num_cols = int(elem.attrib['c'])
        x_mat = np.array([k for k in elem.text.split(",")], dtype=np_type)
        x_mat = x_mat.reshape(num_rows, num_cols)
    return (elem.tag, x_mat)

def untagpolytope(x, np_type=np.float64, use_pickling=False):
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

    h_tag = elem.find('H')
    k_tag = elem.find('K')

    (H_out_name, H_out) = untagmatrix(h_tag, np_type=np_type,
                                      use_pickling=use_pickling)
    (K_out_name, K_out) = untagmatrix(k_tag, np_type=np_type,
                                      use_pickling=use_pickling)

    return (elem.tag, pc.Polytope(H_out, K_out))

def untagregion(x, cast_f_list=str, np_type_P=np.float64, use_pickling=False):
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

    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x
    if elem.attrib['type'] != "region":
        raise ValueError("tag should indicate type of ``region''.")

    (tag_name, list_prop) = untaglist(elem.find("list_prop"), cast_f=cast_f_list,
                                      use_pickling=use_pickling)
    if list_prop is None:
        list_prop = []

    poly_tags = elem.findall("reg_item")
    list_poly = []
    if poly_tags is not None and len(poly_tags) > 0:
        for P_elem in poly_tags:
            (tag_name, P) = untagpolytope(P_elem, np_type=np_type_P,
                                          use_pickling=use_pickling)
            list_poly.append(P)

    return (elem.tag, pc.Region(list_poly=list_poly, list_prop=list_prop))

def tagdict(name, di, pretty=False, use_pickling=False, idt_level=0):
    """Create tag that basically stores a dictionary object.

    If pretty is True, then use indentation and newlines to make the
    resulting XML string more visually appealing.  idt_level is the
    base indentation level on which to create automaton string.  This
    level is only relevant if pretty=True.

    N.B., if you are *not* using pickling, then all keys and values
    are treated as strings (and frequently wrapped in str() to force
    this behavior).

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if not isinstance(name, str):
        raise TypeError("tag name must be a string.")
    if use_pickling:
        return '<'+name+'>' + pickle.dumps(di) + '</'+name+'>'

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

def tagpolytope(name, P, use_pickling=False):
    """Create tag of type ``Polytope'', with given name.

    Polytope is as defined in tulip.polytope_computations module.

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if not isinstance(name, str):
        raise TypeError("tag name must be a string.")
    # Handle nil polytope case
    if P is None or P == []:
        P = pc.Polytope(np.array([]), np.array([]))
    output = '<'+name+' type="polytope">'
    output += tagmatrix("H", P.A, use_pickling=False)
    output += tagmatrix("K", P.b, use_pickling=False)
    output += '</'+name+'>'
    return output

def taglist(name, li, use_pickling=False):
    """Create tag that basically stores a list object.

    N.B., if you are *not* using pickling, then all list elements are
    treated as strings (and frequently wrapped in str() to force this
    behavior).

    Return the resulting string.  On failure, raises an appropriate
    exception, or returns False.
    """
    if not isinstance(name, str):
        raise TypeError("tag name must be a string.")
    if use_pickling:
        return '<'+name+'>' + pickle.dumps(li) + '</'+name+'>'

    output = '<'+name+'>'
    if li is not None:
        for i in range(len(li)-1):
            output += str(li[i]) + ', '
        output += str(li[-1])
    output += '</'+name+'>'
    return output

def tagregion(name, R, pretty=False, use_pickling=False, idt_level=0):
    """Create tag of type ``Region'', with given name.

    Region is as defined in tulip.polytope_computations module.

    If pretty is True, then use indentation and newlines to make the
    resulting XML string more visually appealing.  idt_level is the
    base indentation level on which to create automaton string.  This
    level is only relevant if pretty=True.

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

    # Handle nil Region case
    if R is None or R == []:
        R = pc.Region(list_poly=[], list_prop=[])
    output = idt*idt_level+'<'+name+' type="region">'+nl
    idt_level += 1
    output += idt*idt_level+taglist("list_prop", R.list_prop, use_pickling=use_pickling)+nl
    if R.list_poly is not None and len(R.list_poly) > 0:
        for P in R.list_poly:
            output += idt*idt_level+tagpolytope("reg_item", P)+nl
    output += idt*(idt_level-1)+'</'+name+'>'+nl
    return output

def tagmatrix(name, A, use_pickling=False):
    """Create tag of type ``matrix'', with given name.

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

    if use_pickling:
        return '<'+name+' type="matrix">' + A.dumps() + '</'+name+'>'

    if len(A.shape) == 1 and A.shape[0] == 0:  # Empty matrix?
        output = '<'+name+' type="matrix" r="0" c="0"></'+name+'>'
    elif len(A.shape) == 1:  # Column vector?
        output = '<'+name+' type="matrix" r="'+str(A.shape[0]) \
            +'" c="1">'
        for i in range(A.shape[0]-1):
            output += str(A[i]) + ', '
        output += str(A[-1]) + '</'+name+'>'

    else:  # Otherwise, treat as matrix
        output = '<'+name+' type="matrix" r="'+str(A.shape[0]) \
            +'" c="'+str(A.shape[1])+'">'
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == A.shape[0]-1 and j == A.shape[1]-1:
                    break  # ...since last element is not followed by a comma.
                output += str(A[i][j]) + ', '
        output += str(A[-1][-1]) + '</'+name+'>'

    return output


# This test should be broken into units.
def conxml_test():
#if __name__ == "__main__":
    A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float64)
    b = np.asarray(range(3))
    assert tagmatrix("A", A, use_pickling=False) == '<A type="matrix" r="3" c="3">1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0</A>'
    assert tagmatrix("b", b, use_pickling=False) == '<b type="matrix" r="3" c="1">0, 1, 2</b>'

    P = pc.Polytope(A, b)
    assert tagpolytope("P", P, use_pickling=False) == '<P type="polytope"><H type="matrix" r="3" c="3">1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0</H><K type="matrix" r="3" c="1">0, 1, 2</K></P>'

    li = range(7)
    assert taglist("trans", li, use_pickling=False) == '<trans>0, 1, 2, 3, 4, 5, 6</trans>'

    (li_out_name, li_out) = untaglist(taglist("trans", li,
                                              use_pickling=False),
                                      use_pickling=False)
    assert (li_out_name == "trans") and (li_out == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    di = {1:2, 3:4, 'X0':'boolean'}
    assert tagdict("env_vars", di, use_pickling=False) == '<env_vars><item key="1" value="2" /><item key="X0" value="boolean" /><item key="3" value="4" /></env_vars>'

    (di_out_name, di_out) = untagdict(tagdict("con_vars", di,
                                              use_pickling=False),
                                      use_pickling=False)
    assert (di_out_name == "con_vars") and (di_out == {'1': '2', 'X0': 'boolean', '3': '4'})

    (A_out_name, A_out) = untagmatrix(tagmatrix("A", A, use_pickling=False),
                                      use_pickling=False)
    assert (A_out_name == "A") and (np.all(np.all(A_out == A)))

    (b_out_name, b_out) = untagmatrix(tagmatrix("b", b, use_pickling=False),
                                      np_type=int,
                                      use_pickling=False)
    assert (b_out_name == "b") and (np.all(np.squeeze(b_out) == b))

    (P_out_name, P_out) = untagpolytope(tagpolytope("P", P,
                                                    use_pickling=False),
                                        use_pickling=False)
    assert (P_out_name == "P") and (np.all(np.all(P_out.A == P.A))) and (np.all(np.squeeze(P_out.b) == np.squeeze(P.b)))

    list_prop  = range(10)
    list_poly = [pc.Polytope(A*(k+1), b*(k+1)) for k in range(10)]
    R = pc.Region(list_prop=list_prop, list_poly=list_poly)
    assert tagregion("R", R) == '<R type="region"><list_prop>0, 1, 2, 3, 4, 5, 6, 7, 8, 9</list_prop><reg_item type="polytope"><H type="matrix" r="3" c="3">1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0</H><K type="matrix" r="3" c="1">0, 1, 2</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0</H><K type="matrix" r="3" c="1">0, 2, 4</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0</H><K type="matrix" r="3" c="1">0, 3, 6</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0</H><K type="matrix" r="3" c="1">0, 4, 8</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0</H><K type="matrix" r="3" c="1">0, 5, 10</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0</H><K type="matrix" r="3" c="1">0, 6, 12</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0</H><K type="matrix" r="3" c="1">0, 7, 14</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0</H><K type="matrix" r="3" c="1">0, 8, 16</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0</H><K type="matrix" r="3" c="1">0, 9, 18</K></reg_item><reg_item type="polytope"><H type="matrix" r="3" c="3">10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0</H><K type="matrix" r="3" c="1">0, 10, 20</K></reg_item></R>'

    (R_out_name, R_out) = untagregion(tagregion("R", R), cast_f_list=int)
    assert R_out.list_prop == R.list_prop
    for (ind, P) in enumerate(R_out.list_poly):
        R.list_poly[ind].b = np.asarray(R.list_poly[ind].b, dtype=np.float64)
        assert str(R.list_poly[ind]) == str(P)

    # Check fringe cases
    assert tagmatrix("omfg", []) == '<omfg type="matrix" r="0" c="0"></omfg>'
    assert tagpolytope("rtfm", None) == '<rtfm type="polytope"><H type="matrix" r="0" c="0"></H><K type="matrix" r="0" c="0"></K></rtfm>'
    assert untaglist(taglist("good_times", None)) == ("good_times", [])
