# Copyright (c) 2011-2013 by California Institute of Technology
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
Interface to gr1c

In general, functions defined here will raise CalledProcessError (from
the subprocess module) or OSError if an exception occurs while
interacting with the gr1c executable.

Most functions have a "verbose" argument.  0 means silent (the default
setting), positive means provide some status updates.
"""

import copy
import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
import networkx as nx

from spec import GRSpec
from transys import MealyMachine

GR1C_BIN_PREFIX=""


DEFAULT_NAMESPACE = "http://tulip-control.sourceforge.net/ns/1"

def _untaglist(x, cast_f=float,
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

def _untagdict(x, cast_f_keys=None, cast_f_values=None,
               namespace=DEFAULT_NAMESPACE, get_order=False):
    """Extract dictionary from given tulipcon XML tag (string).

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
    and the dictionary obtained from it.  If get_order is True, then
    return a triple, where the first two elements are as usual and the
    third is the list of keys in the order they were found.
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
    items_li = elem.findall(ns_prefix+"item")
    if cast_f_keys is None:
        cast_f_keys = str
    if cast_f_values is None:
        cast_f_values = str
    di = dict()
    if get_order:
        key_list = []
    for item in items_li:
        # N.B., we will overwrite duplicate keys without warning!
        di[cast_f_keys(item.attrib["key"])] = cast_f_values(item.attrib["value"])
        if get_order:
            key_list.append(item.attrib["key"])
    if get_order:
        return (elem.tag, di, key_list)
    else:
        return (elem.tag, di)


def load_aut_xml(x, namespace=DEFAULT_NAMESPACE):
    """Return GRSpec and MealyMachine constructed from output of gr1c.

    x can be a string or an instance of
    xml.etree.ElementTree._ElementInterface
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
    if int(elem.attrib["version"]) != 1:
        raise ValueError("unsupported tulipcon XML version: "+str(elem.attrib["version"]))

    # Extract discrete variables and LTL specification
    (tag_name, env_vardict, env_vars) = _untagdict(elem.find(ns_prefix+"env_vars"), get_order=True)
    (tag_name, sys_vardict, sys_vars) = _untagdict(elem.find(ns_prefix+"sys_vars"), get_order=True)
    env_domains = []
    for v in env_vars:
        dom = env_vardict[v]
        if dom[0] == "[":
            end_ind = dom.find("]")
            if end_ind < 0:
                raise ValueError("invalid domain for variable \""+str(v)+"\": "+str(dom))
            dom_parts = dom[1:end_ind].split(",")
            if len(dom_parts) != 2:
                raise ValueError("invalid domain for variable \""+str(v)+"\": "+str(dom))
            env_domains.append((int(dom_parts[0]), int(dom_parts[1])))
        elif dom == "boolean":
            env_domains.append("boolean")
        else:
            raise ValueError("unrecognized type of domain for variable \""+str(v)+"\": "+str(dom))
    sys_domains = []
    for v in sys_vars:
        dom = sys_vardict[v]
        if dom[0] == "[":
            end_ind = dom.find("]")
            if end_ind < 0:
                raise ValueError("invalid domain for variable \""+str(v)+"\": "+str(dom))
            dom_parts = dom[1:end_ind].split(",")
            if len(dom_parts) != 2:
                raise ValueError("invalid domain for variable \""+str(v)+"\": "+str(dom))
            sys_domains.append((int(dom_parts[0]), int(dom_parts[1])))
        elif dom == "boolean":
            sys_domains.append("boolean")
        else:
            raise ValueError("unrecognized type of domain for variable \""+str(v)+"\": "+str(dom))
    s_elem = elem.find(ns_prefix+"spec")
    spec = GRSpec(env_vars=env_vars, sys_vars=sys_vars)
    for spec_tag in ["env_init", "env_safety", "env_prog",
                     "sys_init", "sys_safety", "sys_prog"]:
        if s_elem.find(ns_prefix+spec_tag) is None:
            raise ValueError("invalid specification in tulipcon XML string.")
        (tag_name, li) = _untaglist(s_elem.find(ns_prefix+spec_tag),
                                    cast_f=str, namespace=namespace)
        li = [v.replace("&lt;", "<") for v in li]
        li = [v.replace("&gt;", ">") for v in li]
        li = [v.replace("&amp;", "&") for v in li]
        setattr(spec, spec_tag, li)

    aut_elem = elem.find(ns_prefix+"aut")
    if aut_elem is None \
            or ((aut_elem.text is None) and len(aut_elem.getchildren()) == 0):
        aut = None
    else:
        # Assume version 1 of tulipcon XML
        if aut_elem.attrib["type"] != "basic":
            raise ValueError("Automaton class only recognizes type \"basic\".")
        node_list = aut_elem.findall(ns_prefix+"node")
        id_list = []  # For more convenient searching, and to catch redundancy
        A = nx.DiGraph()
        for node in node_list:
            this_id = int(node.find(ns_prefix+"id").text)
            this_name = node.find(ns_prefix+"anno").text  # Assume version 1
            (tag_name, this_name_list) = _untaglist(node.find(ns_prefix+"anno"),
                                                    cast_f=int)
            if len(this_name_list) == 2:
                (mode, rgrad) = this_name_list
            else:
                (mode, rgrad) = (-1, -1)
            (tag_name, this_child_list) = _untaglist(node.find(ns_prefix+"child_list"),
                                                     cast_f=int)
            if tag_name != ns_prefix+"child_list":
                # This really should never happen and may not even be
                # worth checking.
                raise ValueError("failure of consistency check while processing aut XML string.")
            (tag_name, this_state) = _untagdict(node.find(ns_prefix+"state"),
                                                cast_f_values=int,
                                                namespace=namespace)

            if tag_name != ns_prefix+"state":
                raise ValueError("failure of consistency check while processing aut XML string.")
            if this_id in id_list:
                printWarning("duplicate nodes found: "+str(this_id)+"; ignoring...")
                continue
            id_list.append(this_id)
            A.add_node(this_id, state=copy.copy(this_state),
                       mode=mode, rgrad=rgrad)
            for next_node in this_child_list:
                A.add_edge(this_id, next_node)

        mach = MealyMachine()
        mach.add_inputs([(evar, {0,1}) for evar in env_vars])
        mach.add_outputs([(svar, {0,1}) for svar in sys_vars])
        mach.states.add_from(A.nodes())
        for u in A.nodes_iter():
            for v in A.successors_iter(u):
                mach.transitions.add_labeled(u, v, A.node[v]["state"])

    return (spec, mach)


def check_syntax(spec_str, verbose=0):
    """Check whether given string has correct gr1c specification syntax.

    Return True if syntax check passed, False on error.
    """
    f = tempfile.TemporaryFile()
    f.write(spec_str)
    f.seek(0)
    p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c", "-s"],
                         stdin=f,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode == 0:
        return True
    else:
        if verbose > 0:
            print p.stdout.read()
        return False


def check_realizable(spec, verbose=0):
    """Decide realizability of specification defined by given GRSpec object.

    Return True if realizable, False if not, or an error occurs.
    """
    f = tempfile.TemporaryFile()
    f.write(spec.to_gr1c())
    f.seek(0)
    p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c", "-r"],
                         stdin=f,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode == 0:
        return True
    else:
        if verbose > 0:
            print p.stdout.read()
        return False


def synthesize(spec, verbose=0):
    """Synthesize strategy.

    Return strategy as instance of Automaton class, or None if
    unrealizable or error occurs.
    """
    p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c", "-t", "tulip"],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdoutdata, stderrdata) = p.communicate(spec.to_gr1c())
    if p.returncode == 0:
        (spec, aut) = load_aut_xml(stdoutdata)
        return aut
    else:
        if verbose > 0:
            print stdoutdata
        return None


class GR1CSession:
    """Manage interactive session with gr1c.

    Given lists of environment and system variable names determine the
    order of values in state vectors for communication with the gr1c
    process.  Eventually there may be code to infer this directly from
    the spec file.

    **gr1c is assumed not to use GNU Readline.**

    Please compile it that way if you are using this class.
    (Otherwise, GNU Readline will echo commands and make interaction
    with gr1c more difficult.)

    The argument `prompt` is the string printed by gr1c to indicate it
    is ready for the next command.  The default value is a good guess.

    Unless otherwise indicated, command methods return True on
    success, False if error.
    """
    def __init__(self, spec_filename, sys_vars, env_vars=[], prompt=">>> "):
        self.spec_filename = spec_filename
        self.sys_vars = sys_vars[:]
        self.env_vars = env_vars[:]
        self.prompt = prompt
        if self.spec_filename is not None:
            self.p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c",
                                       "-i", self.spec_filename],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        else:
            self.p = None


    def iswinning(self, state):
        """Return True if given state is in winning set, False otherwise.

        state should be a dictionary with keys of variable names
        (strings) and values of the value taken by that variable in
        this state, e.g., as in nodes of the Automaton class.
        """
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write("winning "+" ".join([str(i) for i in state_vector])+"\n")
        if "True\n" in self.p.stdout.readline():
            return True
        else:
            return False


    def getindex(self, state, goal_mode):
        if goal_mode < 0 or goal_mode > self.numgoals()-1:
            raise ValueError("Invalid goal mode requested: "+str(goal_mode))
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write("getindex "+" ".join([str(i) for i in state_vector])+" "+str(goal_mode)+"\n")
        line = self.p.stdout.readline()
        if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
        return int(line[:-1])

    def env_next(self, state):
        """Return list of possible next environment moves, given current state.

        Format of given state is same as for iswinning method.
        """
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write("envnext "+" ".join([str(i) for i in state_vector])+"\n")
        env_moves = []
        line = self.p.stdout.readline()
        while "---\n" not in line:
            if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
            env_moves.append(dict([(k, int(s)) for (k,s) in zip(self.env_vars, line.split())]))
            line = self.p.stdout.readline()
        return env_moves


    def sys_nextfeas(self, state, env_move, goal_mode):
        """Return list of next system moves consistent with some strategy.

        Format of given state and env_move is same as for iswinning
        method.
        """
        if goal_mode < 0 or goal_mode > self.numgoals()-1:
            raise ValueError("Invalid goal mode requested: "+str(goal_mode))
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        emove_vector = range(len(env_move))
        for ind in range(len(self.env_vars)):
            emove_vector[ind] = env_move[self.env_vars[ind]]
        self.p.stdin.write("sysnext "+" ".join([str(i) for i in state_vector])+" "+" ".join([str(i) for i in emove_vector])+" "+str(goal_mode)+"\n")
        sys_moves = []
        line = self.p.stdout.readline()
        while "---\n" not in line:
            if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
            sys_moves.append(dict([(k, int(s)) for (k,s) in zip(self.sys_vars, line.split())]))
            line = self.p.stdout.readline()
        return sys_moves


    def sys_nexta(self, state, env_move):
        """Return list of possible next system moves, whether or not winning.

        Format of given state and env_move is same as for iswinning
        method.
        """
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        emove_vector = range(len(env_move))
        for ind in range(len(self.env_vars)):
            emove_vector[ind] = env_move[self.env_vars[ind]]
        self.p.stdin.write("sysnexta "+" ".join([str(i) for i in state_vector])+" "+" ".join([str(i) for i in emove_vector])+"\n")
        sys_moves = []
        line = self.p.stdout.readline()
        while "---\n" not in line:
            if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
            sys_moves.append(dict([(k, int(s)) for (k,s) in zip(self.sys_vars, line.split())]))
            line = self.p.stdout.readline()
        return sys_moves


    def getvars(self):
        """Return string of environment and system variable names in order.

        Indices are indicated in parens.
        """
        self.p.stdin.write("var\n")
        line = self.p.stdout.readline()
        if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
        return line[:-1]

    def numgoals(self):
        self.p.stdin.write("numgoals\n")
        line = self.p.stdout.readline()
        if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
        return int(line[:-1])

    def reset(self, spec_filename=None):
        """Quit and start anew, reading spec from file with given name.

        If no filename given, then use previous one.
        """
        if self.p is not None:
            self.p.stdin.write("quit\n")
            returncode = self.p.wait()
            self.p = None
            if returncode != 0:
                self.spec_filename = None
                return False
        if spec_filename is not None:
            self.spec_filename = spec_filename
        if self.spec_filename is not None:
            self.p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c",
                                       "-i", self.spec_filename],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        else:
            self.p = None
        return True

    def close(self):
        """End session, and kill gr1c child process."""
        self.p.stdin.write("quit\n")
        returncode = self.p.wait()
        self.p = None
        if returncode != 0:
            return False
        else:
            return True
