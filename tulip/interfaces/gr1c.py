# Copyright (c) 2011-2015 by California Institute of Technology
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
"""Interface to `gr1c`.

- <http://scottman.net/2012/gr1c>
- release documentation at
  <https://tulip-control.github.io/gr1c/>

In general, functions defined here
will raise `CalledProcessError` (from
the `subprocess` module) or `OSError`
if an exception occurs while
interacting with the `gr1c` executable.

Use the `logging` module to throttle verbosity.
"""
import copy
import errno
import json
import logging
import os
from pkg_resources import parse_version
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import networkx as nx
from tulip.spec import GRSpec, translate


GR1C_MIN_VERSION = '0.9.0'
GR1C_BIN_PREFIX = ''
DEFAULT_NAMESPACE = 'http://tulip-control.sourceforge.net/ns/1'
_hl = 60 * '-'
logger = logging.getLogger(__name__)


def check_gr1c():
    """Return `True` if `gr1c >= require_version` found in `PATH`."""
    try:
        v = subprocess.check_output(
            ["gr1c", "-V"],
            universal_newlines=True)
    except OSError:
        return False
    v = v.split()[1]
    if parse_version(v) >= parse_version(GR1C_MIN_VERSION):
        return True
    return False


def _assert_gr1c():
    """Raise `Exception` if `gr1c` not in PATH."""
    if check_gr1c():
        return
    raise Exception(
        f'`gr1c >= {GR1C_MIN_VERSION}` '
        'not found in the PATH.\n'
        'Unless an alternative synthesis tool is installed,\n'
        'it will not be possible to realize GR(1) specifications.\n'
        'Consult installation instructions for gr1c at:\n'
        '\t http://scottman.net/2012/gr1c\n'
        "or the TuLiP User's Guide about alternatives.")


def get_version():
    """Get version of `gr1c` as detected by TuLiP.

    Failure to find the `gr1c` program or errors in parsing the received
    version string will cause an exception.

    @return:
        `(major, minor, micro)`, a `tuple` of `int`
    """
    try:
        v_str = subprocess.check_output(
            ["gr1c", "-V"],
            universal_newlines=True)
    except OSError:
        raise OSError('gr1c not found')
    v_str = v_str.split()[1]
    try:
        major, minor, micro = v_str.split(".")
        major = int(major)
        minor = int(minor)
        micro = int(micro)
    except ValueError:
        raise ValueError(
            '`gr1c` version string '
            f'is not recognized: {v_str}')
    return (major, minor, micro)


def _untaglist(x, cast_f=float,
               namespace=DEFAULT_NAMESPACE):
    """Extract list from given tulipcon XML tag (string).

    Use function `cast_f` for type-casting extracting element strings.
    The default is `float`, but another common case is `cast_f=int` (for
    "integer").  If `cast_f` is set to `None`, then items are left as
    extracted, i.e. as strings.

    The argument x can also have the type of the return value of
    `xml.etree.ElementTree.fromstring()`. This is mainly for internal
    use, e.g. by the function `untagpolytope` and some load/dumpXML
    methods elsewhere.

    Return result as 2-tuple, containing name of the tag (as a string)
    and the list obtained from it.
    """
    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x
    if namespace is None or len(namespace) == 0:
        ns_prefix = ''
    else:
        ns_prefix = '{' + namespace + '}'
    # Extract list
    if cast_f is None:
        cast_f = str
    litems = elem.findall(f'{ns_prefix}litem')
    if len(litems) > 0:
        li = [cast_f(k.attrib['value']) for k in litems]
    elif elem.text is None:
        li = list()
    else:
        li = [cast_f(k) for k in elem.text.split()]
    return (elem.tag, li)


def _untagdict(
        x,
        cast_f_keys=None,
        cast_f_values=None,
        namespace=DEFAULT_NAMESPACE,
        get_order=False):
    """Extract dictionary from given tulipcon XML tag (string).

    Use functions `cast_f_keys` and `cast_f_values` for type-casting
    extracting key and value strings, respectively, or `None`.  The
    default is `None`, which means the extracted keys (resp., values)
    are left untouched (as strings), but another common case is
    `cast_f_values=int` (for "integer") or `cast_f_values=float`
    (for "floating-point numbers"), while leaving `cast_f_keys=None`
    to indicate dictionary keys are strings.

    The argument `x` can also have the type of the return value of
    `xml.etree.ElementTree.fromstring()`. This is mainly for internal
    use, e.g. by the function `untagpolytope` and some load/dumpXML
    methods elsewhere.

    Return result as 2-`tuple`, containing name of the tag (as a string)
    and the dictionary obtained from it.  If get_order is `True`, then
    return a triple, where the first two elements are as usual and the
    third is the list of keys in the order they were found.
    """
    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x
    if namespace is None or len(namespace) == 0:
        ns_prefix = ""
    else:
        ns_prefix = "{"+namespace+"}"
    # Extract dictionary
    items_li = elem.findall(ns_prefix + "item")
    if cast_f_keys is None:
        cast_f_keys = str
    if cast_f_values is None:
        cast_f_values = str
    di = dict()
    if get_order:
        key_list = list()
    for item in items_li:
        # N.B., we will overwrite duplicate keys without warning!
        di[cast_f_keys(item.attrib["key"])] = cast_f_values(
            item.attrib["value"])
        if get_order:
            key_list.append(item.attrib["key"])
    if get_order:
        return (elem.tag, di, key_list)
    else:
        return (elem.tag, di)


def load_aut_xml(x, namespace=DEFAULT_NAMESPACE):
    """Return strategy constructed from output of `gr1c`.

    @param x:
        a string or an instance of
        `xml.etree.ElementTree.fromstring()`
    @return:
        if a strategy is given in the XML string,
        return it as `networkx.DiGraph`.
        Else, return `(GRSpec, None)`,
        where the first element is the specification
        as read from the XML string.
    """
    if isinstance(x, str):
        elem = ET.fromstring(x)
    else:
        elem = x
    if namespace is None or len(namespace) == 0:
        ns_prefix = ''
    else:
        ns_prefix = '{' + namespace + '}'
    if elem.tag != ns_prefix + 'tulipcon':
        raise TypeError(
            'root tag should be tulipcon.')
    if 'version' not in elem.attrib.keys():
        raise ValueError(
            'unversioned tulipcon XML string.')
    if int(elem.attrib['version']) != 1:
        raise ValueError(
            'unsupported tulipcon XML version: '
            f'{elem.attrib["version"]}')
    # Extract discrete variables and LTL specification
    (tag_name, env_vardict, env_vars) = _untagdict(elem.find(
        ns_prefix+"env_vars"), get_order=True)
    (tag_name, sys_vardict, sys_vars) = _untagdict(elem.find(
        ns_prefix+"sys_vars"), get_order=True)
    # variables
    env_vars = _parse_vars(env_vars, env_vardict)
    sys_vars = _parse_vars(sys_vars, sys_vardict)
    # specification
    s_elem = elem.find(f'{ns_prefix}spec')
    spec = GRSpec(env_vars=env_vars, sys_vars=sys_vars)
    spec_tags = [
        'env_init', 'env_safety', 'env_prog',
        'sys_init', 'sys_safety', 'sys_prog']
    for spec_tag in spec_tags:
        if s_elem.find(ns_prefix+spec_tag) is None:
            raise ValueError(
                'invalid specification in '
                'tulipcon XML string.')
        (tag_name, li) = _untaglist(
            s_elem.find(ns_prefix + spec_tag),
            cast_f=str,
            namespace=namespace)
        li = [v.replace("&lt;", "<") for v in li]
        li = [v.replace("&gt;", ">") for v in li]
        li = [v.replace("&amp;", "&") for v in li]
        setattr(spec, spec_tag, li)
    aut_elem = elem.find(ns_prefix + "aut")
    return_now = (
        aut_elem is None or (
        aut_elem.text is None and
            len(aut_elem.getchildren()) == 0))
    if return_now:
        mach = None
        return (spec, mach)
    # Assume version 1 of tulipcon XML
    if aut_elem.attrib['type'] != 'basic':
        raise ValueError(
            'Automaton class only recognizes type "basic".')
    node_list = aut_elem.findall(f'{ns_prefix}node')
    id_list = list()
        # For more convenient searching, and
        # to catch redundancy
    A = nx.DiGraph()
    A.env_vars = env_vars
    A.sys_vars = sys_vars
    for node in node_list:
        this_id = int(node.find(f'{ns_prefix}id').text)
        # this_name = node.find(f'{ns_prefix}anno').text
        #     # Assume version 1
        (tag_name, this_name_list) = _untaglist(
            node.find(f'{ns_prefix}anno'),
            cast_f=int)
        if len(this_name_list) == 2:
            (mode, rgrad) = this_name_list
        else:
            (mode, rgrad) = (-1, -1)
        (tag_name, this_child_list) = _untaglist(
            node.find(f'{ns_prefix}child_list'),
            cast_f=int)
        if tag_name != f'{ns_prefix}child_list':
            # This really should never happen and may not even be
            # worth checking.
            raise ValueError(
                'failure of consistency check '
                'while processing aut XML string.')
        tag_name, this_state = _untagdict(
            node.find(f'{ns_prefix}state'),
            cast_f_values=int,
            namespace=namespace)
        if tag_name != f'{ns_prefix}state':
            raise ValueError(
                'failure of consistency check '
                'while processing aut XML string.')
        if this_id in id_list:
            logger.warning(
                f'duplicate nodes found: {this_id}; '
                'ignoring...')
            continue
        id_list.append(this_id)
        logger.info(
            f'loaded from `gr1c` result:\n\t{this_state}')
        A.add_node(
            this_id,
            state=copy.copy(this_state),
            mode=mode,
            rgrad=rgrad)
        for next_node in this_child_list:
            A.add_edge(this_id, next_node)
    return A


def _parse_vars(variables, vardict):
    """Helper for parsing env, sys variables."""
    domains = list()
    for v in variables:
        dom = vardict[v]
        if dom[0] == "[":
            end_ind = dom.find("]")
            if end_ind < 0:
                raise ValueError(
                    f'invalid domain for variable "{v}":  {dom}')
            dom_parts = dom[1:end_ind].split(",")
            if len(dom_parts) != 2:
                raise ValueError(
                    f'invalid domain for variable "{v}":  {dom}')
            domains.append((int(dom_parts[0]), int(dom_parts[1])))
        elif dom == 'boolean':
            domains.append('boolean')
        else:
            raise ValueError(
                'unrecognized type of domain '
                f'for variable "{v}":  {dom}')
    variables = dict(
        (v, domains[i])
        for i, v in enumerate(variables))
    return variables


def load_aut_json(x):
    """Return strategy constructed from output of `gr1c`

    @param x:
        string or
        file-like object
    @return:
        strategy as `networkx.DiGraph`,
        like the return value of `load_aut_xml`
    """
    try:
        autjs = json.loads(x)
    except TypeError:
        autjs = json.load(x)
    if autjs['version'] != 1:
        raise ValueError(
            'Only `gr1c` JSON format version 1 is supported.')
    # convert to nx
    A = nx.DiGraph()
    symtab = autjs['ENV'] + autjs['SYS']
    A.env_vars = dict([list(v.items())[0] for v in autjs['ENV']])
    A.sys_vars = dict([list(v.items())[0] for v in autjs['SYS']])
    A.initial_nodes = set()
    omit = {'state', 'trans'}
    for node_ID, d in autjs['nodes'].items():
        node_label = {k: d[k] for k in d if k not in omit}
        node_label['state'] = dict(
            [(list(symtab[i].keys())[0],
            autjs['nodes'][node_ID]['state'][i])
            for i in range(len(symtab))])
        A.add_node(node_ID, **node_label)
        if node_label['initial']:
            A.initial_nodes.add(node_ID)
    for node_ID, d in autjs['nodes'].items():
        for to_node in d['trans']:
            A.add_edge(node_ID, to_node)
    return A


def check_syntax(spec_str):
    """Check whether given string has correct `gr1c` specification syntax.

    Return `True` if syntax check passed, `False` on error.
    """
    _assert_gr1c()
    f = tempfile.TemporaryFile()
    f.write(bytes(spec_str, 'utf-8'))
    f.seek(0)
    p = subprocess.Popen(
        [f'{GR1C_BIN_PREFIX}gr1c', '-s'],
        stdin=f,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    p.wait()
    logger.debug(f'gr1c returncode: {p.returncode}')
    logger.debug(f'gr1c stdout: {p.stdout.read()}')
    if p.returncode == 0:
        return True
    else:
        logger.info(p.stdout.read())
        return False


def check_realizable(spec):
    """Decide realizability of specification.

    Consult the documentation of `synthesize` about parameters.

    @return:
        `True` if realizable, `False` if not,
        or an error occurs.
    """
    logger.info('checking realizability...')
    _assert_gr1c()
    init_option = select_options(spec)
    s = translate(spec, 'gr1c')
    f = tempfile.TemporaryFile()
    f.write(bytes(s, 'utf-8'))
    f.seek(0)
    logger.info('starting realizability check')
    p = subprocess.Popen(
        [f'{GR1C_BIN_PREFIX}gr1c',
         '-n',
         init_option,
         '-r'],
        stdin=f,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    p.wait()
    logger.info(f'`gr1c` input:\n{s}{_hl}')
    if p.returncode == 0:
        return True
    else:
        logger.info(p.stdout.read())
        return False


def synthesize(spec):
    """Synthesize strategy realizing the given specification.

    @type spec:
        `GRSpec`
    @param spec:
        specification.

    Consult the [documentation of `gr1c`](
        https://tulip-control.github.io/gr1c/md_spc_format.html#initconditions)
    for a detailed description.

    @return:
        strategy as `networkx.DiGraph`,
        or `None` if unrealizable or error occurs.
    """
    _assert_gr1c()
    init_option = select_options(spec)
    try:
        p = subprocess.Popen(
            [f'{GR1C_BIN_PREFIX}gr1c',
             '-n',
             init_option,
             '-t',
             'json'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise Exception(
                '`gr1c` not found in path.')
        else:
            raise
    s = translate(spec, 'gr1c')
    logger.info(f'\n{_hl}\n gr1c input:\n {s}\n{_hl}')
    # to make debugging by manually running gr1c easier
    fname = 'spec.gr1c'
    try:
        if logger.getEffectiveLevel() < logging.DEBUG:
            with open(fname, 'w') as f:
                f.write(s)
            logger.debug(
                f'wrote input to file "{fname}"')
    except:
        logger.error(
            f'failed to write auxiliary file: "{fname}"')
    stdoutdata, stderrdata = p.communicate(s)
    spaces = 30 * ' '
    msg = (
        f'{spaces} gr1c return code: {p.returncode}\n\n'
        f'{spaces} gr1c stdout, stderr:\n {stdoutdata}\n\n')
    if p.returncode == 0:
        logger.debug(msg)
        strategy = load_aut_json(stdoutdata)
        return strategy
    else:
        print(msg)
        return None


def select_options(spec):
    """Return `gr1c` initial option based on `GRSpec` inits."""
    # Let x denote environment variables,
    # and y component variables.
    assert not spec.moore
    assert not spec.plus_one
    if spec.qinit == r'\A \E':
        # \A x:  \E y:  EnvInit(x) => SysInit(y)
        init_option = 'ALL_ENV_EXIST_SYS_INIT'
    elif spec.qinit == r'\E \A':
        raise ValueError(
            r'`qinit = "\E \A"` not supported by `gr1c`. '
            r'Use `qinit = "\A \E"`.')
    elif spec.qinit == r'\A \A':
        # \A x, y:  EnvInit(x, y)
        # undefined SysInit
        assert not spec.sys_init, spec.sys_init
        init_option = 'ONE_SIDE_INIT'
    elif spec.qinit == r'\E \E':
        # \E x, y:  SysInit(x, y)
        # undefined EnvInit
        assert not spec.env_init, spec.env_init
        init_option = 'ONE_SIDE_INIT'
    else:
        raise ValueError(
            f'unknown option `qinit = {spec.qinit}`.')
    # The option `ALL_INIT` corresponds to:
    #   \A x, y:  EnvInit(x, y) /\ SysInit(x, y)
    # `ONE_SIDE_INIT` is used above for this case
    # (the difference is the presence of conjunction).
    return init_option


def load_mealy(filename, fformat='tulipxml'):
    """Load `gr1c` strategy from file.

    @param filename:
        file name
    @type filename:
        `str`
    @param fformat:
        file format; can be one of
        `"tulipxml"` (default),
        `"json"`. Not case sensitive.
    @type fformat:
        `str`
    @return:
        loaded strategy as an annotated graph.
    @rtype:
        `networkx.DiGraph`
    """
    s = open(filename, 'r').read()
    if fformat.lower() == 'tulipxml':
        strategy = load_aut_xml(s)
    elif fformat.lower() == 'json':
        strategy = load_aut_json(s)
    else:
        ValueError(
            '`gr1c.load_mealy()`: '
            f'Unrecognized file format, "{fformat}"')
    logger.debug(
        'Loaded strategy with nodes: \n'
        f'{strategy.nodes()}\n'
        f'and edges: \n{strategy.edges()}')
    return strategy


class GR1CSession:
    """Manage interactive session with `gr1c`.

    Given lists of environment and system variable names determine the
    order of values in state vectors for communication with the `gr1c`
    process.  Eventually there may be code to infer this directly from
    the spec file.

    **`gr1c` is assumed not to use GNU Readline.**

    Please compile it that way if you are using this class.
    (Otherwise, GNU Readline will echo commands and make interaction
    with `gr1c` more difficult.)

    The argument `prompt` is the string printed by gr1c to indicate it
    is ready for the next command.  The default value is a good guess.

    Unless otherwise indicated, command methods return `True` on
    success, `False` if error.
    """

    def __init__(
            self, spec_filename, sys_vars,
            env_vars=[],
            prompt=">>> "):
        self.spec_filename = spec_filename
        self.sys_vars = sys_vars[:]
        self.env_vars = env_vars[:]
        self.prompt = prompt
        if self.spec_filename is not None:
            self.p = subprocess.Popen(
                [f'{GR1C_BIN_PREFIX}gr1c',
                 '-i',
                 self.spec_filename],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                universal_newlines=True)
        else:
            self.p = None

    def iswinning(self, state):
        """Return `True` if given state is in winning set, `False` otherwise.

        `state` should be a dictionary with keys of variable names
        (strings) and values of the value taken by that variable in
        this state, e.g., as in nodes of the `Automaton` class.
        """
        state_vector = list(range(len(state)))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind + len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write(
            'winning ' +
            _join_as_str(state_vector) +
            '\n')
        if 'True\n' in self.p.stdout.readline():
            return True
        else:
            return False

    def getindex(self, state, goal_mode):
        if goal_mode < 0 or goal_mode > self.numgoals()-1:
            raise ValueError(
                f'Invalid goal mode requested: {goal_mode}')
        state_vector = list(range(len(state)))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind + len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write(
            'getindex ' +
            _join_as_str(state_vector) +
            f' {goal_mode}\n')
        line = self.p.stdout.readline()
        if len(self.prompt) > 0:
            loc = line.find(self.prompt)
            if loc >= 0:
                line = line[len(self.prompt):]
        return int(line[:-1])

    def env_next(self, state):
        """Return `list` of possible next environment moves, given current state.

        Format of given `state` is same as for `iswinning` method.
        """
        state_vector = list(range(len(state)))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write(
            'envnext ' +
            _join_as_str(state_vector) +
            '\n')
        env_moves = list()
        line = self.p.stdout.readline()
        while '---\n' not in line:
            if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
            env_moves.append(dict([
                (k, int(s)) for (k,s) in
                zip(self.env_vars, line.split())
            ]))
            line = self.p.stdout.readline()
        return env_moves

    def sys_nextfeas(self, state, env_move, goal_mode):
        """Return `list` of next system moves consistent with some strategy.

        Format of given `state` and `env_move` is same as for `iswinning`
        method.
        """
        if goal_mode < 0 or goal_mode > self.numgoals() - 1:
            raise ValueError(
                f'Invalid goal mode requested: {goal_mode}')
        state_vector = list(range(len(state)))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind + len(self.env_vars)] = state[self.sys_vars[ind]]
        emove_vector = list(range(len(env_move)))
        for ind in range(len(self.env_vars)):
            emove_vector[ind] = env_move[self.env_vars[ind]]
        self.p.stdin.write(
            'sysnext ' +
            _join_as_str(state_vector) +
            ' ' +
            _join_as_str(emove_vector) +
            f' {goal_mode}\n')
        sys_moves = list()
        line = self.p.stdout.readline()
        while '---\n' not in line:
            if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
            sys_moves.append(dict([
                (k, int(s)) for (k,s)
                in zip(self.sys_vars, line.split())
            ]))
            line = self.p.stdout.readline()
        return sys_moves

    def sys_nexta(self, state, env_move):
        """Return `list` of possible next system moves, whether or not winning.

        Format of given `state` and `env_move` is same as for `iswinning`
        method.
        """
        state_vector = list(range(len(state)))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        emove_vector = list(range(len(env_move)))
        for ind in range(len(self.env_vars)):
            emove_vector[ind] = env_move[self.env_vars[ind]]
        self.p.stdin.write(
            "sysnexta " +
            _join_as_str(state_vector) +
            " " +
            _join_as_str(emove_vector) +
            '\n'
        )
        sys_moves = list()
        line = self.p.stdout.readline()
        while '---\n' not in line:
            if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
            sys_moves.append(dict([
                (k, int(s)) for (k,s) in
                zip(self.sys_vars, line.split())
            ]))
            line = self.p.stdout.readline()
        return sys_moves

    def getvars(self):
        """Return string of environment and system variable names in order.

        Indices are indicated in parentheses.
        """
        self.p.stdin.write('var\n')
        line = self.p.stdout.readline()
        if len(self.prompt) > 0:
                loc = line.find(self.prompt)
                if loc >= 0:
                    line = line[len(self.prompt):]
        return line[:-1]

    def numgoals(self):
        self.p.stdin.write('numgoals\n')
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
            self.p.stdin.write('quit\n')
            returncode = self.p.wait()
            self.p = None
            if returncode != 0:
                self.spec_filename = None
                return False
        if spec_filename is not None:
            self.spec_filename = spec_filename
        if self.spec_filename is not None:
            self.p = subprocess.Popen(
                [f'{GR1C_BIN_PREFIX}gr1c',
                 '-i',
                 self.spec_filename],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                universal_newlines=True)
        else:
            self.p = None
        return True

    def close(self):
        """End session, and terminate `gr1c` subprocess."""
        self.p.stdin.write('quit\n')
        returncode = self.p.wait()
        self.p = None
        if returncode != 0:
            return False
        else:
            return True


def _join_as_str(items) -> str:
    """Return space-joined strings of `items`."""
    return ' '.join(map(str, items))
