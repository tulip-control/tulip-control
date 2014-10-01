from tulip.transys.transys import OpenFTS
from tulip import synth
import copy, re, pycudd
from tulip.spec import GRSpec
from tulip.interfaces import jtlv


def _conj(s0, s1):
    if len(s0) > 0:
        if len(s1) > 0:
            return '('+s0+')&&('+s1+')'
        return s0
    if len(s1) > 0:
        return s1
    return ''


def _conj_all(set0, parth = True):
    set0 = filter(None,set0)
    if parth:
        set0 = ['('+x+')' for x in set0]
    return '&&'.join(set0)


def _disj(s0,s1):
    if len(s0) > 0:
        if len(s1) > 0:
            return '('+s0+')||('+s1+')'
        return s0
    if len(s1) > 0:
        return s1
    return ''


def _disj_all(set0):
    set0 = filter(None,set0)
    set0 = ['('+x+')' for x in set0]
    return '||'.join(set0)


def _iff(s0,s1):
    return '(('+s0+')&&('+s1+'))||(!('+s0+')&&!('+s1+'))'


def _replace_prop(prop, repl, spec):
    '''Replace proposition in specification
    '''
    pre_match = '(^|(?<=[&|!<\->() ]))'
    post_match = '((?=[&|!<\->() ])|$)'
    return re.sub(pre_match+prop+post_match,repl,spec)


def _list_remove_duplicates(lst):
    #Sort by number of twos
    lst = sorted(lst, key=lambda row: sum([2 == x for x in row]), reverse=True)
    reduced = True
    while reduced:
        reduced = False
        i = 0
        while i < len(lst):
            l1 = lst[i]
            j = i+1
            while j < len(lst):
                l2 = lst[j]
                #assume duplicate until disproven
                duplicate = True
                for index in range(0,len(l1)):
                    if l1[index] == 2:
                        #May be removable, check next index
                        continue
                    if l1[index] == l2[index]:
                        #May be removable, check next index
                        continue
                    #Is not removable, check next row
                    duplicate = False
                    break
                if duplicate:
                    del lst[j]
                    reduced = True
                else:
                    j += 1
            i += 1
    return lst

def _list_reduce(lst):
    '''Compresses the information in a list of lists of binary numbers

        Assuming same binary combination is not represented twice
        Results in list of list of 0,1 or 2, where 2 means 0 and 1
        Ex: [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0]] --> [[0,2,2],[1,0,0]]
    @param lst: list of lists of binary numbers
    '''
    reduced = True
    while reduced:
        reduced = False
        i = 0
        while i < len(lst):
            l1 = lst[i]
            j = i +1
            while j < len(lst):
                l2 = lst[j]
                diff = [l1[k] != l2[k] for k in xrange(len(l1))]
                if sum(diff) == 1:
                    l1[diff.index(1)] = 2
                    del lst[j]
                    reduced = True
                j += 1
            i += 1
    return lst

def _cube_to_str(lst, vars, neg = False):
    vals = []
    for i in range(len(lst)):
        vals.append([])
        for j in range(len(lst[i])):
            if (lst[i][j] == 0 and not neg) or (lst[i][j] == 1 and neg):
                vals[i].append('!'+vars[j])
            if (lst[i][j] == 1 and not neg) or (lst[i][j] == 0 and neg):
                vals[i].append(vars[j])
    if not neg:
        return _disj_all([_conj_all(val, False) for val in vals])
    else:
        return _conj_all([_disj_all(val) for val in vals])



#TODO: check init -> phi tautology
class RHTLPProb:
    def __init__(self, disc_dyn, specs, parts, mappings, plan_sets  = None,\
                 phi='', repl_prog = True, add_end_state = True,\
                 partition_mapping=None):
        '''

        @type disc_dyn: OpenFTS
        @param disc_dyn: The discrete dynamics of the system

        @type specs: GRSpec
        @param specs: The specification for the problem.

        @type parts: iterable
        @param parts: List of names of partitions

        @type mappings: dict, {partition1:partition2,...}
        @param mappings: The the short horizon mapping for each of the partitions

        @type plan_sets: dict of iterable, {partition1: {partition1, partition2},...}
        @param plan_sets: The partitions over which to plan when planning from
            partition to mapping of partition. Defaults to entire space.

        @type phi: str
        @param phi: Invariant that restricts short horizon problems. Will be
            both initial and safety condition. Boolean composition of atomic
            propositions from system and environment variables.

        @type repl_prog: bool
        @param repl_prog: Whether or not to replaces the progress
            (always eventually) spec with (eventually) for all but the last
            short horizon problem. Useful when the system may not be able to
            stay in all the partitions.

        @type add_end_state: bool
        @param add_end_state: Add a dummy end state in which the system will
            always be able to enter. Makes sure that eventually requirement can
            be fulfilled even when an infinite trace can not be created.

        @type: Dictionary of lists of nodes or atomic propositions
        @param partition_mapping: Mapping of the partitions to either states or
            atomic propositions. Removes the need of introducing propositions
            in the system for each of the partitions.

        @return:
        '''
        #TODO handle multiple progress

        self.parts = parts
        self.disc_dyn = disc_dyn
        self.spec = specs
        self.phi = phi
        self.mappings = mappings

        if plan_sets is None:
            self.plan_sets = parts
        else:
            self.plan_sets = plan_sets

        self.shortRHTLProbs = {}

        self.global_aps = copy.deepcopy(disc_dyn.aps)

        if partition_mapping is not None:
            self.partition_mapping = self._partition_mapping_to_nodes(partition_mapping)
        else:
            self.partition_mapping = None

        self._verify_inputs()

        #To ensure that dead end in shortrhtlp problems doens't inhibit synth
        if add_end_state:
            self.disc_dyn.states.add('endState', ap={})
            for node in self.disc_dyn.nodes():
                self.disc_dyn.add_edge(node, 'endState')

        for w in parts:
            if w == self.spec.sys_prog[0]:
                repl = False
            else:
                repl = repl_prog
            print "Creating short problem "+w
            self.shortRHTLProbs[w] = ShortRHTLProb(self, w, mappings[w], \
                                                   plan_sets[w], repl,   \
                                                   add_end_state,
                                                   partition_mapping)

    def generate_phi(self):
        perfect = True
        phi = self.phi
        checked_parts = {self.spec.sys_prog[0]}
        #Indicates the order in which the partitions were checked
        check_order = [self.spec.sys_prog[0]]
        remaining_parts = set(self.plan_sets.keys()).difference(checked_parts)

        while len(remaining_parts) > 0:
            #Get all remaining partitions that map to any in checked_parts
            to_any = [w for w in remaining_parts \
                      if self.mappings[w] in checked_parts]
            #Get all partitions from plan_sets that are not already checked
            plan_not_checked = [set(self.plan_sets[w]).difference(checked_parts) \
                                for w in to_any]
            #Pick sets that only plan to the ones already checked (and itself)
            strictly_forward = [w for w in to_any \
                                if len(plan_not_checked[to_any.index(w)]) == 1]
            if len(strictly_forward) > 0:
                for w in strictly_forward:
                    succ = False
                    while not succ:
                        (succ, nr_added, phi) = self.shortRHTLProbs[w].improve_invariant(phi)
                        if nr_added > 0:
                            print w+' added nodes:\n'
                            print phi
                    checked_parts.add(w)
                    check_order.append(w)
                    remaining_parts.remove(w)
            else:
                perfect = False
                raise NotImplementedError("No strict order found among partitions")
                #TODO something smart

        return phi

    def reduce_spec(self):
        reduced_specs = {}
        for w, shortprob in self.shortRHTLProbs.items():
            reduced_specs[w] = shortprob.reduce_spec()
        return reduced_specs

    def synthesize(self, extra_phi=''):
        '''Generates L{MealyMachine} or counter-example for each shortRHTLPProb

        @rtype list of instances of L{MealyMachine} or counter-examples, see
            jtlv.synthesize()
        '''

        aut = {}
        for w, shortprob in self.shortRHTLProbs.items():
            aut[w] = shortprob.synthesize(extra_phi=extra_phi)
        return aut

    def _verify_inputs(self):
        if not isinstance(self.disc_dyn, OpenFTS):
            raise ValueError('The discrete dynamics has to be instance of OpenFTS')

        if not isinstance(self.spec,GRSpec):
            raise ValueError('The specification has to be instance of GRSpec')

        #Verify that parts is partitioning of nodes in disc_dyn
        all_nodes = set()
        for w in self.parts:
            if self.partition_mapping is None:
                nodes = self.prop_to_nodes([w])
            else:
                nodes = set(self.partition_mapping[w])
            if len(all_nodes.intersection(nodes)) != 0:
                raise ValueError('The same node can not exist in several partitions')
            all_nodes.update(nodes)
        if len(set(self.disc_dyn.nodes()).difference(all_nodes)) != 0:
            raise ValueError('Every node must exist in some partition')


        if self.spec.sys_prog[0] not in self.parts:
            raise ValueError('One partition must correspond to progress goal')

        #Verify mapping is partially ordered
        has_path_to_goal = {self.spec.sys_prog[0]}
        updated = True
        while updated:
            updated = False
            for w in self.parts:
                if w not in has_path_to_goal:
                    if self.mappings[w] in has_path_to_goal:
                        has_path_to_goal.add(w)
                        updated = True
        if len(has_path_to_goal) != len(self.parts):
            raise ValueError('There has to be a path from every partition to progress goal through the mapping')

    def _partition_mapping_to_nodes(self,partition_mapping):
        ''' Verifies partition mapping and replaces aps with nodes

            Makes sure that partition mapping contains either only propositions
            or nodes. If contains only propositions, replaces these with the
            corresponding nodes.
        '''

        #Try if partition_mapping contains aps and not nodes
        elems = set()
        for w in partition_mapping:
            elems.update(partition_mapping[w])

        #Only contains nodes
        if len(elems.difference(set(self.disc_dyn.nodes()))) == 0:
            pass
        #Only contains aps:
        elif len(elems.difference(set(self.global_aps))) == 0:
            new_partition_mapping = {}
            nodes = set()
            for w in partition_mapping:
                new_partition_mapping = []
                for ap in partition_mapping[w]:
                    ap_nodes = self.prop_to_nodes([ap])
                    if not len(ap_nodes.intersection(nodes))  == 0:
                        raise ValueError('Two propositions in partition_mapping are not allowed to refer to the same node')
                    else:
                        new_partition_mapping.extend(ap_nodes)
                        nodes.update(ap_nodes)
                partition_mapping[w] = new_partition_mapping
        else:
             raise ValueError('Partition mapping has to consist of either nodes or propositions')
        return partition_mapping

    def prop_to_nodes(self,props):
        '''Get all nodes with any of the propositions in list C{props}
        '''
        nodes = set()
        for prop in props:
            for node, value in self.disc_dyn.node.items():
                if 'ap' in value and prop in value['ap']:
                        nodes.update([node])
        return nodes

class ShortRHTLProb:
    def __init__(self, rhtlp, w, mapping, plan_set, repl_prog, add_end_state,\
                 partition_mapping = None):
        self.rhtlp = rhtlp
        self.w = w
        self.mapping = mapping
        self.plan_set = plan_set
        self.local_disc_dyn = self.rhtlp.disc_dyn.copy()
        if partition_mapping is None:
            self.w_nodes = self.prop_to_nodes(w)
        else:
            self.w_nodes = partition_mapping[w]
        self.w_aps = self.props_in_nodes(self.w_nodes)
        self._final_spec = None

        self.plan_set_nodes = set()
        for part in plan_set:
            if partition_mapping is None:
                self.plan_set_nodes.update(self.prop_to_nodes(plan_set))
            else:
                self.plan_set_nodes.update(partition_mapping[part])
        self.plan_set_aps = self.props_in_nodes(self.plan_set_nodes)

        if add_end_state:
            self.plan_set_nodes.add('endState')

        self.not_plan_set_aps = set(self.local_disc_dyn.aps).difference(self.plan_set_aps)
        self.not_plan_set_nodes = set(self.local_disc_dyn.nodes()).difference(self.plan_set_nodes)

        for node in self.rhtlp.disc_dyn.nodes():
            if node not in self.plan_set_nodes:
                #Remember to handle exclusion of nodes not in plan set,
                # in case they appare in some other spec,
                # to make sure they are false
                self.local_disc_dyn.remove_node(node)
        for ap in self.not_plan_set_aps:
            self.local_disc_dyn.aps.remove(ap)

        self.local_phi, self.local_spec = \
            self.create_local_spec(repl_prog, add_end_state, partition_mapping=partition_mapping)

    def synthesize(self, check_realizable = False, extra_phi = ''):
        '''Synthesize a controller for the problem

        @param extra_phi: Optional additional invariant,
            will skip reduction if set
        @param check_realizable: Only checks realizability if True
        @return: If check_realizable = False: Controller for problem if
            successful, otherwise counter example.
            If check_realizable = True: Boolean
        @rtype C{MealyMachine}, counter example or C{boolean}
        '''
        if self._final_spec is not None and extra_phi == '':
            spec = self._final_spec
        else:
            spec = self.reduce_spec(skip_reduction=True, extra_phi=extra_phi)

        if check_realizable:
             #Consider all initial states! init_option=0
            realizable = jtlv.check_realizable(spec=spec,init_option=0)
            return realizable
        else:
            from tulip.interfaces import gr1c
            #aut = gr1c.synthesize(spec=spec, init_option="ALL_INIT")
            aut = jtlv.synthesize(spec=spec,init_option=0)
            return aut

    def reduce_spec(self, extra_phi = '', skip_reduction=False):
        '''Create final local spec and remove unnecessary variables

        @param extra_phi: optional extra invariant
        @param skip_reduction: Skips reduction part if True, default False
        @return: Specification ready for solver
        '''
        spec = self.local_spec.copy()
        phi = _conj(self.local_phi, extra_phi)

        #This is only needed for the extra_phi
        phi = self._replace_aps_and_props(self.rhtlp.partition_mapping, spec, phi)

        spec.sys_init = [_conj(_conj_all(spec.sys_init), phi)]
        spec.sys_prog = [_conj(_conj_all(spec.sys_prog), phi)]
        #This could force the environment too much?
        #spec.env_init =  [phi]

        spec.sys_safety = [_conj(_conj_all(spec.sys_safety), phi)]

        #Can't use tulip synthesize to get counterexamples
        #TODO This only works with bool states in GR1c so far
        bool_states=True
        action_vars=None
        bool_actions=False
        bool_states, action_vars, bool_actions = synth._check_solver_options(
            'gr1c', bool_states, action_vars, bool_actions
        )
        full_spec = synth.spec_plus_sys(specs=spec,\
                                             env = None,\
                                             sys=self.local_disc_dyn,\
                                             ignore_env_init = False,\
                                             ignore_sys_init = True,\
                                             bool_states = bool_states,\
                                             action_vars = action_vars,\
                                             bool_actions = bool_actions)
        if skip_reduction:
            self._final_spec = full_spec
            return full_spec
        else:
            print "Reducing spec for problem "+self.w
            reduced_spec = reduce_spec(full_spec,
                                       env_vars=full_spec.input_variables.keys(),
                                       sys_vars=full_spec.output_variables.keys())
            self._final_spec = reduced_spec
            return reduced_spec

    def improve_invariant(self, phi):
        '''Improve invariant phi by finding counter examples
        '''
        succ = False
        nr_added = 0
        print 'Checking '+self.w+' for realizability\n'
        realizable = self.synthesize(extra_phi=phi, check_realizable=True)
        if realizable:
            return (True, nr_added, phi)
        print 'Finding counter examples for '+self.w+'\n'
        counter_ex = self.synthesize(extra_phi=phi, check_realizable=False)

        #Ignore dummy variable
        #TODO do not hardcode this
        excep = ['progReached']
        #Rules to exclude in invariant
        exclude = {}

        sys_vars = [var for var in self.local_spec.sys_vars if var not in excep]
        env_vars = [var for var in self.local_spec.env_vars if var not in excep]
        vars = sys_vars+env_vars
        if any([self.local_spec.sys_vars[var] != 'boolean' for var in sys_vars]):
            raise NotImplementedError('Only boolean variables supported')
        if any([self.local_spec.env_vars[var] != 'boolean' for var in env_vars]):
            raise NotImplementedError('Only boolean variables supported')

        for example in counter_ex:
            state = self.get_sys_state_from_props(example)
            prop_list = []
            for var in vars:
                prop_list.append(example[var])
            if state not in exclude:
                exclude[state] = []
            exclude[state].append(prop_list)
            nr_added +=1
        add_phi = []
        print 'List of variables are:'
        print vars
        for state in exclude:
            _list_reduce(exclude[state])
            print 'Imposing constraint for state '+state+' with:'
            print exclude[state]
            add_phi.append(_disj('!'+state,\
                                      _cube_to_str(exclude[state], vars, True)))
        print 'Adding constraint:'
        print add_phi
        phi = _conj(phi, _conj_all(add_phi))
        #phi = _conj(phi, '!('+_conj_all(example_props)+')')
        return (succ, nr_added, phi)



    def get_sys_state_from_props(self, example):
        '''Gets the transition system state represented in counter example

            Assuming mutual exclusion between states, first match is returned
        '''
        for state in self.rhtlp.disc_dyn.states():
            if state in example and example[state] == 1:
                return state


    def create_local_spec(self, repl_prog, add_end_state, partition_mapping=None):
        sys_vars, env_vars, sys_safe, env_safe, env_prog = self._copy_global_spec()
        env_init = ''
        sys_init = self.w

        plan_set_str = _disj_all(self.plan_set)

        if add_end_state:
            sys_safe = _conj(sys_safe,\
                                      '('+plan_set_str+'||endState)')
        else:
            sys_safe = _conj(sys_safe, plan_set_str)

        if repl_prog:
            sys_vars, sys_init, sys_safe, sys_prog = \
                self._add_eventually( sys_vars, sys_init, sys_safe, self.mapping)
        else:
            sys_prog = self.mapping

        local_spec = GRSpec(env_vars, sys_vars, env_init, sys_init,\
            env_safe, sys_safe, env_prog, sys_prog)

        local_phi = self.rhtlp.phi

        #TODO spec.replace in spec too slow, move partition mapping change to original spec
        local_phi = self._replace_aps_and_props(partition_mapping, local_spec, local_phi)

        return local_phi, local_spec

    def _replace_aps_and_props(self, partition_mapping, local_spec, local_phi):
        if partition_mapping != None:
            self._replace_in_spec(local_spec, partition_mapping)
            local_phi = self._replace_in_string(local_phi, partition_mapping)

        replace_ap_and_props = True
        if replace_ap_and_props:
            ap_mapping = {}
            for prop in self.not_plan_set_aps:
                ap_mapping[prop] = ['False']
            self._replace_in_spec(local_spec, ap_mapping)
            local_phi = self._replace_in_string(local_phi, ap_mapping)

            for node in self.not_plan_set_nodes:
                ap_mapping[node] = ['False']
            self._replace_in_spec(local_spec, ap_mapping)
            local_phi = self._replace_in_string(local_phi, ap_mapping)

            ap_mapping = {}
            for ap in self.plan_set_aps:
                nodes = self.prop_to_nodes([ap])
                nodes.difference_update(self.not_plan_set_nodes)
                ap_mapping[ap] = ['('+_disj_all(nodes)+')']
            self._replace_in_spec(local_spec, ap_mapping)
            local_phi = self._replace_in_string(local_phi, ap_mapping)
        return local_phi

    def _replace_in_spec(self, spec, mapping):
        '''Replace variables in spec using dictionary 'mapping' and removes
            them from lists of vars

        @ptype mapping: dict of list of string
        @param mapping: dictionary where the keys represent variables to be
            replaced and the values are lists of strings of the names of the
            nodes that replace them
        '''
        string_mapping = {}
        for key, value in mapping.items():
            string_mapping[key] = _disj_all(value)
            spec.sym_to_prop(string_mapping)
            if key in spec.sys_vars:
                del spec.sys_vars[key]
            if key in spec.env_vars:
                del spec.env_vars[key]

    def _replace_in_string(self,str0, mapping):
        '''Replace variables in spec using dictionary 'mapping' and removes
            them from lists of vars

        @ptype mapping: dict of list of string
        @param mapping: dictionary where the keys represent variables to be
            replaced and the values are lists of strings of the names of the
            nodes that replace them
        @return modified string
        '''
        string_mapping = {}
        liststr = [str0]
        for key, value in mapping.items():
            string_mapping[key] = _disj_all(value)
            liststr[0] = _replace_prop(key, string_mapping[key], liststr[0])
            #_sub_all(liststr, key, string_mapping[key])
        return liststr[0]

    def _find_aps(self, ap):
        '''Find local atomic propositions existing in nodes with this ap
        '''
        aps = set()
        for node_name in self.rhtlp.disc_dyn.node:
            node = self.rhtlp.disc_dyn.node[node_name]
            if ap in node['ap']:
                aps.update(node['ap'].copy())
        return aps

    def _find_aps_from(self, set0):
        '''Find atomic propositions existing in node with any of aps from set0
        '''
        aps = set()
        for ap in set0:
            aps.update(self._find_aps(ap))
        return aps

    def _find_aps_not_in(self, set0):
        '''Find global atomic propositions NOT existing in set0
        '''
        aps = set()
        for ap in self.rhtlp.global_aps:
            if not ap in set0:
                aps.add(ap)
        return aps

    def _copy_global_spec(self):
        '''Copy sys_vars, env_vars, sys_safe, env_safe and env_prog
            from the global specification
        '''
        sys_vars = copy.copy(self.rhtlp.spec.sys_vars)
        env_vars = copy.copy(self.rhtlp.spec.env_vars)


        sys_safe = copy.copy(self.rhtlp.spec.sys_safety)
        if len(sys_safe) > 0:
            sys_safe = _conj_all(sys_safe)
        else:
            sys_safe = ''

        env_safe = copy.copy(self.rhtlp.spec.env_safety)
        if len(env_safe) > 0:
            env_safe = _conj_all(env_safe)
        else:
            env_safe = ''

        env_prog = copy.copy(self.rhtlp.spec.env_prog)
        if len(env_prog) > 0:
            env_prog = _conj_all(env_prog)
        else:
            env_prog = ''

        return sys_vars, env_vars, sys_safe, env_safe, env_prog

    def _add_eventually(self, sys_vars, sys_init, sys_safe,\
                        goal, prog_var='progReached'):
        '''Adds eventually 'goal' to spec using variable 'prog_var'
        '''
        #TODO check protected variabe
        sys_vars[prog_var] = 'boolean'
        remember = self._remember_progress(goal, prog_var)
        sys_safe = _conj(sys_safe, remember)
        sys_prog = prog_var

        sys_init = '('+_conj(sys_init, _iff(prog_var, goal))+')'

        return sys_vars, sys_init, sys_safe, sys_prog


    def _remember_progress(self, progress, prog_reach):
        '''Return specification that remembers 'progress' using ap 'prog_reach'
        '''
        #prog_reach -> next(prog_reach)
        spec = '(!'+prog_reach+'||X('+prog_reach+'))'
        #progress -> prog_reach
        spec += '&&((!'+progress+')||'+prog_reach+')'
        #!(prog_reach||progress||next(progress))->!x(prog_reach)
        spec += '&&('+prog_reach+'||'+progress+'||'+'X('+progress+')||!X('+prog_reach+'))'
        return spec

    def _replace_ap_with_nodes(self, ap, spec):
        '''Replace 'ap' with disjunction of all nodes containing 'ap'
        @type ap: string
        @param ap: the proposition to replace
        @type spec: string
        @param spec: string in which replacement should occur
        '''
        nodes = self.prop_to_nodes([ap])
        nodes_str = '('+_disj_all(nodes)+')'
        return _replace_prop(ap, nodes_str, spec)

    def prop_to_nodes(self,props):
        '''Get all nodes with any of the propositions in list C{props}
        '''
        #
        nodes = set()
        for prop in props:
            for node, value in self.local_disc_dyn.node.items():
                if 'ap' in value and prop in value['ap']:
                        nodes.update([node])
        return nodes

    def props_in_nodes(self,nodes):
        '''Get all ap that exist in any of the nodes in list C{nodes}
        '''
        aps = set()
        for node, value in self.local_disc_dyn.node.items():
            if 'ap' in value and node in nodes:
                    aps.update(value['ap'])
        return aps

def reduce_spec(spec, env_vars, sys_vars):

    if len(spec.sys_safety) > 0:
        sys_safe = "&&".join(["("+row+")" for row in spec.sys_safety])
    else:
        sys_safe = "TRUE"
    if len(spec.sys_init) > 0:
        sys_init = "&&".join(["("+row+")" for row in spec.sys_init])
    else:
        sys_init = "TRUE"
    if len(spec.sys_prog) > 0:
        sys_prog = "&&".join(["("+row+")" for row in spec.sys_prog])
    else:
        sys_prog = "TRUE"

    ss_bdd, ss_vars, ss_mgr = formula_to_bdd(sys_safe)
    si_bdd, si_vars, si_mgr = formula_to_bdd(sys_init)
    sp_bdd, sp_vars, sp_mgr = formula_to_bdd(sys_prog)

    env_not_in_sys = []
    for var in env_vars:
        if (var not in ss_vars.keys())\
                and (var not in si_vars.keys())\
                and (var not in sp_vars.keys()):
            env_not_in_sys.append(var)

    ss_support = get_support(ss_mgr, ss_bdd)
    si_support = get_support(si_mgr, si_bdd)
    sp_support = get_support(sp_mgr, sp_bdd)

    indep_env_vars_in_sys_bdd = []
    for var in env_vars:
        ss_indep = False
        si_indep = False
        sp_indep = False

        ss_indep = not in_support(var, ss_vars, ss_support)
        si_indep = not in_support(var, si_vars, si_support)
        sp_indep = not in_support(var, sp_vars, sp_support)

        if ss_indep and si_indep and sp_indep:
            spec.sym_to_prop({var:"FALSE"})
            del spec.env_vars[var]

    #CREATE NEW SPEC
    new_ss = spec_from_bdd(ss_bdd, ss_vars, ss_mgr, indep_env_vars_in_sys_bdd)
    new_si = spec_from_bdd(si_bdd, si_vars, si_mgr, indep_env_vars_in_sys_bdd)
    new_sp = spec_from_bdd(sp_bdd, sp_vars, sp_mgr, indep_env_vars_in_sys_bdd)

    if len(spec.env_safety) > 0:
        env_safe = "&&".join(["("+row+")" for row in spec.env_safety])
    else:
        env_safe = "TRUE"

    if len(spec.env_init) > 0:
        env_init = "&&".join(["("+row+")" for row in spec.env_init])
    else:
        env_init = "TRUE"

    if len(spec.env_prog) > 0:
        env_prog = "&&".join(["("+row+")" for row in spec.env_prog])
    else:
        env_prog = "TRUE"

    es_bdd, es_vars, es_mgr = formula_to_bdd(env_safe)
    ei_bdd, ei_vars, ei_mgr = formula_to_bdd(env_init)
    ep_bdd, ep_vars, ep_mgr = formula_to_bdd(env_prog)

    vars_to_remove_from_es = [var for var in indep_env_vars_in_sys_bdd+env_not_in_sys]
    vars_to_remove_from_es.extend(["X "+var for var in vars_to_remove_from_es])

    es_exist_bdd = exist_abstract(es_bdd, es_vars, es_mgr,
                                  vars_to_remove_from_es)
    new_es = spec_from_bdd(es_exist_bdd, es_vars, es_mgr, [])

    ei_exist_bdd = exist_abstract(ei_bdd, ei_vars, ei_mgr,
                                  indep_env_vars_in_sys_bdd+env_not_in_sys)

    ep_exist_bdd = exist_abstract(ep_bdd, ep_vars, ep_mgr,
                                  indep_env_vars_in_sys_bdd+env_not_in_sys)

    new_ei = spec_from_bdd(ei_exist_bdd, ei_vars, ei_mgr, [])
    new_ep = spec_from_bdd(ep_exist_bdd, ep_vars, ep_mgr, [])
    #new_ei = spec_from_bdd(ei_bdd, ei_vars, ei_mgr, indep_env_vars_in_sys_bdd+env_not_in_sys)
    #new_ep = spec_from_bdd(ep_bdd, ep_vars, ep_mgr, indep_env_vars_in_sys_bdd+env_not_in_sys)


    #prev_len = -1
    #while len(ei_cube_set) != prev_len:
    #    prev_len = len(ei_cube_set)
    #    print prev_len
    #    _list_remove_duplicates(ei_cube_set)
    #    _list_reduce(ei_cube_set)

    #prev_len = -1
    #while len(ep_cube_set) != prev_len:
    #    prev_len = len(ep_cube_set)
    #    print prev_len
    #    _list_remove_duplicates(ep_cube_set)
    #    _list_reduce(ep_cube_set)
    #print "done reducing"

    new_env_vars = []
    for var in env_vars:
        if var not in indep_env_vars_in_sys_bdd+env_not_in_sys:
            new_env_vars.append(var)

    reduced_spec = GRSpec(env_vars = new_env_vars, sys_vars=sys_vars,
                          sys_init = new_si, env_init = new_ei,
                          sys_safety = new_ss, env_safety = new_es,
                          sys_prog = new_sp, env_prog = new_ep)

    #This might be needed depending on gc bug in pycudd
    #del ss_mgr
    #del si_mgr
    #del sp_mgr
    #del es_mgr
    #del ep_mgr
    #del ei_mgr
    return reduced_spec

def exist_abstract(bdd, vars, mgr, vars_to_abstract):
    '''Exist abstract away variables in vars_to_abstract
    '''
    #Create conjunction of variables to abstract away
    all_cube = bdd_conj(vars_to_abstract, vars, mgr)
    #New bdd with variables abstracted away
    return bdd.ExistAbstract(all_cube)

def get_support(mgr, bdd):
    '''Gets cube of variables that can affect evaluation of bdd

    @type mgr: DdManager
    @type bdd: DdNode

    @rtype: tuple
    '''
    mgr.SetDefault()
    for cube in bdd.Support():
        return cube

def in_support(var, vars, support):
    if (var in vars):
        index = vars[var][1]
        if support[index] != 2:
            return True
    return False

def spec_from_bdd(bdd, vars, mgr, exclude):
    mgr.SetDefault()
    pycudd.set_iter_meth(0)
    cube_set_str = []
    for cube in bdd:
        cube_str = []
        for key, val in vars.iteritems():
            if key in exclude:
                continue
            if key[:2] == "X " and key[2:] in exclude:
                continue

            if key[:2] == "X ":
                key = "X("+key[2:]+")"
            if cube[val[1]] == 0:
                cube_str.append("!"+key)
            elif cube[val[1]] == 1:
                cube_str.append(key)
        cube_set_str.append(cube_str)
    spec = "||".join(['('+'&&'.join(row)+')' for row in cube_set_str])
    if spec == "()":
        return ""
    return spec

def formula_to_bdd(formula, mgr = None, vars = None, parser = "pyparsing"):
    from tulip.spec.ast_extras import to_NNF, nnf_op_to_var, simplify, NNF_to_CNF
    import pycudd
    from tulip.spec.parser import parse

    if mgr == None and vars != None:
        raise ValueError("dDNodes requires a manager")
    if vars == None:
        vars = {}
    if mgr == None:
        mgr = pycudd.DdManager()
        mgr.AutodynEnable(4)
    mgr.SetDefault()

    #For large formulas, change to "pyparsing" to avoid recursion limit
    #Pyparsing requires primed variables instead of X(*)
    import re
    #formula = re.sub('X\((.*?)\)', lambda x:x.group(1)+'\'', formula)
    ast = parse(formula,parser)
    ast = nnf_op_to_var(to_NNF(ast))

    bdd = ast.to_bdd(mgr, vars)
    return bdd, vars, mgr

def ddNodes_from_list(var_names):
    import pycudd
    mgr = pycudd.DdManager()
    #mgr.AutodynEnable(4)
    mgr.SetDefault()
    vars = {}
    for var_name in var_names:
        if var_name not in vars.keys():
            node_index = len(vars)+1
            vars[var_name] = (mgr.IthVar(node_index), node_index)
            vars["X "+var_name] = (mgr.IthVar(node_index+1), node_index+1)
    return vars, mgr

def create_spec_info(spec):
    parts = ["sys_safety", "env_safety", "sys_prog", "env_prog", "init"]
    spec_info = {"parts": parts, "part": {}, "var" : {}, "values": {},
                 "sys_vars": spec.sys_vars.keys(),
                 "env_vars": spec.env_vars.keys()}
    ###### INSPECT THE SPECS AND SAVE DATA ABOUT SPECS AND VARS#####
    #Inspect the specs and save data about specs and vars
    for part in parts:
        if part == "init":
            formula = _conj(_conj_all(spec.sys_init), _conj_all(spec.env_init))
        else:
            formula = _conj_all(getattr(spec, part))
        if formula == "":
            formula = "True"
        spec_info["part"][part] = {"formula": formula, "vars": [] }
        #TODO we don't care about the actual bdd at this stage
        bdd, vars, mgr = formula_to_bdd(formula, parser="ply")
        #spec_info["part"][part]["bdd"] = bdd
        for var in spec.sys_vars:
            if var in vars.keys() or "X "+var in vars.keys():
                spec_info["part"][part]["vars"].append(var)
        for var in spec.env_vars:
            if var in vars.keys() or  "X "+var in vars.keys():
                spec_info["part"][part]["vars"].append(var)

    #Collect info about vars locations
    for var in spec_info["sys_vars"]+spec_info["env_vars"]:
        if var in spec_info["sys_vars"]:
            value = 0
        if var in spec_info["env_vars"]:
            value = 1
        for i, part in enumerate(parts):
            if var in spec_info["part"][part]["vars"]:
                value += 2**(i+1)
        spec_info["var"][var] = {"value": value}
        if value in spec_info["values"]:
            spec_info["values"][value]["nr"] += 1
            spec_info["values"][value]["vars"].append(var)
        else:
            spec_info["values"][value] = {"nr": 1, "vars": [var]}

    return spec_info

def bdd_conj(conj_vars, all_vars, mgr):
    mgr.SetDefault()
    pycudd.set_iter_meth(0)
    all_cube = mgr.ReadOne()
    for var in conj_vars:
        if var in all_vars.keys():
            all_cube = all_cube & all_vars[var][0]
    return all_cube

def combination_perturbations(lst):
    """Create product of permutations

    Outputs product of permutations of each of the lists in lst
    Example:
        input: lst = [[1,2],[3,4],[5]]
    Output:
        [[1,2,3,4,5], [1,2,4,3,5], [2,1,3,4,5], [2,1,3,4,5]

    @param lst: List of lists
    """
    import itertools
    out = []
    if len(lst) > 1:
        for perm in itertools.permutations(lst[0]):
            perm_list = [x for x in perm]
            extended_perm = [perm_list+comb for comb in combination_perturbations(lst[1:])]
            out.extend(extended_perm)
    else:
        for perm in itertools.permutations(lst[0]):
            out.append([x for x in perm])
    return out

def spec_equals(spec0, spec1):
    #Check same number of variables
    if len(spec0["sys_vars"]) != len(spec1["sys_vars"]):
        return False
    if len(spec0["env_vars"]) != len(spec1["env_vars"]):
        return False
    #Check same variable occurrences
    for val0, val_info0 in spec0["values"].items():
        if not val0 in spec0["values"].keys():
            return False
        if val_info0["nr"] != spec1["values"][val0]["nr"]:
            return False
    print "Spec "+spec0["name"]+" and "\
          +spec1["name"]+" have the same variable distribution"


    #Check if info has been previously generated
    if "bdd_info" in spec1.keys():
        mgr = spec1["bdd_info"]["mgr"]
        mgr.SetDefault()
        node_list = spec1["bdd_info"]["node_list"]
        spec_1_cubes = spec1["bdd_info"]["cubes"]
    else:
        spec1["bdd_info"] = {}
        mgr = pycudd.DdManager()
        spec1["bdd_info"]["mgr"] = mgr
        mgr.SetDefault()
        node_list = []
        #for value in sorted(spec1["values"]):
        #    for var in spec1["values"][value]["vars"]:
        for i in range(len(spec0["sys_vars"])+len(spec0["env_vars"])):
            node_list.append(mgr.NewVar())
            node_list.append(mgr.NewVar())
        spec1["bdd_info"]["node_list"] = node_list
        #Do the following for each permutation and each formula
        # (only when permuted variables exist in that formula)

        #Keep this constant vary other
        vars1 = {}
        i = 0
        print spec1["values"]
        print spec0["values"]
        print len(node_list)
        for value in sorted(spec1["values"]):
            for var in spec1["values"][value]["vars"]:
                vars1[var] = (node_list[i*2], i*2)
                vars1["X "+var] = (node_list[i*2+1], i*2+1)
                i += 1

        spec_1_cubes = {}
        for part in spec1["part"].keys():
            #TODO Parser problems
            bdd1, vars, mgr1 = formula_to_bdd(spec1["part"][part]["formula"], mgr, vars1, parser="ply")
            cube_set1 = []
            pycudd.set_iter_meth(0)
            for cube in bdd1:
                cube_set1.append(cube)
            spec_1_cubes[part] = cube_set1
        spec1["bdd_info"]["cubes"] = spec_1_cubes

    #Create list of indices for permutations
    # Ex: [[1,2,3],[4,5],[6]], will be permuted into e.g [[3,1,2],[5,4],[6]]
    i = 0
    index_list = []
    for value in sorted(spec0["values"]):
        index_list.append(range(i,i+spec0["values"][value]["nr"]))
        i += spec0["values"][value]["nr"]

    #Permute indices
    for permutation in combination_perturbations(index_list):
        print permutation
        vars0 = {}
        i = 0
        for value in sorted(spec0["values"]):
            for var in spec0["values"][value]["vars"]:
                idx = permutation[i]*2
                vars0[var] = (node_list[idx], idx)
                vars0["X "+var] = (node_list[idx+1], idx+1)
                i += 1

        for part in spec0["part"].keys():
            #This can be optimized by checking if this permutation of variables
            #have been checked for this part previously
            #TODO Parser problems
            bdd0, vars, mgr0 = formula_to_bdd(spec0["part"][part]["formula"], mgr, vars0, parser="ply")
            cube_set0 = []
            pycudd.set_iter_meth(0)
            for cube in bdd0:
                cube_set0.append(cube)
            #Compare this bdd for this spec and permutation
            print "Parts "+part+" equal: " + str(cube_set0 == spec_1_cubes[part])
            if cube_set0 == spec_1_cubes[part]:
                #Test the rest of the parts, or if last, go to "else return"
                continue
            else:
                #Test next permutation
                break
        else:
            return True


    return False