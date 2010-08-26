#!/usr/bin/env python

""" 
--------------------
Specification Module
--------------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 25, 2010
:Version: 0.1.0
"""

import re, copy

class GRSpec:
    """
    GRSpec class for specifying a GR[1] specification of the form

        (env_init & []env_safety & []<>env_prog) -> (sys_init & []sys_safety & []<>sys_prog).

    A GRSpec object contains the following fields:

    - `env_init`: a string or a list of string that specifies the assumption about 
      the initial state of the environment.
    - `env_safety`: a string or a list of string that specifies the assumption about 
      the evolution of the environment state.
    - `env_prog`: a string or a list of string that specifies the justice assumption on 
      the environment.
    - `sys_init`: a string or a list of string that specifies the requirement on the 
      initial state of the system.
    - `sys_safety`: a string or a list of string that specifies the safety requirement.
    - `sys_prog`: a string or a list of string that specifies the progress requirement.
    """
    def __init__(self, env_init='', sys_init='', env_safety='', sys_safety='', \
                     env_prog='', sys_prog=''):
        self.env_init = copy.deepcopy(env_init)
        self.sys_init = copy.deepcopy(sys_init)
        self.env_safety = copy.deepcopy(env_safety)
        self.sys_safety = copy.deepcopy(sys_safety)
        self.env_prog = copy.deepcopy(env_prog)
        self.sys_prog = copy.deepcopy(sys_prog)
    
    def sym2prop(self, props, verbose=0):
        """Replace the symbols of propositions in this spec with the actual propositions"""
        # Replace symbols for propositions on discrete variables with the actual 
        # propositions
        if (props is not None):
            for propSymbol, prop in props.iteritems():
                if (verbose > 1):
                    print '\t' + propSymbol + ' -> ' + prop
                if (isinstance(self.env_init, str)):
                    self.env_init = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.env_init)
                else:
                    for i in xrange(0, len(self.env_init)):
                        self.env_init[i] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                        self.env_init[i])
                    
                if (isinstance(self.sys_init, str)):
                    self.sys_init = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.sys_init)
                else:
                    for i in xrange(0, len(self.sys_init)):
                        self.sys_init[i] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                        self.sys_init[i])

                if (isinstance(self.env_safety, str)):
                    self.env_safety = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.env_safety)
                else:
                    for i in xrange(0, len(self.env_safety)):
                        self.env_safety[i] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                        self.env_safety[i])

                if (isinstance(self.sys_safety, str)):
                    self.sys_safety = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.sys_safety)
                else:
                    for i in xrange(0, len(self.sys_safety)):
                        self.sys_safety[i] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                        self.sys_safety[i])
                
                if (isinstance(self.env_prog, str)):
                    self.env_prog = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.env_prog)
                else:
                    for i in xrange(0, len(self.env_prog)):
                        self.env_prog[i] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                      self.env_prog[i])

                if (isinstance(self.sys_prog, str)):
                    self.sys_prog = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.sys_prog)
                else:
                    for i in xrange(0, len(self.sys_prog)):
                        self.sys_prog[i] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                      self.sys_prog[i])

    def toJTLVSpec(self):
        """Return a list of two strings [assumption, guarantee] corresponding to this GR[1]
        specification."""
        spec = ['', '']
        # env init
        if (isinstance(self.env_init, str)):
            if (len(self.env_init) > 0 and not self.env_init.isspace()):
                spec[0] += '-- valid initial env states\n'
                spec[0] += '\t' + self.env_init
        else:
            desc_added = False
            for env_init in self.env_init:
                if (len(env_init) > 0):
                    if (len(spec[0]) > 0):
                        spec[0] += ' & \n'
                    if (not desc_added):
                        spec[0] += '-- valid initial env states\n'
                        desc_added = True
                    spec[0] += '\t' + env_init
        
        # env safety
        if (isinstance(self.env_safety, str)):
            if (len(self.env_safety) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & \n'
                spec[0] += '-- safety assumption on environment\n'
                spec[0] += '\t[](' + self.env_safety + ')'
        else:
            desc_added = False
            for env_safety in self.env_safety:
                if (len(env_safety) > 0):
                    if (len(spec[0]) > 0):
                        spec[0] += ' & \n'
                    if (not desc_added):
                        spec[0] += '-- safety assumption on environment\n'
                        desc_added = True
                    spec[0] += '\t[](' + env_safety + ')'

        # env progress
        if (isinstance(self.env_prog, str)):
            if (len(self.env_prog) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & \n'
                spec[0] += '-- justice assumption on environment\n'
                spec[0] += '\t[]<>(' + self.env_prog + ')'
        else:
            desc_added = False
            for prog in self.env_prog:
                if (len(prog) > 0):
                    if (len(spec[0]) > 0):
                        spec[0] += ' & \n'
                    if (not desc_added):
                        spec[0] += '-- justice assumption on environment\n'
                        desc_added = True
                    spec[0] += '\t[]<>(' + prog + ')'

        # sys init
        if (isinstance(self.sys_init, str)):
            if (len(self.sys_init) > 0 and not self.sys_init.isspace()):
                spec[1] += '-- valid initial system states\n'
                spec[1] += '\t' + self.sys_init
        else:
            desc_added = False
            for sys_init in self.sys_init:
                if (len(sys_init) > 0):
                    if (len(spec[1]) > 0):
                        spec[1] += ' & \n'
                    if (not desc_added):
                        spec[1] += '-- valid initial system states\n'
                        desc_added = True
                    spec[1] += '\t' + sys_init

        # sys safety
        if (isinstance(self.sys_safety, str)):
            if (len(self.sys_safety) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' & '
                spec[1] += '-- safety requirement on system\n'
                spec[1] +='\t[](' + self.sys_safety + ')'
        else:
            desc_added = False
            for sys_safety in self.sys_safety:
                if (len(sys_safety) > 0):
                    if (len(spec[1]) > 0):
                        spec[1] += ' & \n'
                    if (not desc_added):
                        spec[1] += '-- safety requirement on system\n'
                        desc_added = True
                    spec[1] += '\t[](' + sys_safety + ')'

        # sys progress
        if (isinstance(self.sys_prog, str)):
            if (len(self.sys_prog) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' & \n'
                spec[1] += '-- progress requirement on system\n'
                spec[1] += '\t[]<>(' + self.sys_prog + ')'
        else:
            desc_added = False
            for prog in self.sys_prog:
                if (len(prog) > 0):
                    if (len(spec[1]) > 0):
                        spec[1] += ' & \n'
                    if (not desc_added):
                        spec[1] += '-- progress requirement on system\n'
                        desc_added = True
                    spec[1] += '\t[]<>(' + prog + ')'
        return spec
