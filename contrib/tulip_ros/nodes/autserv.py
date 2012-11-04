#!/usr/bin/env python
"""
SCL; 1 August 2012.
"""

import roslib; roslib.load_manifest("tulip_ros")
from tulip_ros.srv import *
import rospy
import numpy as np
import xml.etree.ElementTree as ET
import tulip.conxml as cx
from tulip.automaton import Automaton
import sys


def loadXMLaut(x, namespace=cx.DEFAULT_NAMESPACE):
    """Return (environment_variables, system_variables, Automaton)."""
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

    aut_elem = elem.find(ns_prefix+"aut")
    if aut_elem is None \
            or ((aut_elem.text is None) and len(aut_elem.getchildren()) == 0):
        aut = None
    else:
        aut = Automaton()
        if not aut.loadXML(aut_elem, namespace=cx.DEFAULT_NAMESPACE):
            ep.printError("failed to read Automaton from given tulipcon XML string.")
            aut = None
    (tag_name, env_vars) = cx.untagdict(elem.find(ns_prefix+"env_vars"))
    (tag_name, sys_disc_vars) = cx.untagdict(elem.find(ns_prefix+"sys_vars"))
    return (env_vars.keys(), sys_disc_vars.keys(), aut)


class AutServer:
    def __init__(self, env_vars, sys_vars, aut):
        # In the case of "raw" (bitvector) messages, the order of
        # elements in a state vector are assumed to match that of
        # env_vars followed by sys_vars.
        self.env_vars = env_vars[:]
        self.sys_vars = sys_vars[:]
        self.aut = aut.copy()
        if len(self.aut) > 0:
            self.current_node = self.aut.states[0]
        else:
            self.current_node = None

    def query(self, req):
        pvars = self.env_vars[:]
        pvars.extend(self.sys_vars)
        if self.current_node is not None:
            return QueryResponse(variables=pvars,
                                 state=[self.current_node.state[v] for v in pvars])
        else:
            return QueryResponse()

    def initialize(self, req):
        self.current_node = self.aut.findAutState(dict(zip(req.variables, list(req.state))))
        if self.current_node == -1:
            self.current_node = None
            return InitializeResponse(False)
        else:
            return InitializeResponse(True)

    def step(self, req):
        next_node = self.aut.findNextAutState(self.current_node, env_state=dict(zip(req.env_variables, list(req.env_move))))
        if next_node == -1:
            return StepResponse()
        else:
            self.current_node = next_node
            return StepResponse(sys_variables=self.sys_vars,
                                sys_move=[self.current_node.state[s] for s in self.sys_vars])

    def raw_query(self, req):
        if self.current_node is not None:
            pvars = self.env_vars[:]
            pvars.extend(self.sys_vars)
            return RawQueryResponse([self.current_node.state[v] for v in pvars])
        else:
            return RawQueryResponse([])

    def raw_initialize(self, req):
        pvars = self.env_vars[:]
        pvars.extend(self.sys_vars)
        self.current_node = self.aut.findAutState(dict([(pvars[i], req.state[i]) for i in range(len(req.state))]))
        if self.current_node == -1:
            self.current_node = None
            return RawInitializeResponse(False)
        else:
            return RawInitializeResponse(True)

    def raw_step(self, req):
        next_node = self.aut.findNextAutState(self.current_node, env_state=dict([(self.env_vars[i], req.env_move[i]) for i in range(len(req.env_move))]))
        if next_node == -1:
            return RawStepResponse([])
        else:
            self.current_node = next_node
            return RawStepResponse([self.current_node.state[s] for s in self.sys_vars])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: autserv.py FILE"
        exit(1)

    with open(sys.argv[1], "r") as f:
        (env_vars, sys_vars, aut) = loadXMLaut(f.read())
    if aut is None:
        print "ERROR: no automaton found in given tulipcon XML file."
        exit(-1)
    serv = AutServer(env_vars=env_vars, sys_vars=sys_vars, aut=aut)

    rospy.init_node("autserv")
    #s_rquery = rospy.Service(rospy.get_name()+"_rawquery", RawQuery, serv.raw_query)
    #s_rinit = rospy.Service(rospy.get_name()+"_rawinit", RawInitialize, serv.raw_initialize)
    #s_rstep = rospy.Service(rospy.get_name()+"_rawstep", RawStep, serv.raw_step)
    s_query = rospy.Service(rospy.get_name()+"_query", Query, serv.query)
    s_init = rospy.Service(rospy.get_name()+"_init", Initialize, serv.initialize)
    s_step = rospy.Service(rospy.get_name()+"_step", Step, serv.step)
    rospy.spin()
