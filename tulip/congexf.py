#! /usr/bin/env python
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


"""
Convert an AUT file (from JTLV) to a GEXF (in Gephi) file.

Accepts a *.aut filename and, optionally, a destination filename. If no
destination is given, default is to use input filename with new ending ".gexf"

Also note that command-line arguments are handled manually, since they
are so simple...

Originally written by Scott Livingston for GraphViz DOT files.

Modified by Yuchen Lin for Gephi GEXF files.

"""

import sys
from subprocess import call
import re

from tulip.automaton import AutomatonState, Automaton
from tulip.errorprint import printWarning, printError


def tagGexfAttr(var_name, var_type, idt_lvl=3):
    """
    Encode the variable specified in a gexf-readable attribute.

    Arguments:
    var_name -- the string name of the desired variable.
    var_type -- Python type of the variable.
    idt_lvl -- level of indentation of the attribute. Default is 3,
        as gexf attributes are usually three levels in.

    Return:
    String representing a gexf attribute.
    """

    if not (isinstance(var_name, str) and isinstance(var_type, type) and
            isinstance(idt_lvl, int)):
        raise TypeError("Invalid arguments to tagGexfAttr")
    
    nl = "\n"  # newline
    idt = "  "  # indentation
    
    # First create a dictionary that translates Python data type names
    # to gexf data type names.
    type_dict = {str: 'string', int: 'integer',
                 float: 'float', bool: 'boolean'}

    # Generate a line of XML for the attribute.
    attribute = idt * idt_lvl + '<attribute id="' + var_name + \
                '" type="' + type_dict[var_type] + '" />' + nl
    
    return attribute
    
    
def tagGexfAttvalue(var_name, var_val, idt_lvl=5):
    """
    Encode the attribute specified in a gexf-readable attribute value.

    Arguments:
    var_name -- the string name of the attribute.
    var_val -- value of the attribute for this node/edge.
    idt_lvl -- level of indentation of the attribute value. Default is 5,
        as gexf attributes are usually five levels in.
    
    Return:
    String representing a gexf attribute value.
    """

    if not (isinstance(var_name, str) and isinstance(idt_lvl, int)):
        raise TypeError("Invalid arguments to tagGexfAttrVal")

    nl = "\n"  # newline
    idt = "  "  # indentation

    # Generate a line of XML for the attribute value.
    attvalue = idt * idt_lvl + '<attvalue for="' + var_name + \
               '" value="' + str(var_val) + '" />' + nl
    
    return attvalue
    
    
def tagGexfNode(state, label, supernode_id=0, idt_lvl=3):
    """
    Encode the state specified in a gexf-readable node. The variables
    in the specified state are stored as attribute values.

    Arguments:
    state -- the AutomatonState object to be encoded.
    label -- a string name for the node.
    supernode_id -- the ID of the supernode (Automaton or W set)
    idt_lvl -- level of indentation of the node. Default is 3,
        as gexf nodes are usually three levels in.

    Return:
    String representing a gexf node.
    """
    
    if not (isinstance(state, AutomatonState) and
            isinstance(label, str) and
            isinstance(supernode_id, int) and isinstance(idt_lvl, int)):
        raise TypeError("Invalid arguments to tagGexfNode")
    
    state_ID = str(supernode_id) + '.' + str(state.id)
    
    nl = "\n"  # newline
    idt = "  "  # indentation

    # Generate a line of XML for the node.
    node = idt * idt_lvl + '<node id="' + state_ID + \
           '" label="' + label + '">' + nl
    idt_lvl += 1
    node += idt * idt_lvl + '<attvalues>' + nl

    # Iterate through attributes.
    idt_lvl += 1
    attr_dict = state.state.copy()
    # is_active is a node attribute used to simulate the automaton.
    # It changes to 1 when the automaton is currently at that node.
    attr_dict['is_active'] = 0
    for (k, v) in attr_dict.items():
        node += tagGexfAttvalue(k, v, idt_lvl=idt_lvl)
    idt_lvl -= 1

    # Close attributes and node.
    node += idt * idt_lvl + '</attvalues>' + nl
    idt_lvl -= 1
    node += idt * idt_lvl + '</node>' + nl
    
    return node
        
    
def tagGexfEdge(source, target, edge_id, label, supernode_id=0, idt_lvl=3):
    """
    Encode the transition specified in a gexf-readable edge. The variables
    in the target's state are stored as attribute values.

    Arguments:
    source -- the AutomatonState for the 'tail' of the edge.
    target -- the AutomatonState for the 'head' of the edge.
    edge_id -- the ID number of the desired edge.
    label -- a string name for the edge.
    supernode_id -- the ID of an optional supernode (Automaton or W set)
    idt_lvl -- level of indentation of the edge. Default is 3,
        as gexf edges are usually three levels in.

    Return:
    String representing a gexf edge.
    """
    
    if not (isinstance(source, AutomatonState) and
            isinstance(target, AutomatonState) and
            isinstance(edge_id, int) and isinstance(label, str) and
            isinstance(supernode_id, int) and isinstance(idt_lvl, int)):
        raise TypeError("Invalid arguments to tagGexfEdge")
    
    source_ID = str(supernode_id) + '.' + str(source.id)
    target_ID = str(supernode_id) + '.' + str(target.id)
    
    nl = "\n"  # newline
    idt = "  "  # indentation
    
    # Generate a line of XML for the edge.
    edge = idt * idt_lvl + '<edge id="' + str(edge_id) + \
           '" source="' + source_ID + \
           '" target="' + target_ID + \
           '" label="' + label + '">' + nl
    idt_lvl += 1
    edge += idt * idt_lvl + '<attvalues>' + nl

    # Iterate through attributes.
    idt_lvl += 1
    attr_dict = target.state.copy()
    # is_active is an edge attribute used to simulate the automaton.
    # It changes to 1 when the edge is a valid transition.
    attr_dict['is_active'] = 0
    for (k, v) in attr_dict.items():
        edge += tagGexfAttvalue(k, v, idt_lvl=idt_lvl)
    idt_lvl -= 1
    
    # Close attributes and node.
    edge += idt * idt_lvl + '</attvalues>' + nl
    idt_lvl -= 1
    edge += idt * idt_lvl + '</edge>' + nl
    
    return edge

    
def dumpGexf(aut_list):
    """
    Writes the automaton to a Gephi 'gexf' string. Nodes represent system
    states and edges represent transitions between nodes.

    Arguments:
    aut_list -- a list of Automaton objects. Generate a hierarchical graph
                with each Automaton as a supernode. Note: if a single
                Automaton is given instead, will wrap in a list.

    Return:
    A gexf formatted string that can be written to file.
    """

    if isinstance(aut_list, Automaton):
        # Wrap aut_or_list in a list, for simpler code later.
        aut_list = [aut_list]
    elif isinstance(aut_list, list):
        for aut in aut_list:
            if not isinstance(aut, Automaton):
                raise TypeError("Invalid arguments to dumpGexf.")
    else:
        raise TypeError("Invalid arguments to dumpGexf.")
    
    nl = "\n"  # newline
    idt = "  "  # indentation
    idt_lvl = 0 # indentation level
    output = '' # string to be written to file
    
    # Open xml, gexf, and node attributes tags.
    output += idt * idt_lvl + '<?xml version="1.0" encoding="UTF-8"?>' + nl
    output += idt * idt_lvl + '<gexf version="1.2">' + nl
    idt_lvl += 1
    output += idt * idt_lvl + '<graph defaultedgetype="directed">' + nl
    idt_lvl += 1
    output += idt * idt_lvl + '<attributes class="node">' + nl

    # Build gexf node attributes (used to specify the types of data
    # that can be stored) from the 'state' dictionary of
    # AutomatonState states.
    idt_lvl += 1
    # is_active is a node attribute used to simulate the automaton.
    # It changes to True when the automaton is currently at that node.
    output += tagGexfAttr('is_active', int, idt_lvl=idt_lvl)
    node_attr_list = ['is_active']
    for aut in aut_list:
        for state in aut.states:
            for (k, v) in state.state.items():
                if k not in node_attr_list:
                    output += tagGexfAttr(k, type(v), idt_lvl=idt_lvl)
                    node_attr_list.append(k)
    idt_lvl -= 1
    
    # Close node attributes tag and open edge attributes tag.
    output += idt * idt_lvl + '</attributes>' + nl
    output += idt * idt_lvl + '<attributes class="edge">' + nl

    # Build gexf edge attributes (used to specify the types of data
    # that can be stored) from the 'state' dictionary of
    # AutomatonState states.
    idt_lvl += 1
    # is_active is an edge attribute used to simulate the automaton.
    # It changes to True when the edge is a valid transition.
    output += tagGexfAttr('is_active', int, idt_lvl=idt_lvl)
    edge_attr_list = ['is_active']
    for aut in aut_list:
        for state in aut.states:
            for (k, v) in state.state.items():
                if k not in edge_attr_list:
                    output += tagGexfAttr(k, type(v), idt_lvl=idt_lvl)
                    edge_attr_list.append(k)
    idt_lvl -= 1
    
    # Close edge attributes tag and open nodes tag.
    output += idt * idt_lvl + '</attributes>' + nl
    output += idt * idt_lvl + '<nodes>' + nl
    
    # Build hierarchical gexf nodes.
    idt_lvl += 1
    aut_id = 0  # Each automaton is numbered, starting from 0.
    for aut in aut_list:
        # Open supernode (a single Automaton) and subnodes.
        output += idt * idt_lvl + '<node id="' + str(aut_id) + \
                  '" label="W' + str(aut_id) + '">' + nl
        idt_lvl += 1
        output += idt * idt_lvl + '<nodes>' + nl

        # Build gexf nodes from AutomatonState states.
        idt_lvl += 1
        for state in aut.states:
            output += tagGexfNode(state, str(state.state.values()),
                                  supernode_id=aut_id, idt_lvl=idt_lvl)
        idt_lvl -= 1
        
        # Close subnodes and supernode.
        output += idt * idt_lvl + '</nodes>' + nl
        idt_lvl -= 1
        output += idt * idt_lvl + '</node>' + nl
        
        aut_id += 1
    idt_lvl -= 1
    
    # Close nodes tag and open edges tag.
    output += idt * idt_lvl + '</nodes>' + nl
    output += idt * idt_lvl + '<edges>' + nl
    
    # Build hierarchical gexf edges.
    idt_lvl += 1
    aut_id = 0
    edge_id = 0  # Each edge is numbered, starting from 0.
    for aut in aut_list:        
        for state in aut.states:
            for trans in state.transition:
                # Locate target node.
                for aut_state in aut.states:
                    if trans == aut_state.id:
                        target = aut_state
                        # target has been found, leave for loop.
                        break
                output += tagGexfEdge(state, target, edge_id,
                                      str(target.state.values()),
                                      supernode_id=aut_id, idt_lvl=idt_lvl)
                edge_id += 1
        aut_id += 1
    idt_lvl -= 1

    # Close edges, graph, and gexf tags.
    output += idt * idt_lvl + '</edges>' + nl
    idt_lvl -= 1
    output += idt * idt_lvl + '</graph>' + nl
    idt_lvl -= 1
    output += idt * idt_lvl + '</gexf>' + nl

    assert idt_lvl == 0
    
    return output


def changeGexfAttvalue(gexf_string, att_name, att_val,
                       node_id=None, edge_id=None):
    """
    Change an attribute for the given node or edge (or both).

    Arguments:
    gexf_string -- a gexf-formatted string (writable to file).
    att_name -- the string name of the attribute to be changed.
    att_val -- the new value of the attribute to be changed.
    node_id -- optional id for the node to be changed.
    edge_id -- optional id for the edge to be changed.

    Return:
    A changed gexf output string. If changes are not possible, returns
    original string.
    """
    if not (isinstance(gexf_string, str) and isinstance(att_name, str)):
        raise TypeError("Invalid arguments to changeGexfAttvalue.")
    
    # Note that 'str.find' returns -1 if pattern isn't found.

    if node_id != None:
        # Search for desired node and set bounds to only this node.
        start = gexf_string.find('<node id="' + node_id)
        end = start + gexf_string[start:].find('</attvalues>')
        
        # Search for desired attvalue and set bounds to only this attvalue.
        start = start + gexf_string[start:end].find('<attvalue for="' + att_name)
        end = start + gexf_string[start:end].find('>') + 1
        
        # Check if attvalue has been found.
        if start != -1:
            gexf_string = gexf_string[:start] + \
                          re.sub(r'value=".*"', 'value="' + str(att_val) + '"',
                                 gexf_string[start:end])+ \
                          gexf_string[end:]
    
    if edge_id != None:
        # Search for desired edge and set bounds to only this edge.
        start = gexf_string.find('<edge id="' + edge_id)
        end = start + gexf_string[start:].find('</attvalues>')
        
        # Search for desired attvalue and set bounds to only this attvalue.
        start = start + gexf_string[start:end].find('<attvalue for="' + att_name)
        end = start + gexf_string[start:end].find('>') + 1
        
        # Check if attvalue has been found.
        if start != -1:
            gexf_string = gexf_string[:start] + \
                          re.sub(r'value=".*"', 'value="' + str(att_val) + '"',
                                 gexf_string[start:end])+ \
                          gexf_string[end:]
    
    return gexf_string



if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "Usage: aut_to_gexf.py INPUT.aut [OUTPUT.gexf]"
        exit(1)

    if len(sys.argv) == 3:
        destfile = sys.argv[2]
    else:
        if len(sys.argv[1]) < 4:  # Handle weirdly short names
            destfile = sys.argv[1] + ".gexf"
        else:
            destfile = sys.argv[1][:-4] + ".gexf"
    
    testAutState0 = AutomatonState(id=0, state={'a':0, 'b':1}, transition=[1])
    testAutState1 = AutomatonState(id=1, state={'a':2, 'b':3}, transition=[0])
    testAut = Automaton([testAutState0, testAutState1])
    print "Testing tagGexfAttr..."
    assert tagGexfAttr('a', int, idt_lvl=3) == \
           '      <attribute id="a" type="integer" />\n'   
    print "Testing tagGexfAttvalue..."
    assert tagGexfAttvalue('a', 0, idt_lvl=5) == \
           '          <attvalue for="a" value="0" />\n'
    print "Testing tagGexfNode..."
    assert tagGexfNode(testAutState0, 'foo', supernode_id=0, idt_lvl=3) == \
           '      <node id="0.0" label="foo">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="0" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </node>\n'
    print "Testing tagGexfEdge..."
    assert tagGexfEdge(testAutState0, testAutState1,
                               0, 'bar', idt_lvl=3) == \
           '      <edge id="0" source="0.0" target="0.1" label="bar">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n'
    print "Testing dumpGexf..."
    assert dumpGexf(testAut) == \
           '<?xml version="1.0" encoding="UTF-8"?>\n' + \
           '<gexf version="1.2">\n' + \
           '  <graph defaultedgetype="directed">\n' + \
           '    <attributes class="node">\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <attributes class="edge">\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <nodes>\n' + \
           '      <node id="0" label="W0">\n' + \
           '        <nodes>\n' + \
           '          <node id="0.0" label="[0, 1]">\n' + \
           '            <attvalues>\n' + \
           '              <attvalue for="a" value="0" />\n' + \
           '              <attvalue for="b" value="1" />\n' + \
           '              <attvalue for="is_active" value="0" />\n' + \
           '            </attvalues>\n' + \
           '          </node>\n' + \
           '          <node id="0.1" label="[2, 3]">\n' + \
           '            <attvalues>\n' + \
           '              <attvalue for="a" value="2" />\n' + \
           '              <attvalue for="b" value="3" />\n' + \
           '              <attvalue for="is_active" value="0" />\n' + \
           '            </attvalues>\n' + \
           '          </node>\n' + \
           '        </nodes>\n' + \
           '      </node>\n' + \
           '    </nodes>\n' + \
           '    <edges>\n' + \
           '      <edge id="0" source="0.0" target="0.1" label="[2, 3]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '      <edge id="1" source="0.1" target="0.0" label="[0, 1]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="0" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '    </edges>\n' + \
           '  </graph>\n' + \
           '</gexf>\n'
    print "Testing changeGexfAttvalue..."
    assert changeGexfAttvalue(dumpGexf(testAut), "a", 5,
                             node_id="0.0", edge_id="1") == \
           '<?xml version="1.0" encoding="UTF-8"?>\n' + \
           '<gexf version="1.2">\n' + \
           '  <graph defaultedgetype="directed">\n' + \
           '    <attributes class="node">\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <attributes class="edge">\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <nodes>\n' + \
           '      <node id="0" label="W0">\n' + \
           '        <nodes>\n' + \
           '          <node id="0.0" label="[0, 1]">\n' + \
           '            <attvalues>\n' + \
           '              <attvalue for="a" value="5" />\n' + \
           '              <attvalue for="b" value="1" />\n' + \
           '              <attvalue for="is_active" value="0" />\n' + \
           '            </attvalues>\n' + \
           '          </node>\n' + \
           '          <node id="0.1" label="[2, 3]">\n' + \
           '            <attvalues>\n' + \
           '              <attvalue for="a" value="2" />\n' + \
           '              <attvalue for="b" value="3" />\n' + \
           '              <attvalue for="is_active" value="0" />\n' + \
           '            </attvalues>\n' + \
           '          </node>\n' + \
           '        </nodes>\n' + \
           '      </node>\n' + \
           '    </nodes>\n' + \
           '    <edges>\n' + \
           '      <edge id="0" source="0.0" target="0.1" label="[2, 3]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '      <edge id="1" source="0.1" target="0.0" label="[0, 1]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="5" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '    </edges>\n' + \
           '  </graph>\n' + \
           '</gexf>\n'
    print "Tests passed."



    
    aut = Automaton(sys.argv[1])
    print "Generating GEXF file, " + destfile
    try:
        f = open(destfile, "w")
        f.write(dumpGexf([aut]))
        f.close()
    except IOError:
        f.close()
        print "Failed to create GEXF file, " + destfile
        exit(-1)

    if raw_input("Do you want to open in Gephi? (y/n)") == 'y':
        try:
            print "Opening GEXF file in Gephi."
            call(["gephi", destfile])
        except:
            print "Failed to open " + destfile + " in Gephi. Try:\n\n" + \
                  "gephi " + destfile + "\n\n"
