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
Implements a class for streaming graph changes to Gephi, which displays
the automaton as a graph.

Note that the Gephi Graph Streaming API plugin must be installed.
"""

import threading
import time
from subprocess import call
from urllib import *

from automaton import AutomatonState
from errorprint import printWarning

class GephiStream:
    """
    Class for reading, adding, changing, and deleting nodes and edges
    on a live Gephi graph workspace.

    Fields:
    url_in -- string representing the incoming URL from the Gephi server.
    url_out -- string representing the outgoing URL to the Gephi server.
    readThread -- asynchronous thread for printing graph changes to console.
    """
    
    def __init__(self, url='http://localhost:8080/workspace',
                 workspace=0, verbose=False):
        if not (isinstance(url, str) and isinstance(workspace, int) and
                isinstance(verbose, bool)):
            raise TypeError("Invalid arguments to GephiStream.")
        self.url_in = url + str(workspace) + '?operation=getGraph'
        self.url_out = url + str(workspace) + '?operation=updateGraph'

        self.readThread = False
        if verbose:
            self.readThread = threading.Thread(target=self.printGraph)
            self.readThread.daemon = True
            self.readThread.start()
    
    
    def __del__(self):
        if self.readThread:
            self.readThread.join()
    
    
    def printGraph(self):
        """
        Prints a list of all updates to the current graph.
        
        Arguments:
        (none)
        
        Return:
        (nothing)
        """
        
        urlrecv = urlopen(self.url_in)
        try:
            msg = True
            while msg:
                msg = urlrecv.readline()
                print "recv: " + msg
            urlrecv.close()
        except IOError:
            urlrecv.close()
    
    
    def wrapJSON(self, command, target, attribute_dict={}):
        """
        Encode the graph change command in JavaScript Object Notation.
        
        Arguments:
        command -- a string representing the command to be given. Choose from:
            "an": Add node.
            "cn": Change node attribute.
            "dn": Delete node.
            "ae": Add edge.
            "ce": Change edge attribute.
            "de": Delete edge.
        target -- a string representing the ID of the target node or edge.
        attribute_dict -- a dictionary of attributes to be changed. The keys
            are attribute names and values are attribute values.
        
        Return:
        String representing a JSON graph change command.
        """
        if not (command in ["an", "cn", "dn", "ae", "ce", "de"] and
                isinstance(target, str) and isinstance(attribute_dict, dict)):
            raise TypeError("Invalid arguments to wrapJSON")
        
        output = '{"' + command + '":{"' + target + '":{'
        if attribute_dict != {}:
            for (attribute, value) in attribute_dict.items():
                if not isinstance(attribute, str):
                    raise TypeError("Invalid arguments to wrapJSON")
                
                output += '"' + attribute + '":'
                if isinstance(value, str):
                    output += '"' + value + '"'
                else:
                    output += str(value)
                output += ','
            # Delete the extra comma at the end.
            output = output[:-1]
        output += '}}}'
        
        return output
    

    def addNode(self, w_id, state):
        """
        Add a node to the current graph.

        Arguments:
        w_id -- an integer representing this node's Automaton/W set ID.
        state -- an AutomatonState object.

        Return:
        (nothing)
        """
        
        if not (isinstance(w_id, int) and
                isinstance(state, AutomatonState)):
            raise TypeError("Invalid arguments to addNode")
        
        node_ID = str(w_id) + '.' + str(state.id)
        attribute_dict = state.state.copy()
        
        msg = self.wrapJSON("an", node_ID, attribute_dict)
        print "send: " + msg
        urlsend = urlopen(self.url_out, msg)
        urlsend.close()
    
    
    def changeNode(self, w_id, state, change_dict):
        """
        Change a node of the current graph.

        Arguments:
        w_id -- an integer representing this node's Automaton/W set ID.
        state -- an AutomatonState object.
        change_dict -- a dictionary of attributes to change.

        Return:
        (nothing)
        """
        
        if not (isinstance(w_id, int) and
                isinstance(state, AutomatonState) and
                isinstance(change_dict, dict)):
            raise TypeError("Invalid arguments to changeNode")
        
        node_ID = str(w_id) + '.' + str(state.id)
        
        msg = self.wrapJSON("cn", node_ID, change_dict)
        print "send: " + msg
        urlsend = urlopen(self.url_out, msg)
        urlsend.close()
    
    
    def deleteNode(self, w_id, state):
        """
        Delete a node of the current graph.

        Arguments:
        w_id -- an integer representing this node's Automaton/W set ID.
        state -- an AutomatonState object.

        Return:
        (nothing)
        """
        
        if not (isinstance(w_id, int) and
                isinstance(state, AutomatonState)):
            raise TypeError("Invalid arguments to deleteNode")
        
        node_ID = str(w_id) + '.' + str(state.id)
        
        msg = self.wrapJSON("dn", node_ID, {})
        print "send: " + msg
        urlsend = urlopen(self.url_out, msg)
        urlsend.close()
    
    
    def addEdge(self, sourcew_id, source, targetw_id, target):
        """
        Add an edge to the current graph.

        Arguments:
        sourcew_id -- an integer representing the source's Automaton/W set ID.
        source -- an AutomatonState object.
        targetw_id -- an integer representing the target's Automaton/W set ID.
        target -- an AutomatonState object.

        Return:
        (nothing)
        """
        
        if not (isinstance(sourcew_id, int) and
                isinstance(source, AutomatonState) and
                isinstance(targetw_id, int) and
                isinstance(target, AutomatonState)):
            raise TypeError("Invalid arguments to addEdge")
        
        source_ID = str(sourcew_id) + '.' + str(source.id)
        target_ID = str(targetw_id) + '.' + str(target.id)
        edge_ID = source_ID + '-' + target_ID
        
        attribute_dict = target.state.copy()
        attribute_dict['source'] = source_ID
        attribute_dict['target'] = target_ID
        
        msg = self.wrapJSON("ae", edge_ID, attribute_dict)
        print "send: " + msg
        urlsend = urlopen(self.url_out, msg)
        urlsend.close()
    
    
    def changeEdge(self, sourcew_id, source, targetw_id, target, change_dict):
        """
        Change an edge of the current graph.

        Arguments:
        sourcew_id -- an integer representing the source's Automaton/W set ID.
        source -- an AutomatonState object.
        targetw_id -- an integer representing the target's Automaton/W set ID.
        target -- an AutomatonState object.
        change_dict -- a dictionary of attributes to change.

        Return:
        (nothing)
        """
        
        if not (isinstance(sourcew_id, int) and
                isinstance(source, AutomatonState) and
                isinstance(targetw_id, int) and
                isinstance(target, AutomatonState) and
                isinstance(change_dict, dict)):
            raise TypeError("Invalid arguments to changeEdge")
        
        source_ID = str(sourcew_id) + '.' + str(source.id)
        target_ID = str(targetw_id) + '.' + str(target.id)
        edge_ID = source_ID + '-' + target_ID
        
        msg = self.wrapJSON("ce", edge_ID, change_dict)
        print "send: " + msg
        urlsend = urlopen(self.url_out, msg)
        urlsend.close()
    
    
    def deleteEdge(self, sourcew_id, source, targetw_id, target):
        """
        Delete an edge from the current graph.
        
        Arguments:
        sourcew_id -- an integer representing the source's Automaton/W set ID.
        source -- an AutomatonState object.
        targetw_id -- an integer representing the target's Automaton/W set ID.
        target -- an AutomatonState object.

        Return:
        (nothing)
        """
        print "Warning: 'deleteEdge' does not work with directed graphs. " + \
              "It appears to delete an arbitrary edge" + \
              "connecting the two nodes."
        
        if not (isinstance(sourcew_id, int) and
                isinstance(source, AutomatonState) and
                isinstance(targetw_id, int) and
                isinstance(target, AutomatonState)):
            raise TypeError("Invalid arguments to deleteEdge")
        
        source_ID = str(sourcew_id) + '.' + str(source.id)
        target_ID = str(targetw_id) + '.' + str(target.id)
        edge_ID = source_ID + '-' + target_ID
        
        msg = self.wrapJSON("de", edge_ID, {})
        print "send: " + msg
        urlsend = urlopen(self.url_out, msg)
        urlsend.close()
    
    
    
if __name__ == "__main__":
    # Number of seconds to pause between tests
    delay = 2
    
    print "*** NOTE: Tests are still in development. When Gephi opens,\n" + \
          "press ctrl-shift-n, then go to the 'Streaming' tab,\n" + \
          "right-click on 'Master Server', and click 'Start'. ***\n\n" + \
          "You have " + str(10 * delay) + " seconds."
    
    print "Running tests..."
    # 'call' must be wrapped in a lambda function to avoid starting early.
    gephi_thread = threading.Thread(target=(lambda: call("gephi")))
    gephi_thread.start()
    time.sleep(10 * delay)
    
    print "Testing printGraph..."
    test_node = AutomatonState(id=1, state={'foo':1, 'bar':2})
    gs = GephiStream(verbose=True)
    time.sleep(delay)
    print "Testing wrapJSON..."
    assert gs.wrapJSON("an", "0." + str(test_node.id),
                       attribute_dict=test_node.state) == \
           '{"an":{"0.1":{"foo":1,"bar":2}}}'
    time.sleep(delay)
    
    print "Testing addNode..."
    gs.addNode(0, test_node)
    gs.addNode(1, test_node)
    gs.addNode(2, test_node)
    time.sleep(delay)
    print "Testing changeNode..."
    gs.changeNode(0, test_node, {'foo':3})
    gs.changeNode(1, test_node, {'bar':4})
    time.sleep(delay)
    print "Testing deleteNode..."
    gs.deleteNode(2, test_node)
    time.sleep(delay)
    
    print "Testing addEdge..."
    gs.addEdge(0, test_node, 1, test_node)
    gs.addEdge(1, test_node, 0, test_node)
    time.sleep(delay)
    print "Testing changeEdge..."
    gs.changeEdge(1, test_node, 0, test_node, {'foo':5})
    time.sleep(delay)
    print "Testing deleteEdge..."
    gs.deleteEdge(1, test_node, 0, test_node)
    time.sleep(delay)
    
    print "Close Gephi to conclude unit tests."
    gephi_thread.join()
    print "Tests done."
