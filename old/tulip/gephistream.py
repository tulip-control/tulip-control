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

import time
from multiprocessing import Process, Manager
from subprocess import call
from urllib import urlopen
from SocketServer import ThreadingMixIn, TCPServer
from BaseHTTPServer import BaseHTTPRequestHandler

from automaton import AutomatonState

class GraphHandler(BaseHTTPRequestHandler):
    """
    Class for handling HTTP graph requests.
    """
    
    def do_GET(self):
        """
        Accept only graph requests, then stream commands.

        Arguments:
        (none)

        Return:
        (nothing)
        """
        try:
            if not self.path.endswith("?operation=getGraph"): raise IOError
            # Accept GET request and return a success code.
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Send messages until the empty string is passed in 'msg_list'.
            # The server must have a multithreaded list named 'msg_list'.
            msg_list = self.server.msg_list
            index = 0
            msg = True
            while msg:
                # Send only messages that have not yet been sent.
                for msg in msg_list[index:]:
                    print 'send-to %s: %s' % (str(self.client_address), msg)
                    self.wfile.write(msg)
                    index += 1
        except IOError:
            self.send_error(404, "File not found: " + self.path)
    

class GephiStream:
    """
    Class for adding, changing, and deleting nodes and edges
    on a live Gephi graph workspace.

    Note: The 'close()' function should be called when streaming is complete.
    
    Arguments:
    mode -- 'server' or 'client'? 'mode' determines whether each instance
        is a client or a server. In client mode, connect to an existing
        Gephi server. [The Gephi streaming server is still unstable and
        in development.] In server mode, start a graph streaming server.
    host -- name of the server host. In 'client' mode, connect to 'host'
        and try to send data there. In 'server' mode, set up an HTTP server
        at 'host' and stream data to any clients that might try to connect
        there. Default is 'localhost'.
    port -- name of the server port. In 'client' mode, connect to 'port'
        and try to send data there. In 'server' mode, set up an HTTP server
        at 'port' and stream data to any clients that might try to connect
        there. Default is 8080.
    workspace -- an int representing the Gephi workspace of this stream.
        Default is 0.

    Fields:
    mode -- see above.
    host -- see above.
    port -- see above.
    url -- the server's streaming URL.
    msg_list -- a list of all commands being streamed in 'server' mode.
    """
    
    def __init__(self, mode, host="localhost", port=8080, workspace=0):
        if not ((mode == 'client' or mode == 'server') and
                isinstance(host, str) and isinstance(port, int) and
                isinstance(workspace, int)):
            raise TypeError("Invalid arguments to GephiStream.")
        self.mode = mode
        self.host = host
        self.port = port
        self.url = 'http://%s:%d/workspace%d?operation=getGraph' % \
                   (host, port, workspace)
        self.msg_list = Manager().list()
        stream_thread = Process(target=self.stream)
        stream_thread.daemon = True
        stream_thread.start()
    
    
    def close(self):
        """
        Close the server or client and stop streaming.
        
        Arguments:
        (none)
        
        Return:
        (nothing)
        """
        # Sending an empty string should terminate all graph streams.
        self.send('')
        # Deleting self here forces the stream daemon thread to stop.
        del self
    
    
    def stream(self):
        """
        Streams the graph to/from Gephi.
        
        Arguments:
        (none)
        
        Return:
        (nothing)
        """
        print 'Graph streaming from:\n' + self.url + '\n'
        if self.mode == 'client':
            # Receive streamed graph changes from the server.
            urlrecv = urlopen(self.url)
            try:
                msg = True
                while msg:
                    msg = urlrecv.readline()
                    print "recv: " + msg
                urlrecv.close()
            except IOError:
                urlrecv.close()
        
        elif self.mode == 'server':
            # Start a TCP server to stream graph changes.
            # Server must be closed by deleting each instance.
            class newTCPServer(ThreadingMixIn, TCPServer):
                msg_list = self.msg_list
            server = newTCPServer((self.host, self.port), GraphHandler)
            server.serve_forever()
    
    
    def send(self, msg):
        """
        Send the message to the current stream.

        Arguments:
        msg -- a string message to be sent.

        Return:
        (nothing)
        """
        print "send: " + msg
        if self.mode == 'client':
            urlsend = urlopen(self.url.replace('getGraph', 'updateGraph'), msg)
            urlsend.close()
            
        elif self.mode == 'server':
            self.msg_list.append(msg)
    
    
    def wrapJSON(self, command, target, attribute_dict={}):
        """
        Encode the graph change command in JavaScript Object Notation.
        
        Arguments:
        command -- a string representing the command to be given. Choose from:
            "an": Add node.
            "cn": Change node attributes.
            "dn": Delete node.
            "ae": Add edge.
            "ce": Change edge attributes.
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
        output += '}}}\r\n'
        
        return output
    

    def addNode(self, p_id, state):
        """
        Add a node to the current graph.

        Arguments:
        p_id -- an integer representing this node's parent Automaton.
        state -- an AutomatonState object.

        Return:
        (nothing)
        """
        
        if not (isinstance(p_id, int) and
                isinstance(state, AutomatonState)):
            raise TypeError("Invalid arguments to addNode")
        
        node_ID = str(p_id) + '.' + str(state.id)
        attribute_dict = state.state.copy()
        
        self.send(self.wrapJSON("an", node_ID, attribute_dict))
    
    
    def changeNode(self, p_id, state, change_dict):
        """
        Change a node of the current graph.

        Arguments:
        p_id -- an integer representing this node's parent Automaton.
        state -- an AutomatonState object.
        change_dict -- a dictionary of attributes to change.

        Return:
        (nothing)
        """
        
        if not (isinstance(p_id, int) and
                isinstance(state, AutomatonState) and
                isinstance(change_dict, dict)):
            raise TypeError("Invalid arguments to changeNode")
        
        node_ID = str(p_id) + '.' + str(state.id)
        
        self.send(self.wrapJSON("cn", node_ID, change_dict))
    
    
    def deleteNode(self, p_id, state):
        """
        Delete a node of the current graph.

        Arguments:
        p_id -- an integer representing this node's parent Automaton.
        state -- an AutomatonState object.

        Return:
        (nothing)
        """
        
        if not (isinstance(p_id, int) and
                isinstance(state, AutomatonState)):
            raise TypeError("Invalid arguments to deleteNode")
        
        node_ID = str(p_id) + '.' + str(state.id)
        
        self.send(self.wrapJSON("dn", node_ID, {}))
    
    
    def addEdge(self, sourcep_id, source, targetp_id, target):
        """
        Add an edge to the current graph.

        Arguments:
        sourcep_id -- an integer representing the source's parent Automaton.
        source -- an AutomatonState object.
        targetp_id -- an integer representing the target's parent Automaton.
        target -- an AutomatonState object.

        Return:
        (nothing)
        """
        
        if not (isinstance(sourcep_id, int) and
                isinstance(source, AutomatonState) and
                isinstance(targetp_id, int) and
                isinstance(target, AutomatonState)):
            raise TypeError("Invalid arguments to addEdge")
        
        source_ID = str(sourcep_id) + '.' + str(source.id)
        target_ID = str(targetp_id) + '.' + str(target.id)
        edge_ID = source_ID + '-' + target_ID
        
        attribute_dict = target.state.copy()
        attribute_dict['source'] = source_ID
        attribute_dict['target'] = target_ID
        
        self.send(self.wrapJSON("ae", edge_ID, attribute_dict))
    
    
    def changeEdge(self, sourcep_id, source, targetp_id, target, change_dict):
        """
        Change an edge of the current graph.

        Arguments:
        sourcep_id -- an integer representing the source's parent Automaton.
        source -- an AutomatonState object.
        targetp_id -- an integer representing the target's parent Automaton.
        target -- an AutomatonState object.
        change_dict -- a dictionary of attributes to change.

        Return:
        (nothing)
        """
        
        if not (isinstance(sourcep_id, int) and
                isinstance(source, AutomatonState) and
                isinstance(targetp_id, int) and
                isinstance(target, AutomatonState) and
                isinstance(change_dict, dict)):
            raise TypeError("Invalid arguments to changeEdge")
        
        source_ID = str(sourcep_id) + '.' + str(source.id)
        target_ID = str(targetp_id) + '.' + str(target.id)
        edge_ID = source_ID + '-' + target_ID
        
        self.send(self.wrapJSON("ce", edge_ID, change_dict))
    
    
    def deleteEdge(self, sourcep_id, source, targetp_id, target):
        """
        Delete an edge from the current graph.
        
        Arguments:
        sourcep_id -- an integer representing the source's parent Automaton.
        source -- an AutomatonState object.
        targetp_id -- an integer representing the target's parent Automaton.
        target -- an AutomatonState object.

        Return:
        (nothing)
        """
        print "Warning: 'deleteEdge' does not work with directed graphs. " + \
              "It appears to delete an arbitrary edge " + \
              "connecting the two nodes."
        
        if not (isinstance(sourcep_id, int) and
                isinstance(source, AutomatonState) and
                isinstance(targetp_id, int) and
                isinstance(target, AutomatonState)):
            raise TypeError("Invalid arguments to deleteEdge")
        
        source_ID = str(sourcep_id) + '.' + str(source.id)
        target_ID = str(targetp_id) + '.' + str(target.id)
        edge_ID = source_ID + '-' + target_ID
        
        self.send(self.wrapJSON("de", edge_ID, {}))
    
    
    
if __name__ == "__main__":
    # Number of seconds to pause between tests
    delay = 2
    
    print "*** NOTE: Tests are not self-enclosed. Manually open Gephi,\n" + \
          "go to the 'Streaming' tab, and enter the streaming URL and\n" + \
          "start the master server. ***\n\n"
    
    print "Running tests..."
    # 'call' must be wrapped in a lambda function to avoid starting early.
    gephi_thread = Process(target=(lambda: call("gephi")))
    gephi_thread.start()
    time.sleep(10 * delay)
    
    print "Testing stream..."
    test_node = AutomatonState(id=1, state={'foo':1, 'bar':2})
    gs_client = GephiStream('client', port=8080)
    gs_server = GephiStream('server', port=8081)
    
    print "Testing wrapJSON..."
    assert gs_client.wrapJSON("an", "0." + str(test_node.id),
                    attribute_dict=test_node.state) == \
           '{"an":{"0.1":{"foo":1,"bar":2}}}\r\n'
    time.sleep(delay)
    
    print "Testing addNode..."
    gs_client.addNode(0, test_node)
    gs_server.addNode(0, test_node)
    gs_client.addNode(1, test_node)
    gs_server.addNode(1, test_node)
    gs_client.addNode(2, test_node)
    gs_server.addNode(2, test_node)
    time.sleep(delay)
    print "Testing changeNode..."
    gs_client.changeNode(0, test_node, {'foo':3})
    gs_server.changeNode(0, test_node, {'foo':3})
    gs_client.changeNode(1, test_node, {'bar':4})
    gs_server.changeNode(1, test_node, {'bar':4})
    time.sleep(delay)
    print "Testing deleteNode..."
    gs_client.deleteNode(2, test_node)
    gs_server.deleteNode(2, test_node)
    time.sleep(delay)
    
    print "Testing addEdge..."
    gs_client.addEdge(0, test_node, 1, test_node)
    gs_server.addEdge(0, test_node, 1, test_node)
    gs_client.addEdge(1, test_node, 0, test_node)
    gs_server.addEdge(1, test_node, 0, test_node)
    time.sleep(delay)
    print "Testing changeEdge..."
    gs_client.changeEdge(1, test_node, 0, test_node, {'foo':5})
    gs_server.changeEdge(1, test_node, 0, test_node, {'foo':5})
    time.sleep(delay)
    print "Testing deleteEdge..."
    gs_client.deleteEdge(1, test_node, 0, test_node)
    gs_server.deleteEdge(1, test_node, 0, test_node)
    time.sleep(delay)
    
    s = urlopen(gs_server.url)
    assertion_list = \
                     ['{"an":{"0.1":{"foo":1,"bar":2}}}\r\n',
                     '{"an":{"1.1":{"foo":1,"bar":2}}}\r\n',
                     '{"an":{"2.1":{"foo":1,"bar":2}}}\r\n',
                     '{"cn":{"0.1":{"foo":3}}}\r\n',
                     '{"cn":{"1.1":{"bar":4}}}\r\n',
                     '{"dn":{"2.1":{}}}\r\n',
                     '{"ae":{"0.1-1.1":{"source":"0.1","foo":1,"bar":2,"target":"1.1"}}}\r\n',
                     '{"ae":{"1.1-0.1":{"source":"1.1","foo":1,"bar":2,"target":"0.1"}}}\r\n',
                     '{"ce":{"1.1-0.1":{"foo":5}}}\r\n',
                     '{"de":{"1.1-0.1":{}}}\r\n']
    for i in range(10):
        assert s.readline() == assertion_list[i]
    
    print 'Closing streams...'
    gs_client.close()
    gs_server.close()
    s.close()
    print "Close Gephi to conclude unit tests."
    gephi_thread.join()
    print "Tests done."
