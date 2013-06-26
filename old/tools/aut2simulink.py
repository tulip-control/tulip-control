import re, copy, os, random, sys
import xml.etree.ElementTree as ET
import Queue
import exceptions


#Written by Robert Rogersten during SURF June 2012,
#Co-mentors Mumu Xu, Necmiye Ozay and Ufuk Topcu.

"""
This program takes an aut and smv file from a generated TuLiP controller
and automatically writes the MATLAB compatible script for that controller.
Run this program by typing "python programfile.py nameofautfile nameofmatlabfile", aut and smv file shall have the same name.
Do not include file extensions.
"""

class AutomatonState:
    """AutomatonState class for representing a state in a finite state
    automaton.  An AutomatonState object contains the following
    fields:

    - `stateid`: an integer specifying the state id of this AutomatonState object.
        - `state`: a dictionary whose keys are the names of the variables
      and whose values are the values of the variables.
    - `transition`: a list of id's of the AutomatonState objects to
      which this AutomatonState object can transition.
    """
    def __init__(self, stateid=-1, state={},transition=[]):
        self.stateid = stateid
        self.state = copy.copy(state)
        self.transition = transition[:]

def question(string):
    """This function ask a yes/no question and return their answer.
    The "answer" return value is one of "yes" or "no".
    """
    default="yes"
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    prompt = " [Y/n] "
    while True:
        print string
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print "Please respond with 'yes' or 'no'\n"

def loadFile(aut_file):
    """
    Construct an Automatonstate object from aut_file and
    put that object in a Queue.

    Input:

    - `aut_file`: the name of the text file containing the
    automaton, or an (open) file-like object.
    """
    if isinstance(aut_file, str):
        f = open(aut_file, 'r')
    else:
        f = aut_file  
    stateid = -1
    for line in f:
        # parse states
        if (line.find('State ') >= 0):
            stateid = re.search('State (\d+)', line)
            stateid = int(stateid.group(1))
            state = dict(re.findall('(\w+):(\w+)', line))
            state1 = dict(re.findall('(\w+):(-\w+)', line))
            state.update(state1)
        if re.search('successors', line):
            transition = list(re.findall('\d+', line))
            automaton=(stateid,state,transition)
            queue.put(automaton)
            queue1.put(automaton)
            queue2.put(automaton)
            
def readVariables(smv_file):
    """
    readVariables puts the enviroment and system variables from smv_file in two different
    Queues called system and enviroment.

    Input:

    - `smv_file`: the name of the text file containing the
      automaton, or an (open) file-like object.
    """
    if isinstance(smv_file, str):
        f = open(smv_file, 'r')
    else:
        f = smv_file  
    for line in f:
        if re.search('MODULE env',line):
            for line in f:
                if re.search(' : ', line):
                    env = str(re.findall('(\w+) :', line))
                    env = env[2:len(env)-2]
                    enviroment.put(env)
                if re.search('MODULE sys',line):
                    break
        if re.search('MODULE sys',line):
            for line in f:
                if re.search(' : ', line):
                    sys = str(re.findall('(\w+) :', line))
                    sys = sys[2:len(sys)-2]
                    system.put(sys)

def writeStartLine(enviroment,system,f):
    """
    writeStartLine writes the first lines before the
    switch cases in the matlab file.

    Input:

    - enviroment queue
    - system queue
    - fileobject f
    """
    f.write('function [')
    for i in range(system.qsize()):
        count = system.qsize()
        temp = system.get()
        if count == i+1:
            f.write(temp)
        else:
            f.write(temp+',')
        system.put(temp)
    f.write('] = '+sys.argv[2]+'(')
    for i in range(enviroment.qsize()):
        count = enviroment.qsize()
        temp = enviroment.get()
        if count == i+1:
            f.write(temp)
        else:
            f.write(temp+',')
        enviroment.put(temp)
    f.write(")\nglobal state;\ncoder.extrinsic('disp');\nswitch state\n")

def writeCase(enviroment,system,f,verbosem):
    """
    writeCase writes the
    switch cases in the matlab file.

    Input:

    - enviroment queue
    - system queue
    - fileobject f
    - verbosem
    """

    #for each case
    for i in range(queue.qsize()):
        f.write('\tcase '+str(i)+'\n')

        #for each condition within each case
        temp=queue.get()
        ef=0
        for k in range(queue1.qsize()):
            temp2=queue1.get()
            if str(k) in temp[2]:
                if ef == 0:
                    f.write('\t\tif ')
                    ef=1
                else:
                    f.write('\t\telseif ')
                for l in range(enviroment.qsize()):
                    count=enviroment.qsize()
                    temp1=enviroment.get()
                    if count == l+1:
                        f.write(temp1+' == '+temp2[1][temp1])
                    else:
                        f.write(temp1+' == '+temp2[1][temp1]+' && ')
                    enviroment.put(temp1)
                f.write('\n')
                if verbosem==1:
                    f.write('\t\t\tstate = '+str(temp2[0])+';\n')
                elif verbosem==0:
                    f.write('\t\t\tstate = '+str(temp2[0])+'\n')
                else:
                    raise Exception
                for l in range(system.qsize()):
                    temp1=system.get()
                    if verbosem==1:        
                        f.write('\t\t\t'+temp1+' = '+temp2[1][temp1]+';\n')
                    elif verbosem==0:
                        f.write('\t\t\t'+temp1+' = '+temp2[1][temp1]+'\n')
                    else:
                        raise Exception
                    system.put(temp1)
            queue1.put(temp2)


        #else statement for each case
        if not temp[2]:
            for l in range(system.qsize()):
                temp1=system.get()
                if verbosem==1:  
                    f.write('\t\t'+temp1+' = '+temp[1][temp1]+';\n')
                elif verbosem==0:
                    f.write('\t\t'+temp1+' = '+temp[1][temp1]+'\n')
                else:
                    raise Exception
                system.put(temp1)
        else:
            f.write('\t\telse\n')
            f.write("\t\t\tdisp('Cannot find a valid successor, environment assumption is like to be violated')\n")
            for l in range(system.qsize()):
                temp1=system.get()
                if verbosem==1:
                    f.write('\t\t\t'+temp1+' = '+temp[1][temp1]+';\n')
                elif verbosem==0:
                    f.write('\t\t\t'+temp1+' = '+temp[1][temp1]+'\n')
                else:
                    raise Exception
                system.put(temp1)
            f.write('\t\tend\n')
        queue.put(temp)

    #the last case is an otherwise statement
    f.write('\totherwise\n')
    f.write("\t\tdisp('Cannot find a valid successor, environment assumption is like to be violated')\n")
    for l in range(system.qsize()):
        temp1=system.get()
        if verbosem==1: 
            f.write('\t\t'+temp1+' = 0;\n')
        elif verbosem==0:
            f.write('\t\t'+temp1+' = 0\n')
        else:
            raise Exception
        system.put(temp1)
    f.write('end')

def writeCaseNo(enviroment,system,f,verbosem):
    """
    writeCase writes the
    switch cases in the matlab file and exclude no successors.

    Input:

    - enviroment queue
    - system queue
    - fileobject f
    - verbosem
    """

    #for each case
    li=list()
    for i in range(queue.qsize()):
        q=queue.get()
        li.append(q[0])
        queue.put(q)
        
    for i in range(queue.qsize()):

        #for each condition within each case
        temp=queue.get()
        f.write('\tcase '+str(temp[0])+'\n')
        ef=0
        for k in range(queue2.qsize()):
            temp2=queue2.get()
            if str(k) in temp[2] and k in li:
                if ef == 0:
                    f.write('\t\tif ')
                    ef=1
                else:
                    f.write('\t\telseif ')
                for l in range(enviroment.qsize()):
                    count=enviroment.qsize()
                    temp1=enviroment.get()
                    if count == l+1:
                        f.write(temp1+' == '+temp2[1][temp1])
                    else:
                        f.write(temp1+' == '+temp2[1][temp1]+' && ')
                    enviroment.put(temp1)
                f.write('\n')
                if verbosem==1:
                    f.write('\t\t\tstate = '+str(k)+';\n')
                elif verbosem==0:
                    f.write('\t\t\tstate = '+str(k)+'\n')
                else:
                    raise Exception
                for l in range(system.qsize()):
                    temp1=system.get()
                    if verbosem==1:        
                        f.write('\t\t\t'+temp1+' = '+temp2[1][temp1]+';\n')
                    elif verbosem==0:
                        f.write('\t\t\t'+temp1+' = '+temp2[1][temp1]+'\n')
                    else:
                        raise Exception
                    system.put(temp1)
            queue2.put(temp2)


        #else statement for each case
        if not temp[2]:
            for l in range(system.qsize()):
                temp1=system.get()
                if verbosem==1:  
                    f.write('\t\t'+temp1+' = '+temp[1][temp1]+';\n')
                elif verbosem==0:
                    f.write('\t\t'+temp1+' = '+temp[1][temp1]+'\n')
                else:
                    raise Exception
                system.put(temp1)
        else:
            f.write('\t\telse\n')
            f.write("\t\t\tdisp('Cannot find a valid successor, environment assumption is like to be violated')\n")
            for l in range(system.qsize()):
                temp1=system.get()
                if verbosem==1:
                    f.write('\t\t\t'+temp1+' = '+temp[1][temp1]+';\n')
                elif verbosem==0:
                    f.write('\t\t\t'+temp1+' = '+temp[1][temp1]+'\n')
                else:
                    raise Exception
                system.put(temp1)
            f.write('\t\tend\n')
        queue.put(temp)

    #the last case is an otherwise statement
    f.write('\totherwise\n')
    f.write("\t\tdisp('Cannot find a valid successor, environment assumption is like to be violated')\n")
    for l in range(system.qsize()):
        temp1=system.get()
        if verbosem==1: 
            f.write('\t\t'+temp1+' = 0;\n')
        elif verbosem==0:
            f.write('\t\t'+temp1+' = 0\n')
        else:
            raise Exception
        system.put(temp1)
    f.write('end')
    
queue=Queue.Queue()
queue1=Queue.Queue()
queue2=Queue.Queue()
enviroment=Queue.Queue()
system=Queue.Queue()
try:
    loadFile(sys.argv[1]+'.aut')
    readVariables(sys.argv[1]+'.smv')
    q=question('Shall there be a semicolon printed after each variable assignment? [Y/n]')
    q2=question('Shall the script exclude no successors? [Y/n]')
    if q:
        verbosem=1
    else:
        verbosem=0
    if not os.path.isfile(sys.argv[2]+'.m'):
        f=open(sys.argv[2]+'.m','w')
        writeStartLine(enviroment,system,f)
        if q2:
            for i in range(queue.qsize()):
                temp=queue.get()
                temp1=queue1.get()
                if not temp[2] == []:
                    queue.put(temp)
                    queue1.put(temp1)
            writeCaseNo(enviroment,system,f,verbosem)
        else:
            writeCase(enviroment,system,f,verbosem)
        f.close()
        if queue.get()[0]==-1:
            raise IOError
        print 'MATLAB script written to '+sys.argv[2]+'.m'+' with success\n'
    else:
        print 'Enter a matlab filename that not exists'
except IOError:
    print 'Enter correct filename for a TuLiP generated controller, aut and smv file shall have the same name'
except IndexError:
    print 'Run this program by typing "python programfile.py nameofautfile nameofmatlabfile", aut and smv file shall have the same name'
