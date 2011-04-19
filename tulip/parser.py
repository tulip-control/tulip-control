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

########################################################
#add comment
def create_dict(prompt, vars):
    """Takes line and puts keys/vars into dictionary"""
    a = prompt.partition(':')
    name = ((a[0]).lstrip()).rstrip()
    value = ((a[2]).lstrip()).rstrip()
    vars[name] = value
    return vars

def create_list(prompt, list):
    """Reads line and puts names(string) into list"""
    a = prompt.partition(':')
    name = ((a[0]).lstrip()).rstrip()
    list.append(name)
    return list
########################################################
#Reads in file length and line byte length
file = open('specFileEx.spc', 'r')
line_offset = []
offset = 0
for line in file:
    line_offset.append(offset)
    offset += len(line)
file.seek(0)
lnum = len(line_offset)
file.close()

########################################################
#Reads in start line numbers for variable types
file = open('specFileEx.spc','r')
var_line = []
for i in range(0,lnum):
    text = file.readline()
    i += 1
    if text == 'env vars\n':
        var_line.append(i)
    elif text == 'sys disc vars\n':
        var_line.append(i)
    elif text == 'disc prop\n':
        var_line.append(i)
    elif text == 'sys cont vars\n':
        var_line.append(i)
    elif text == 'cont prop\n':
        var_line.append(i)
    elif text == 'sys dyn\n':
        var_line.append(i)
    elif text == 'spec\n':
        var_line.append(i)
file.close()

########################################################
#Saves spec file to variables
file = open('specFileEx.spc', 'r')
vars1, vars2, vars3, vars4, vars5, vars6, vars7 = {}, {}, {}, {}, {}, {}, {}
list1, list2, list3, list4, list5, list6, list7 = [], [], [], [], [], [], []

for j in range(var_line[0]+1, var_line[1]):
    line = file.seek(line_offset[j-1])
    text = file.readline()
    env_vars = create_dict(text, vars1)

for j in range(var_line[1]+1, var_line[2]):
    line = file.seek(line_offset[j-1])
    text = file.readline()
    disc_sys_vars = create_dict(text, vars2)

for j in range(var_line[2]+1, var_line[3]):
    line = file.seek(line_offset[j-1])
    text = file.readline()
    disc_props = create_dict(text, vars3)

for j in range(var_line[3]+1, var_line[4]):
    line = file.seek(line_offset[j-1])
    text = file.readline()
    cont_vars = create_list(text, list1)

for j in range(var_line[4]+1, var_line[5]):
    line = file.seek(line_offset[j-1])
    text = file.readline()
    cont_props = create_list(text, list2)

for j in range(var_line[5]+1, var_line[6]):
    line = file.seek(line_offset[j-1])
    text = file.readline()
    sys_dyn = create_dict(text, vars6)

if lnum > var_line[6]+1:
    for j in range(var_line[6]+1, lnum):
        line = file.seek(line_offset[j-1])
        text = file.readline()
#         spec = create_dict(text, vars7)
else:
    line = file.seek(line_offset[var_line[6]])
    text = file.readline()
#     spec = create_dict(text, vars7)

print env_vars
print disc_sys_vars
print disc_props
print cont_vars
print cont_props
print sys_dyn
# print spec

file.close()        

