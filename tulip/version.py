# Copyright (c) 2014 by California Institute of Technology
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
tulip package version
"""
version_info = (1, 1, 'a')

version = '.'.join([str(x) for x in version_info[:2] ])
version += version_info[2]


# Append annotation to version string to indicate development versions.
#
# An empty (modulo comments and blank lines) commit_hash.txt is used
# to indicate a release, in which case nothing is appended to version
# string as defined above.
import os.path
path_to_hashfile = os.path.join(os.path.dirname(__file__), "commit_hash.txt")
if os.path.exists(path_to_hashfile):
    commit_hash = ""
    with open(path_to_hashfile, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                # Ignore blank lines and comments, the latter being
                # any line that begins with #.
                continue

            # First non-blank line is assumed to be the commit hash
            commit_hash = line
            break

    if len(commit_hash) > 0:
        version += "-dev-" + commit_hash
else:
    version += "-dev-unknown-commit"
