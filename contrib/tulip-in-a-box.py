#!/bin/env python
#
# Copyright (c) 2017 by California Institute of Technology
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
"""Bootstrap into a TuLiP installation using Travis CI configuration.

Requirements: `pyyaml`

N.B., this script requires `sudo` capabilities. It should run
without interruption if `sudo` escalation can occur without asking for
a password on the terminal, e.g., as typical in virtual machines.

However, you can remove explicit use of `sudo` by providing the switch
--no-sudo. One motivating use-case is Docker containers where the user
is `root` and `sudo` is not available.

N.B., this script installs for Python 3. If you want to instead use
Python 2.7, then change the `-p` switch that is given to `virtualenv`
in the corresponding call of `subprocess.check_call()` below.

This script should be run from the root of the sourcetree. E.g., you
can run `vm-bootstrap.sh` from within a virtual machine (VM), or
instead you can manually enter the following:

    sudo apt-get -y install git
    git clone https://github.com/tulip-control/tulip-control.git
    cd tulip-control
    python contrib/tulip-in-a-box.py
"""
import argparse
import os
import subprocess
import sys
import tempfile

import yaml


TRAVIS_PATH = '.travis.yml'


def main():
    sudo_prefix, travis_yml_path = parse_args()
    with open(travis_yml_path) as fp:
        travis_config = yaml.load(fp.read())
    arrange_base_env(sudo_prefix, travis_config)
    run_travis_commands(travis_config)


def arrange_base_env():
    subprocess.check_call(
        (sudo_prefix +
         'apt-get -y install python-pip libpython-dev ' +
         'libpython3-dev python-virtualenv'
         ).split())
    subprocess.check_call((sudo_prefix + 'pip install -I -U pip').split())
    subprocess.check_call(
        (sudo_prefix + 'apt-get -y install ' +
         ' '.join(travis_config['addons']['apt']['packages'])
         ).split())
    subprocess.check_call(['virtualenv', '-p', 'python3', 'PYt'])


def run_travis_commands(travis_config):
    section_names = ['before_install', 'install', 'before_script', 'script']
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as fp:
        fp.write('source PYt/bin/activate\n')
        for section in section_names:
            fp.write('\n'.join(travis_config[section]))
            fp.write('\n')
    subprocess.check_call(['/bin/bash', '-e', path])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-sudo', action='store_true',
                   help='do not use `sudo`')
    p.add_argument('travis_yml_path', type=str,
                   help='path to `.travis.yml` file')
    args = p.parse_args()
    if args.no_sudo:
        sudo_prefix = ''
    else:
        sudo_prefix = 'sudo '
    if args.travis_yml_path is None:
        travis_yml_path = TRAVIS_PATH
    else:
        travis_yml_path = args.travis_yml_path
    return sudo_prefix, travis_yml_path


if __name__ == '__main__':
    main()
