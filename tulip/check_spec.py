# Copyright (c) 2011, 2012 by California Institute of Technology
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
"""
---------------------------------------------------
Syntax Checking functions for JTLV, nuSMV, and Spin
---------------------------------------------------

This module performs syntax check on LTL specs before they are sent to JTLV,
nuSMV, and Spin. It checks for:

1) That all parentheses are matched
2) That all variable names are spelled correctly.
3) That keys are all strings
4) That all values are numbers, booleans, or strings, but not alphanumeric
   strings
5) That JTLV input is in GR(1) format
6) That Spin and nuSMV input are in standard LTL format. 


It does NOT check:

1) That you used the right variables in the right places.
2) That you never set a variable to a value that it shouldn't be set to.
"""

import pyparsing
from pyparsing import *
from tulip import ltl_parse
from tulip.ltl_parse import *

def check_jtlv(assumption, guarantee, env_vars, sys_vars, disc_dynamics):
	"""
	Checks that an input (assumption spec, guarantee spec, and variables)
	form a valid GR(1) specification.
	"""

	# Check that dictionaries are in the correct format
	if not check_keys(env_vars):
		return False
	if not check_keys(sys_vars):
		return False
	if not check_values(env_vars):
		return False
	if not check_values(sys_vars):
		return False

	# Check for parentheses mismatch
	if not check_parentheses(assumption):
		return False
	if not check_parentheses(guarantee):
		return False

	# Combine dictionaries together
	total_dictionary = dict(sys_vars.items() + env_vars.items())
	try:
		discrete_dynamics_symbols = disc_dynamics.list_prop_symbol
		dummy_list = [ "" for symbol in discrete_dynamics_symbols ]
		temp_dictionary = dict(zip(discrete_dynamics_symbols, dummy_list))
		total_dictionary = dict(total_dictionary.items() +
		  temp_dictionary.items())
	except:
		pass

	# Check that all non-special-characters metioned are variable names
	# or possible values
	if not check_vars(assumption, total_dictionary):
		return False
	if not check_vars(guarantee, total_dictionary):
		return False

	# Check that the syntax is GR(1). This uses pyparsing
	prop = ltl_parse.proposition
	UnaryTemporalOps = ~bool_keyword + oneOf("next") + ~Word(nums + "_")
	next_ltl_expr = operatorPrecedence(prop,
        [("'", 1, opAssoc.LEFT, ASTUnTempOp),
        ("!", 1, opAssoc.RIGHT, ASTNot),
        (UnaryTemporalOps, 1, opAssoc.RIGHT, ASTUnTempOp),
        (oneOf("& &&"), 2, opAssoc.LEFT, ASTAnd),
        (oneOf("| ||"), 2, opAssoc.LEFT, ASTOr),
        (oneOf("xor ^"), 2, opAssoc.LEFT, ASTXor),
        ("->", 2, opAssoc.RIGHT, ASTImp),
        ("<->", 2, opAssoc.RIGHT, ASTBiImp),
        (oneOf("= !="), 2, opAssoc.RIGHT, ASTComparator),
        ])
	always_expr = pyparsing.Literal("[]") + next_ltl_expr
	always_eventually_expr = pyparsing.Literal("[]") + \
	  pyparsing.Literal("<>") + next_ltl_expr
	gr1_expr = next_ltl_expr | always_expr | always_eventually_expr

	# Final Check
	GR1_expression = pyparsing.operatorPrecedence(gr1_expr, 
	 [("&", 2, pyparsing.opAssoc.RIGHT)])
	try: 
		GR1_expression.parseString(assumption)
	except ParseException:
		print "Assumption is not in GR(1) format."
		return False
	try:
		GR1_expression.parseString(guarantee)
	except ParseException:
		print "Guarantee is not in GR(1) format"
		return False
	return True


def check_other(spec, variable_dictionary, disc_dynamics):
	"""
	Checks that a general LTL formula is has the correct syntax.
	"""
	try:
		discrete_dynamics_symbols = disc_dynamics.list_prop_symbol
		dummy_list = [ "" for symbol in discrete_dynamics_symbols ]
		temp_dictionary = dict(zip(discrete_dynamics_symbols, dummy_list))
		variable_dictionary = dict(variable_dictionary.items(), 
		  temp_dictionary.items())
	except:
		pass

	if not check_keys(variable_dictionary):
		return False
	if not check_values(variable_dictionary):
		return False
	if not check_parentheses(spec):
		return False
	if not check_vars(spec, variable_dictionary):
		return False
	try:
		ast = ltl_parse.parse(spec)
	except pyparsing.ParseException:
		print "Spec is not valid LTL."
		return False
	return True


def check_values(dictionary):
	"""Checks that all the possible values of the variables are either
	   strings or numbers (ints or floats), but not alphanumeric."""

	# Check that all values of variables are either "boolean", integers,
	# strings, and NOT alphanumeric strings.
	for key, value in dictionary.iteritems():
		# Convert the values to a list if it's not a list
		temp_value = ""
		try:
			temp_value = list(value)
		except TypeError:
			temp_value = [value]

		# Iterate over the list of values and make sure that no values are 
		# alphanumeric
		for possible_value in temp_value:
			if value == "boolean":
				continue
			elif (type(possible_value) == float) | \
			     (type(possible_value) == int):
				continue
			elif (type(possible_value) == str):
				found_string = False
				found_int = False
				for char in possible_value:
					try:
						temp = int(char)
						found_int = True
					except ValueError:
						found_string = True
					if found_string & found_int:
						print "Value " + str(possible_value) + \
						  " is invalid."
						return False
	return True


def check_parentheses(spec):
	"""Checks whether all the parentheses in a spec are closed.
	   Returns False if there are errors and True when there are no
	   errors."""

	open_parens = 0

	for index, char in enumerate(spec):
		if char == "(":
			open_parens += 1
		elif char == ")":
			open_parens -= 1

	if open_parens != 0:
		if open_parens > 0:
			print "The spec is missing " + str(open_parens) + " close-" + \
			  "parentheses or has " + str(open_parens) + " too many " + \
			  "open-parentheses"
		elif open_parens < 0:
			print "The spec is missing " + str(-open_parens) + " open-" + \
			  "parentheses or has " + str(open_parens) + " too many " + \
			  "close-parentheses"
		return False

	return True


def check_keys(dictionary):
	"""Yells at the user if any key is not a string and if a key is a
	   number."""

	for key in dictionary.keys():
		# Check that the keys are strings
		if type(key) != str:
			print "Key " + str(key) + " is invalid"
			return False

		# Check that the keys are not numbers
		try:
			int(key)
			print "Key " + str(key) + " is invalid"
			return False
		except ValueError:
			continue
		try:
			float(key)
			print "Key " + str(key) + " is invalid"
			return False
		except ValueError:
			continue
	return True


def check_vars(spec, dictionary):
	"""Make sure that all non operators in "spec" are in the dictionary."""

	# Replace all special characters with whitespace
	special_characters = ["next(", "[]", "<>", "->", "&", "|", "!", "+", \
	  "-", "=", "*", "(", ")", "\n", "<", ">", "<=", ">=", "<->", "^", \
	  "\t"]
	for word in special_characters:
		spec = spec.replace(word, "")

	# Now, replace all variable names and values with whitespace as well.
	possible_values = dictionary.keys()
	possible_values.extend(dictionary.values())
	for value in possible_values:
		if isinstance(value, (list, tuple)):
			for individual_value in value:
				spec = spec.replace(str(individual_value), "")
		else:
			spec = spec.replace(value, "")

	# Remove all instances of "true" and "false"
	spec = spec.lower()
	spec.replace("true", "")
	spec.replace("false", "")

	# Make sure that the resulting string is empty
	spec = spec.split()
	return not spec
