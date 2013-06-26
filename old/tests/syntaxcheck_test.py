#!/usr/bin/env python

"""
Unit Tests for syntax checking module.
"""

from tulip import check_spec
from tulip import prop2part


# List of variables and specs
variable_dictionary = {}
variable_dictionary['isZoomed2'] = 'boolean'
variable_dictionary['isZoomed1'] = 'boolean'
variable_dictionary['x2'] = range(3)
variable_dictionary['n1'] = range(4)
variable_dictionary['n2'] = range(4)
variable_dictionary['x1'] = range(3)
variable_dictionary['z0'] = range(3)
system_dictionary = {'z0':range(3)}
environment_dictionary = {}
environment_dictionary['isZoomed2'] = 'boolean'
environment_dictionary['isZoomed1'] = 'boolean'
environment_dictionary['x2'] = range(3)
environment_dictionary['n1'] = range(4)
environment_dictionary['n2'] = range(4)
environment_dictionary['x1'] = range(3)


okay_assumption = """((x1 = 0) & (n1 = 0) & !isZoomed1) 
  & ((x2 = 0) & (n2 = 0) & !isZoomed2) 
  & []<>(x1 = 0)
  & []<>(x2 = 0)
  & [](((x1 = 1) & (n1 < 3)) -> next(x1 != 0))
  & [](((x2 = 1) & (n2 < 3)) -> next(x2 != 0))
  & [](((x1 = 0) -> (n1 = 0))
  & (((x1 = 0) & next(x1 = 1)) -> (next(n1) = 1))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 0)) -> (next(n1) = 1))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 1)) -> (next(n1) = 2))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 2)) -> (next(n1) = 3))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 3)) -> (next(n1) = 3)))
  & [](((x2 = 0) -> (n2 = 0))
  & (((x2 = 0) & next(x2 = 1)) -> (next(n2) = 1))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 0)) -> (next(n2) = 1))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 1)) -> (next(n2) = 2))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 2)) -> (next(n2) = 3))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 3)) -> (next(n2) = 3)))
  & [](
	   ((x1 = 0) -> next((x1 = 0) | (x1 = 1) | (x1 = 2) | (x1 = 3)))
	 & ((x1 = 1) -> next((x1 = 0) | (x1 = 1) | (x1 = 2)))
	 & ((x1 = 2) -> next((x1 = 0) | (x1 = 1) | (x1 = 2) | (x1 = 3)))
	 & ((x1 = 0) -> next((x1 = 0) | (x1 = 1) | (x1 = 2) | (x1 = 3)))
	)
  & [](
	   ((x2 = 0) -> next((x2 = 0) | (x2 = 1) | (x2 = 2) | (x2 = 3)))
	 & ((x2 = 1) -> next((x2 = 0) | (x2 = 1) | (x2 = 2)))
	 & ((x2 = 2) -> next((x2 = 0) | (x2 = 1) | (x2 = 2) | (x2 = 3)))
	 & ((x2 = 0) -> next((x2 = 0) | (x2 = 1) | (x2 = 2) | (x2 = 3)))
	)
  & [](((x1 != 0) & isZoomed1 & next(x1 != 0)) -> next(isZoomed1))
  & [](((x2 != 0) & isZoomed2 & next(x2 != 0)) -> next(isZoomed2))
  & [](((x1 = 0) & next(x1 = 1)) -> next(!isZoomed1))
  & [](((x2 = 0) & next(x2 = 1)) -> next(!isZoomed2))
  & []((z0 = 1) -> next(isZoomed1))
  & []((z0 = 2) -> next(isZoomed2))
  & [](((z0 != 1) & (!isZoomed1) & (x1!= 0)) -> next(!isZoomed1))
  & [](((z0 != 2) & (!isZoomed2) & (x2!= 0)) -> next(!isZoomed2))
  & []!((x1 = 0) & next(x1 > 1) & next(!isZoomed1) & (x2 = 0) & 
	   next(x2 > 1) & next(!isZoomed2))"""

okay_guarantee = """[]((z0 = 0) | (z0 = 1) | (z0 = 2))
  & []((z0 = 0) -> next((z0 = 0) | (z0 = 1) | (z0 = 2)))
  & []((z0 = 1) -> next((z0 = 0) | (z0 = 1) | (z0 = 2)))
  & []((z0 = 2) -> next((z0 = 0) | (z0 = 1) | (z0 = 2)))
  & []((x1 = 0) -> (z0 != 1))
  & []((x2 = 0) -> (z0 != 2))
  & [](isZoomed1 -> (z0 != 1))
  & [](isZoomed2 -> (z0 != 2))
  & []!((x1 = 1) & !isZoomed1 & next(x1 = 0) & next(!isZoomed1))
  & []!((x2 = 1) & !isZoomed2 & next(x2 = 0) & next(!isZoomed2))
  & []!((x1 != 0) & !isZoomed1 & (x1 > 1) & next(x1 = 0) & 
	   next(!isZoomed1) & (x2 != 0) & !isZoomed2& (x2 > 1) & 
	   next(x2 = 0) & next(!isZoomed2))"""

not_okay_assumption = """((x1 = 0) & (n1 = 0) & !isZoomed1) 
  & ((x2 = 0) & (n2 = 0) & !isZoomed2) 
  & []<>(x1 = 0)
  & []<>(x2 = 0)
  & [](((x1 = 1) & (n1 < 3)) -> next(x1 != 0))
  & [](((x2 = 1) & (n2 < 3)) -> next(x2 != 0))
  & [](((x1 = 0) -> (n1 = 0))
  & (((x1 = 0) & next(x1 = 1)) -> (next(n1) = 1))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 0)) -> (next(n1) = 1))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 1)) -> (next(n1) = 2))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 2)) -> (next(n1) = 3))
  & (((x1 != 0) & next(x1 != 0) & (n1 = 3)) -> (next(n1) = 3)))
  & [](((x2 = 0) -> (n2 = 0))
  & (((x2 = 0) & next(x2 = 1)) -> (next(n2) = 1))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 0)) -> (next(n2) = 1))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 1)) -> (next(n2) = 2))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 2)) -> (next(n2) = 3))
  & (((x2 != 0) & next(x2 != 0) & (n2 = 3)) -> (next(n2) = 3)))
  & [](
	   ((x1 = 0) -> next((x1 = 0) | (x1 = 1) | (x1 = 2) | (x1 = 3)))
	 & ((x1 = 1) -> next((x1 = 0) | (x1 = 1) | (x1 = 2)))
	 & ((x1 = 2) -> next((x1 = 0) | (x1 = 1) | (x1 = 2) | (x1 = 3)))
	 & ((x1 = 0) -> next((x1 = 0) | (x1 = 1) | (x1 = 2) | (x1 = 3)))
	)
  & [](
	   ((x2 = 0) -> next((x2 = 0) | (x2 = 1) | (x2 = 2) | (x2 = 3)))
	 & ((x2 = 1) -> next((x2 = 0) | (x2 = 1) | (x2 = 2)))
	 & ((x2 = 2) -> next((x2 = 0) | (x2 = 1) | (x2 = 2) | (x2 = 3)))
	 & ((x2 = 0) -> next((x2 = 0) | (x2 = 1) | (x2 = 2) | (x2 = 3)))
	)
  & [](((x1 != 0) & isZoomed1 & next(x1 != 0)) -> next(isZoomed1))
  & [](((x2 != 0) & isZoomed2 & next(x2 != 0)) -> next(isZoomed2))
  & [](((x1 = 0) & next(x1 = 1)) -> next(!isZoomed1))
  & [](((x2 = 0) & next(x2 = 1)) -> next(!isZoomed2))
  & []((z0 = 1) -> next(isZoomed1))
  & []((z0 = 2) -> next(isZoomed2))
  & [](((z0 != 1) & (!isZoomed1) & (x1!= 0)) -> next(!isZoomed1))
  & [](((z0 != 2) & (!isZoomed2) & (x2!= 0)) -> next(!isZoomed2))
  & []!((x1 = 0) & next(x1 > 1) & next(!isZoomed1) & (x2 = 0) & 
	   next(x2 > 1) & next(!isZoomed2))"""

not_okay_guarantee = """[]((z0 = 0) | (z0 = 1) | (z0 = 2))
  & []((z0 = 0) -> next((z0 = 0) | (z0 = 1) | (z0 = 2)))
  & []((z0 = 1) -> next((z0 = 0) | (z0 = 1) | (z0 = 2)))
  & []((z0 = 2) -> next((z0 = 0) | (z8 = 1) | (z0 = 2)))
  & []((x1 = 0) -> (z0 != 1))
  & []((x2 = 0) -> (z0 != 2))
  & [](isZoomed1 -> (z0 != 1))
  & [](isZoomed2 -> (z0 != 2))
  & []!((x1 = 1) & !isZoomed1 & next(x1 = 0) & next(!isZoomed1))
  & []!((x2 = 1) & !isZoomed2 & next(x2 = 0) & next(!isZoomed2))
  & []!((x1 != 0) & !isZoomed1 & (x1 > 1) & next(x1 = 0) & 
	   next(!isZoomed1) & (x2 != 0) & !isZoomed2& (x2 > 1) & 
	   next(x2 = 0) & next(!isZoomed2))"""



def parentheses_test():
	"""
	Checks that the function that tests for parentheses matchng is okay.
	"""
	correct_string = '(a (b c) (b s) ((d s) b))'
	too_many_open = '((a (b c) (b s) ((d s) b))'
	too_many_close = '(a (b c) (b s) ((d s) b)))'

	assert(check_spec.check_parentheses(correct_string))
	assert(not check_spec.check_parentheses(too_many_open))
	assert(not check_spec.check_parentheses(too_many_close))

	#print "Passed parentheses test"



def check_dict_keys():
	"""
	Makes sure that the functions that check keys are working.
	"""

	# Make sure that a valid dictionary goes through
	okay_dictionary = {}
	okay_dictionary['a'] = 'boolean'
	okay_dictionary['b'] = 'boolean'
	okay_dictionary['c'] = ['x', 'y', 'z']
	okay_dictionary['d'] = [1, 2, 3]
	okay_dictionary['e'] = ['1', '2', '3']
	okay_dictionary['f'] = '1'
	okay_dictionary['g'] = 'u'
	assert(check_spec.check_keys(okay_dictionary))

	# All keys should be strings. The strings should be words and not integers
	key_not_okay_dict = {}
	key_not_okay_dict['a'] = 'boolean'
	key_not_okay_dict[1] = 'boolean'
	assert(not check_spec.check_keys(key_not_okay_dict))

	# Make sure that things like '1' are also not okay. (Alphanumeric okay)
	key_not_okay_1 = {}
	key_not_okay_1['a'] = 'boolean'
	key_not_okay_1['1'] = 'boolean'
	assert(not check_spec.check_keys(key_not_okay_1))

	#print "Keys test passed"



def check_dict_values():
	"""
	Checks that the function that checks that all variables are in the right
	format is okay.
	"""

	# Make sure that a valid dictionary goes through.
	okay_dictionary = {}
	okay_dictionary['a'] = 'boolean'
	okay_dictionary['b'] = 'boolean'
	okay_dictionary['c'] = ['x', 'y', 'z']
	okay_dictionary['d'] = [1, 2, 3]
	okay_dictionary['e'] = ['1', '2', '3']
	okay_dictionary['f'] = '1'
	okay_dictionary['g'] = 'u'
	assert(check_spec.check_values(okay_dictionary))

	# Values should not be alphanumeric.
	values_not_okay_dict = {}
	values_not_okay_dict['a'] = 'boolean'
	values_not_okay_dict['b'] = ['a1', 'a2', 'a3']
	assert(not check_spec.check_values(values_not_okay_dict))

	#print "Values test passed."


def variable_test():
	"""
	Checks that the function that checks to make sure that everything is 
	spelled correctly.
	"""

	assert(check_spec.check_vars(okay_assumption, variable_dictionary))
	assert(not check_spec.check_vars(not_okay_guarantee, variable_dictionary))

	#print "Spelling check passed."
	

def jtlv_test():
	assert(check_spec.check_jtlv(okay_assumption, okay_guarantee, 
	  environment_dictionary, system_dictionary, []))

	#print "GR(1)/JTLV LTL parsing check passed"


def other_test():
	total_dictionary = dict(system_dictionary.items() + 
	  environment_dictionary.items())
	assert(check_spec.check_other(okay_assumption, total_dictionary, []))

	LTL_spec = ""

	#print "General LTL parsing check passed."


if __name__ == "__main__":
	parentheses_test()
	check_dict_keys()
	check_dict_values()
	variable_test()
	other_test()
	jtlv_test()
