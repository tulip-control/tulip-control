"""Function(s) that creates a Stateflow diagram from a Mealy Machine."""


import numpy
import copy
from tulip.transys import Mealy
import pdb


def to_stateflow(TS, filename):
	"""Converts a Mealy Machine to a Stateflow diagram.
	
	Inputs:
		- TS: an instance of tulip.transys.machines.Mealy()
		- filename: a string that is the valid filename of a MATLAB script
		  (must end in '.m')

	The function has no Python outputs, it writes a MATLAB script that will
	create a Simulink model with a Stateflow chart containing a Mealy Machine
	when run.
	"""

	# Check that the filename ends in ".m"
	if filename[-2:] != ".m":
		print "Filename must end in '.m'"
		raise

	# Get the intro text
	head_text = "sfnew('-Mealy','Tulip Controller');\n"
	head_text = head_text + "root = sfroot;\n"
	head_text = head_text + "m = root.find('-isa', 'Simulink.BlockDiagram');\n"
	head_text = head_text + "ch = m.find('-isa', 'Stateflow.Chart');\n"
	head_text = head_text + "ch.Name = 'TulipFSM';\n\n"

	# Get list of nodes (remove Sinit from list) and add to Stateflow`
	state_list = TS.states.find()
	sinit = TS.states.find('Sinit')[0]
	sinit_index = state_list.index(sinit)
	state_list.pop(sinit_index)
	num_nodes = len(state_list)
	node_positions = get_positions(num_nodes)
	(state_string, states_dict) = write_states(state_list, node_positions)

	# Get list of transitions (remove those from Sinit) and add to Stateflow
	transitions_list = TS.transitions.find()
	tsinit_list = TS.transitions.find(from_states=('Sinit',))
	for transition in tsinit_list:
		index = transitions_list.index(transition)
		transitions_list.pop(index)
	inputs = TS.inputs.keys()
	outputs = TS.outputs.keys()
	transitions_string = write_transitions(transitions_list, inputs, outputs,
		states_dict)

	# Get default transition for initial state
	initial_transitions = TS.transitions.find(from_states=('Sinit',))
	initial_str = write_init_string(initial_transitions, states_dict, inputs)

	# Declare Inputs and outputs on model
	input_string = write_data_string(inputs, "Input")
	output_string = write_data_string(outputs, "Output")

	# Open file handle and write the 
	filehandle = open(filename, 'w')
	filehandle.write(head_text)
	filehandle.write(state_string)
	filehandle.write(transitions_string)
	filehandle.write(initial_str)
	filehandle.write(input_string)
	filehandle.write(output_string)
	filehandle.close()



def get_positions(num_nodes):
	"""Temporary function that finds positions for each of the nodes. Will be
	replaced with something that will produce a more readable automaton later.
	In Stateflow, all states are represented with a rectangle with rounded 
	corners.

	Inputs:

		-num_nodes: integer containing the number of states in the Mealy Machine


	Outputs:

		- a (num_nodes x 4) numpy array in which each row has the format
			[top_left_xcoord, top_left_ycoord, rect_width, rect_height]
	"""

	# Make all rectangles 50 x 50 pixels big
	node_width = 50 * numpy.ones((num_nodes,1))
	node_height = 50 * numpy.ones((num_nodes,1))

	# Rectangles will be in a column. The top rectangle will be 100 pixels from
	# the top of the diagram and 100 pixels from the left. There will be 100
	# pixels of space in between each 50x50 rectangle.
	start_x = 100 * numpy.ones((num_nodes,1))
	start_y = 100 * numpy.ones((num_nodes,1))
	start_y_offsets = 150 * numpy.array(numpy.transpose(numpy.matrix( 
		range(num_nodes))))
	start_y += start_y_offsets

	# Assemble and return matrix
	return numpy.hstack([start_x, start_y, node_width, node_height])



def write_data_string(name_list, opt):
	"""Writes the part of the MATLAB script that creates inputs and outputs.

	Inputs:
		-name_list: a list of strings that contains the names of either all the
		inputs or all the outputs.

		-opt: a string that is either "Input" or "Output" (to match name_list)


	Outputs:
		-text: a string that contains the MATLAB code to write
	"""

	# Check that opt is one of two possible values
	assert((opt == "Input") or (opt == "Output"))

	cell_var = "data_" + opt
	text = cell_var + " = cell(1," + str(len(name_list)) + ");\n"
	for i, name in enumerate(name_list):
		start_string = cell_var + "{" + str(i+1) + "}"
		text = text + start_string + " = Stateflow.Data(ch);\n"
		text = text + start_string + ".Name = '" + name + "';\n"
		text = text + start_string + ".Scope = '" + opt + "';\n"
	text = text + "\n"
	return text



def write_init_string(initial_transitions, states_dict, inputs):
	"""Given a set of possible initial states, writes MATLAB code so that
	Stateflow's initial state labels are added to each initial state.

	Inputs:

		-initial_states: An non-string iterable containing the names of the
		initial states in the Python Mealy Machine

		-states_dict: A dictionary mapping label of state in the Python Mealy
		Machine to the name of the MATLAB variable that will contain a handle to
		the corresponding state object in stateflow.


	Outputs:

		-text: a string that contains MATLAB code
	"""
	num_inits = len(initial_transitions)
	text = "inits = cell(1," + str(num_inits) + ");\n"
	for i, init_transition in enumerate(initial_transitions):

		init_state = init_transition[1]
		values_dict = init_transition[2]

		# Name of variable that will contain handle to default transition obj
		start_string = "inits{" + str(i+1) + "}"

		# Handle containing state object that is an initial state
		init_state_str = str(states_dict[init_state])

		text = text + start_string + " = Stateflow.Transition(ch);\n"
		text = text + start_string + ".Destination = " + init_state_str + ";\n"

		# Set position of ends of arrows and beginnings of arrows
		text = text + start_string + ".DestinationOClock = 9;\n"
		text = text + start_string + ".SourceEndPoint = [" + \
			init_state_str + ".Position(1) - 30, " + \
			init_state_str + ".Position(2) + 25];\n"
		text = text + start_string + ".MidPoint = [" + \
			init_state_str + ".Position(1) - 15, " + \
			init_state_str + ".Position(2) + 25];\n"

		# Set string label from environment input
		label_string = "'["
		for env_var in inputs: # outputs
			env_value = values_dict[env_var]
			label_string = label_string + "(" + str(env_var) + "==" + \
				str(env_value) + ")&&"
		label_string = label_string[:-2] # Remove the last two & symbols
		label_string = label_string + "]'"

		text = text + start_string + ".LabelString = " + label_string + ";\n"
		text = text + "\n"


	text = text + "\n"
	return text	



def write_states(state_list, node_positions):
	"""Generates MATLAB code that creates a list of states and a dictionary
	mapping Python state names to MATLAB state names.


	Inputs:

		-state_list: a list of the names of each state in the Mealy Machine.
		(output of Mealy.states.list)

		-node_positions: a (number of states by 4) numpy array containing
		positions in Stateflow format. Output of get_positions(...) above.


	Outputs;

		-text: String containing MATLAB code

		-states_dict: A dictionary that maps Python state names to names of
		MATLAB variables containing handles to the corresponding Stateflow
		state.
	"""
	num_nodes = len(state_list)
	states_dict = {}
	text = "states = cell(1," + str(num_nodes) + ");\n"
	
	for i, state_tuple in enumerate(state_list):
		state_name = state_tuple[0]

		# Name of MATLAB variable containing handle
		line_start = "states{" + str(i+1) + "}"

		# Set name and position of state. Position must be set otherwise states
		# will overlap and Stateflow will throw an error.
		text = text + line_start + " = Stateflow.State(ch);\n"
		text = text + line_start + ".Name = 's" + str(state_name) + "';\n"
		text = text + line_start + ".Position = " + str(node_positions[i]) + \
			";\n"

		states_dict[state_name] = line_start

	text = text + "\n"
	return (text, states_dict)



def write_transitions(transitions_list, inputs, outputs, states_dict):
	"""Generates MATLAB code that creates transitions between states.

	Inputs:
		
		-transitions_list: The output of Mealy.transitions.find(), a list of
		tuples in the format:
			(start state, end state, dict with input and output values)

		-inputs: a list of the names of all the inputs, the output of
		Mealy.inputs.keys()

		-outputs: a list of the names of all the outputs, the output of
		Mealy.outputs.keys()

		-states_dict: a dict mapping Python state names to MATLAB state object
		handle names, the second output of write_states(...) defined above

	
	Outputs:
		
		-text: A string containing MATLAB code
	"""

	num_transitions = len(transitions_list)
	text = "trans = cell(1," + str(num_transitions) + ");\n"

	for i, transition in enumerate(transitions_list):
		# Variable containing the handle to the transition object (in MATLAB)
		line_start = "trans{" + str(i+1) + "}"

		start_state = transition[0]
		end_state = transition[1]
		values_dict = transition[2]
		
		text = text + line_start + " = Stateflow.Transition(ch);\n"
		text = text + line_start + ".Source = " + states_dict[start_state] + \
			";\n"
		text = text + line_start + ".Destination = " + states_dict[end_state] +\
			";\n"
	
		# String containing label to transition
		label_string = "'["
		
		# Environment Input
		for env_var in inputs: # outputs
			env_value = values_dict[env_var]
			label_string = label_string + "(" + str(env_var) + "==" + \
				str(env_value) + ")&&"
		label_string = label_string[:-2] # Remove the last two & symbols
		label_string = label_string + "]{"

		# System Output
		for sys_var in outputs:
			sys_value = values_dict[sys_var]
			label_string = label_string + str(sys_var) + "=" + str(sys_value) +\
				";"
		label_string = label_string + "}'"

		text = text + line_start + ".LabelString = " + label_string + ";\n"
		text = text + "\n"

	return text
