import numpy
import copy
from tulip.transys import Mealy

def get_positions(num_nodes):
	"""Temporary function that finds positions for each of the nodes. Will be
	replaced with a function that makes a more readable function later.
	
	Returns a num_nodes by 4 array with positions in Stateflow format."""

	node_width = 50 * numpy.ones((num_nodes,1))
	node_height = 50 * numpy.ones((num_nodes,1))

	start_x = 100 * numpy.ones((num_nodes,1))
	start_y = 100 * numpy.ones((num_nodes,1))

	start_y_offsets = 150 * numpy.array(numpy.transpose(numpy.matrix( 
		range(num_nodes))))

	start_y += start_y_offsets

	return numpy.hstack([start_x, start_y, node_width, node_height])



def to_stateflow(TS, filename):
	"""Takes a Mealy Machine and then writes a matlab script that will create
	the same Mealy Machine in Stateflow."""

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

	# Get list of nodes and add to Stateflow
	num_nodes = len(TS.states.list)
	node_positions = get_positions(num_nodes)
	(state_string, states_dict) = write_states(TS.states.list, node_positions)

	# Get list of transitions and add to Stateflow
	transitions_list = TS.transitions.find()
	inputs = TS.inputs.keys()
	outputs = TS.outputs.keys()
	transitions_string = write_transitions(transitions_list, inputs, outputs,
		states_dict)

	# Get default transition for initial state
	initial_states = tuple(TS.states.initial)
	initial_str = write_init_string(initial_states, states_dict)

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


def write_data_string(name_list, opt):
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


def write_init_string(initial_states, states_dict):
	num_inits = len(initial_states)
	text = "inits = cell(1," + str(num_inits) + ");\n"
	for i, init_state in enumerate(initial_states):
		start_string = "inits{" + str(i+1) + "}"
		init_state_str = str(states_dict[init_state])
		text = text + start_string + " = Stateflow.Transition(ch);\n"
		text = text + start_string + ".Destination = " + init_state_str + ";\n"
		text = text + start_string + ".DestinationOClock = 9;\n"
		text = text + start_string + ".SourceEndPoint = [" + \
			init_state_str + ".Position(1) - 30, " + \
			init_state_str + ".Position(2) + 25];\n"
		text = text + start_string + ".MidPoint = [" + \
			init_state_str + ".Position(1) - 15, " + \
			init_state_str + ".Position(2) + 25];\n"
	text = text + "\n"
	return text	

def write_states(state_list, node_positions):
	num_nodes = len(state_list)
	states_dict = {}
	text = "states = cell(1," + str(num_nodes) + ");\n"
	
	for i, state_name in enumerate(state_list):
		line_start = "states{" + str(i+1) + "}"
		text = text + line_start + " = Stateflow.State(ch);\n"
		text = text + line_start + ".Name = 's" + str(state_name) + "';\n"
		text = text + line_start + ".Position = " + str(node_positions[i]) + \
			";\n"
		states_dict[state_name] = line_start

	text = text + "\n"
	return (text, states_dict)


def write_transitions(transitions_list, inputs, outputs, states_dict):
	num_transitions = len(transitions_list)
	text = "trans = cell(1," + str(num_transitions) + ");\n"

	for i, transition in enumerate(transitions_list):
		line_start = "trans{" + str(i+1) + "}"
		start_state = transition[0]
		end_state = transition[1]
		values_dict = transition[2]
		
		text = text + line_start + " = Stateflow.Transition(ch);\n"
		text = text + line_start + ".Source = " + states_dict[start_state] + \
			";\n"
		text = text + line_start + ".Destination = " + states_dict[end_state] +\
			";\n"
	
		# Compute the label string
		label_string = "'["
		for env_var in inputs:
			env_value = values_dict[env_var]
			label_string = label_string + "(" + str(env_var) + "==" + \
				str(env_value) + ")&&"
		label_string = label_string[:-2] # Remove the last two & symbols
		label_string = label_string + "]{"
		for sys_var in outputs:
			sys_value = values_dict[sys_var]
			label_string = label_string + str(sys_var) + "=" + str(sys_value) +\
				";"
		label_string = label_string + "}'"

		text = text + line_start + ".LabelString = " + label_string + ";\n"
		text = text + "\n"

	return text
