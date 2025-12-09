import sys
from parse import InputParse
from mappers import *
from machine import Machine, MachineParams, Trap, Segment
from ejf_schedule import Schedule, EJFSchedule
from analyzer import *
from test_machines import *
import numpy as np
import matplotlib.pyplot as plt
import cyclone_graphs
import pickle as pkl

np.random.seed(12345)

#Command line args
#Machine attributes
openqasm_file_name = sys.argv[1]
machine_type = sys.argv[2]
num_ions_per_region = int(sys.argv[3])
mapper_choice = sys.argv[4]
reorder_choice = sys.argv[5]
serial_trap_ops = int(sys.argv[6])
serial_comm = int(sys.argv[7])
serial_all = int(sys.argv[8])
gate_type = sys.argv[9]
swap_type = sys.argv[10]

##########################################################
mpar_model1 = MachineParams()
mpar_model1.alpha = 0.003680029
mpar_model1.beta = 39.996319971
mpar_model1.split_merge_time = 80
mpar_model1.shuttle_time = 5
""" ORIGINAL JUNCTION TIMES

mpar_model1.junction2_cross_time = 5
mpar_model1.junction3_cross_time = 100
mpar_model1.junction4_cross_time = 120

"""
mpar_model1.junction2_cross_time = 5
mpar_model1.junction3_cross_time = 100
mpar_model1.junction4_cross_time = 120
mpar_model1.gate_type = gate_type
mpar_model1.swap_type = swap_type
mpar_model1.ion_swap_time = 42
machine_model = "MPar1"

'''
mpar_model2 = MachineParams()
mpar_model2.alpha = 0.003680029
mpar_model2.beta = 39.996319971
mpar_model2.split_merge_time = 80
mpar_model2.shuttle_time = 5
mpar_model2.junction2_cross_time = 5
mpar_model2.junction3_cross_time = 100
mpar_model2.junction4_cross_time = 120
mpar_model2.alpha
machine_model = "MPar2"
'''

print("Simulation")
print("Program:", openqasm_file_name)
print("Machine:", machine_type)
print("Model:", machine_model)
print("Ions:", num_ions_per_region)
print("Mapper:", mapper_choice)
print("Reorder:", reorder_choice)
print("SerialTrap:", serial_trap_ops)
print("SerialComm:", serial_comm)
print("SerialAll:", serial_all)
print("Gatetype:", gate_type)
print("Swaptype:", swap_type)

#Create a test machine
if machine_type == "G2x3":
    m = test_trap_2x3(num_ions_per_region, mpar_model1)
elif machine_type == "L6":
    m = make_linear_machine(6, num_ions_per_region, mpar_model1)
elif machine_type == "L1":
    m = make_linear_machine(1, num_ions_per_region, mpar_model1)
elif machine_type == "H6":
    m = make_single_hexagon_machine(num_ions_per_region, mpar_model1)
elif machine_type == "3x3":
    m = make_3x3_grid(num_ions_per_region, mpar_model1)
elif machine_type[:2] == "Gx":
    number = int(machine_type[2:])
    m = make_nxn_grid(number, num_ions_per_region, mpar_model1)
elif machine_type[:1] == "C":
    number = int(machine_type[1:])
    m = make_circle_machine_length_n(num_ions_per_region, mpar_model1, number)
elif machine_type[:1] == "J":
    number = int(machine_type[1:])
    m = make_junction_network(number, num_ions_per_region, mpar_model1)
else:
    assert 0

labels = {node: node.id for node in m.graph.nodes}
nx.draw(m.graph, labels=labels, with_labels=True, font_weight='bold', node_color='red', node_size=700)
plt.show()

#Parse the input program DAG
ip = InputParse()
ip.parse_ir(openqasm_file_name)

#Map the program onto the machine regions
#For every program qubit, this gives a region id
if mapper_choice == "LPFS":
    qm = QubitMapLPFS(ip,m)
elif mapper_choice == "Agg":
    qm = QubitMapAgg(ip, m)
elif mapper_choice == "Random":
    qm = QubitMapRandom(ip, m)
elif mapper_choice == "PO":
    qm = QubitMapPO(ip, m)
elif mapper_choice == "Greedy":
    qm = QubitMapGreedy(ip, m)
else:
    assert 0
mapping = qm.compute_mapping()

#Reorder qubits within a region to increse the use of high fidelity operations
if mapper_choice == "Greedy":
    init_qubit_layout = mapping
else:
    qo = QubitOrdering(ip, m, mapping)
    if reorder_choice == "Naive":
        init_qubit_layout = qo.reorder_naive()
    elif reorder_choice == "Fidelity":
        init_qubit_layout = qo.reorder_fidelity()
    else:
        assert 0

print(init_qubit_layout)
old_mapping = init_qubit_layout

#Schedule gates in the prorgam in topological sorted order
#EJF = earliest job first, here it refers to earliest gate first
#This step performs the shuttling
#h = np.loadtxt('../quits/parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt', dtype=int)
#code = cyclone_graphs.generate_code(h)


#grand_cx_arr,ordered_cx_arr = cyclone_graphs.create_grand_cx(code)
shortName = f"QLPDC[[72,12,6]]"
with open("grand_cx_72-12-6.pkl", "rb") as file:
            grand_cx_arr = pkl.load(file)
with open("ordered_cx_72-12-6.pkl", "rb") as file:
    ordered_cx_arr = pkl.load(file)
data_qubits = []
for o in ordered_cx_arr:
    for p in o:
        if (p not in data_qubits):
            data_qubits.append(p)
num_data = 72

#data_qubits = code.data_qubits
ancilla_placements, scheduling_gates_dict, stab_to_ancilla_placements = cyclone_graphs.generate_openqasm_file(shortName, grand_cx_arr, data_qubits, ordered_cx_arr)
##IN THIS SPECIAL CASE DATA PER NODE VALUE IS 1

node_containers = {}
for i in range(72):
    node_containers[i] = [data_qubits[i]]
init_qubit_layout = node_containers
print("ancilla placements", ancilla_placements)
for a in ancilla_placements.keys():
    data = ancilla_placements[a]
    node = cyclone_graphs.find_node(data, node_containers)
    init_qubit_layout[node].append(a)


ejfs = EJFSchedule(ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout, serial_trap_ops, serial_comm, serial_all)
#ejfs = EJFSchedule(ip.gate_graph, ip.cx_gate_map, m, old_mapping, serial_trap_ops, serial_comm, serial_all)


ejfs.run_GRIP(grand_cx_arr, stab_to_ancilla_placements)
#ejfs.run()
#Analyze the output schedule and print statistics
#analyzer = Analyzer(ejfs.schedule, m, old_mapping)
analyzer = Analyzer(ejfs.schedule, m, init_qubit_layout)
_, execution_time = analyzer.move_check()
print("Execution Time", execution_time)
print("SplitSWAP:", ejfs.split_swap_counter)
print("Split roadblocks", ejfs.split_roadblock_counter)
print("Merge roadblocks", ejfs.merge_roadblock_counter)
print("Gate roadblocks", ejfs.gate_roadblock_counter)
print("Total roadblocks", ejfs.merge_roadblock_counter + ejfs.split_roadblock_counter + ejfs.gate_roadblock_counter)
#analyzer.print_events()
print("----------------")
