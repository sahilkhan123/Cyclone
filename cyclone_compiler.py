import qiskit
from machine import Machine, MachineParams, Trap, Segment
from test_machines import *
import numpy as np
import pickle as pkl
import copy
import math
from ldpc.codes import hamming_code
from bposd.css import css_code
from bposd.css_decode_sim import css_decode_sim
import numpy as np
from ldpc import bposd_decoder
from quits.qldpc_code import *
from quits.circuit import get_qldpc_mem_circuit
from quits.decoder import sliding_window_bposd_circuit_mem
from quits.simulation import get_stim_mem_result
import time

def rotate_ancillas(current_mapping, ancillas, trap_num):
    ancilla_assignment = {}
    for x in current_mapping.keys():
        for a in current_mapping[x]:
            if (a in ancillas):
                ancilla_assignment[a] = (x + 1) % trap_num
    new_mapping = {}
    #print("ancillas here", ancillas)
    for i in range(trap_num): #CHANGED FROM len(ancillas) 11/13/24 to the more accurate, trapnum to account for general case of trap folding
        new_mapping[i] = []
    for x in current_mapping.keys():
        for d in current_mapping[x]:
            if (d not in ancillas):
                new_mapping[x].append(d)
    for x in ancillas:
        assignment = ancilla_assignment[x]
        #print("assignemnt", assignment)
        new_mapping[assignment].append(x)
    return new_mapping


def checkEmpty(dictionary):
    for x in dictionary.keys():
        outside = dictionary[x]
        for e in outside:
            if (len(e) > 0):
                return False
    return True

def parse_to_tuple(matrix, n): #n is the n in [[n,k,d]]
    arr = matrix.split("\n")
    x_arr = []
    z_arr = []
    cx_arr = []
    for x in arr:
        x_temp = []
        z_temp = []
        cx_temp = []
        splits = x.split("|")
        for i in range(len(splits)): #2 times
            side = splits[i]
            nums = side.split(" ")
            print("nums", nums)
            print(len(nums))
            print(n)
            assert(len(nums) == n)
            for j in range(len(nums)):
                number = nums[j]
                number = number.replace("[", "")
                number = number.replace("]", "")
                #print("number", number)
                if (int(number) == 1):
                    if (i == 0):
                        x_temp.append(j)
                    else:
                        assert(i == 1)
                        z_temp.append(j)
                    if (j not in cx_temp):
                        cx_temp.append(j)
        #print("x temp", x_temp)
        x_arr.append(x_temp)
        z_arr.append(z_temp)
        cx_arr.append(cx_temp)

    

    return (x_arr, z_arr, cx_arr, n)

def max_stabilizer_weight(input_tuple):
    cx_arr = input_tuple[2]
    print("cx_arr", cx_arr)
    max_length = -1
    for x in cx_arr:
        if len(x) > max_length:
            max_length = len(x)
    return max_length



def get_data_from_cx_arr(cx_arr):
    data = []
    for x in cx_arr:
        for d in x:
            if (d not in data):
                data.append(d)
    return data   

def get_cyclone_mapping(cx_arr,ancillas, machine, machString, ions):
    trap_amount = len(machine.traps)
    machNumberArr = machString.split("C")
    machNumber = int(machNumberArr[1])
    #assert(trap_amount == machNumber)
    mapping_dict = {}
    data_list = get_data_from_cx_arr(cx_arr=cx_arr)
    for i in range(trap_amount):
        mapping_dict[i] = []
    for i in range(len(data_list)):
        #NAIVE MAPPING - REPLACE WITH FIXED PARTITIONER
        mapping_dict[i % trap_amount].append(data_list[i])
    
    #later add ancilla
    for i in range(len(ancillas)): #attempting to fix with mod because I originally wanted num_ancilla must be = num_traps
        mapping_dict[i % trap_amount].append(ancillas[i])
    
    for x in mapping_dict.keys():
        pass
        #assert(len(mapping_dict[x]) <= ions) #otherwise we have exceeded maximum capacity. Above algorithm guarantees maximum dispersity of ions so that means something is wrong with numerics.
    
    return mapping_dict

def cyclone_official_compiler(name, input_tuple, numAncilla, machine, machString, ions, shuttling_times=165,ionSWAP=False): #shuttling times is 165 + gateSWAP + one gate at initial parallelization layer
    time_array = []
    trap_amount = len(machine.traps)


    component_gate = 0
    component_split = 0
    component_merge = 0
    component_move = 0
    component_swap = 0
    #can compute unrolled components via math

    
    print("ions", ions)
    machNumberArr = machString.split("C")
    machNumber = int(machNumberArr[1])
    #assert(trap_amount == machNumber) #the whole partitioning algorithm would break otherwise
    #assert(numAncilla == trap_amount)
    x_arr, z_arr, cx_arr, n = input_tuple
    ancillas = []
    for i in range(numAncilla):
        ancilla = n + i
        ancillas.append(ancilla)
    #print("ancillas", ancillas)
    circ = qiskit.QuantumCircuit(n + numAncilla)
    for i in range(len(cx_arr)): #this should really be n, n+ i
        circ.h(i)
    time_array.append(100) #single qubit gates parallelized
    ancilla_pointer = 0
    ancilla_assignment = {}
    cx_assignment = {}
    for i in range(len(ancillas)):
        cx_assignment[ancillas[i]] = []
    for i in range(len(cx_arr)):
        cx_assignment[ancillas[i % len(ancillas)]].append(cx_arr[i])
    print("cx array", cx_arr)
    print("x array", x_arr)
    print("z array", z_arr)
    print("cx assignment", cx_assignment)
    init_mapping = get_cyclone_mapping(cx_arr, ancillas, machine, machString, ions)
    print("init mapping here", init_mapping)
    for x in init_mapping: #assumes data are numbered before ancilla
        print("x", x)
        print("init mapping at x", init_mapping[x])
        print("length of init mapping at x", len(init_mapping[x]))
        for d in copy.deepcopy(init_mapping[x]):
            print("d", d)
            print("add", n + (len(ancillas)/2))
            if (d >= n + (len(ancillas) / 2)):
                init_mapping[x].remove(d)
    new_ancillas = []
    for x in range(len(ancillas)):
        if (x < len(ancillas)/2):
            new_ancillas.append(ancillas[x])
    ancillas = new_ancillas

    print("ancillas", ancillas)
    print("init mapping is:", init_mapping)
    print("x arr", x_arr)
    current_mapping = init_mapping
    cx_timings = []
    grand_cx_assignment = {}
    for i in range(len(ancillas)):
        grand_cx_assignment[ancillas[i]] = []
    for i in range(len(x_arr)):
        if (len(x_arr[i]) > 0):
            grand_cx_assignment[ancillas[i % len(ancillas)]].append(x_arr[i])
    #grand_cx_assignment = copy.deepcopy(cx_assignment)
    print("grand cx_assignment is", grand_cx_assignment)
    for potential_cx in grand_cx_assignment:
        print(potential_cx)
        print(grand_cx_assignment)
        print(grand_cx_assignment[potential_cx])
        assert(len(grand_cx_assignment[potential_cx]) == 1) 
    while (checkEmpty(grand_cx_assignment) == False):
        ancillas_used = [] #tracker to keep track of how many gates/cycle we are doing
        #gcost = 100
        gcost = max(100, 13.33*ions-54) #we assume one gate will happen per trap and be parallelized (this is the ideal case, but depending on how much data/trap it could increase)
        print("GCOST IS", gcost)
        cx_timing_temp_layer = [[]]
        print("current mapping", current_mapping)
        for x in current_mapping.keys():
            temp = current_mapping[x]
            assignment = []
            touching_ancillas = []
            for d in temp:
                 if (d not in ancillas):
                     assignment.append(d) #this is an array that includes the ancilla, it's looking for data anyway
                 elif (d in ancillas):
                     touching_ancillas.append(d)
                 else:
                     assert(0)
            for t in touching_ancillas:
                ancilla_assignment[t] = assignment
        parallelism_level = 1 #added
        cx_in_traps_dict = {}
        for m in machine.traps:
            cx_in_traps_dict[m.id] = 0 #binary, indicates if a gate in that trap is fired
        print("cx in traps dict", cx_in_traps_dict)
        for a in ancilla_assignment.keys():
            print("actual parallelism level", parallelism_level)
            new_parallelism_level = 1 #added
            temp = ancilla_assignment[a]
            for d in temp:
                print("new parallelism", new_parallelism_level)
                cx_temp = grand_cx_assignment[a] #2D array of the data qubits in a's check (although we only use one dimension because deprecation)
                #find first nonempty
                cx_first = [] #if there is no ancilla left, then we shouldn't be looking for any data from it
                for t in range(len(cx_temp)):
                    if (len(cx_temp[t]) > 0):
                        cx_first = cx_temp[t]
                        break
                #CX first is to ensure that the ancilla has yet to finish its checks
                print("cx first - the ancilla's data left to check with", cx_first)
                print("temp (the ancilla assignment's touching data)", temp)
                ###LEFT OFF HERE SUSPICION OF ERROR IS THAT I ONLY ADD TO PARALELLISM WHEN A SPECIFIC ANCILLA IS BEING REUSED (THIS IS ONLY THE CASE IN BASE CYCLONE)
                ###BUT MULTIPLE ANCILLA IN THE SAME TRAP COULD BE TRYING TO SIMULTANEOUSLY DO A GATE ALSO, THIS SHOULD ALSO ADD TO PARALLELIZATION LAYER. 
                
                for q in current_mapping.keys():
                    if (a in current_mapping[q]):
                        curr_trap = q
                if (d in cx_first):
                    circ.cx(d, a)
                    if (a not in ancillas_used):
                        cx_timing_temp_layer[0].append((d,a))
                        ancillas_used.append(a)
                        if (cx_in_traps_dict[curr_trap] == 0):
                            cx_in_traps_dict[curr_trap] = 1
                        else:
                            assert(cx_in_traps_dict[curr_trap] > 0)
                            cx_in_traps_dict[curr_trap] += 1
                            new_parallelism_level = cx_in_traps_dict[curr_trap]
                    else: #using the ancilla again in same layer - must add to parallelism level. Because this is an if/else it is not double counted with above case where its same trap
                        curr_timing_layer = []
                        for a_tup in cx_timing_temp_layer[0]:
                            curr_timing_layer.append(a_tup[0])
                            curr_timing_layer.append(a_tup[1])
                        len_tracker = 1
                        while (a not in curr_timing_layer and len_tracker < len(cx_timing_temp_layer)):
                            curr_timing_layer = []
                            for a_tup in cx_timing_temp_layer[len_tracker]:
                                curr_timing_layer.append(a_tup[0])
                                curr_timing_layer.append(a_tup[1])
                            len_tracker += 1
                        if (len_tracker == len(cx_timing_temp_layer)):
                            cx_timing_temp_layer.append([])
                        cx_timing_temp_layer[len_tracker].append((d, a))
                        cx_in_traps_dict[curr_trap] += 1
                        new_parallelism_level = cx_in_traps_dict[curr_trap]
                    #print("got here")
                    cx_first.remove(d)
                    grand_cx_assignment[a] = cx_temp #added line here 6/29/24

                    if (len(cx_first) == 0): #now we've completed a stabilizer, recheck for doing the max amount of stabilizers in parallel
                        print("got to this check")
            if new_parallelism_level > parallelism_level:
                parallelism_level = new_parallelism_level 
        print("parallelism level is now", parallelism_level)
        layer_gcost = gcost * parallelism_level                     
        print("grand is now", grand_cx_assignment)
        #print("real is now", real_cx_assignment)
        assert(d not in ancillas)
        #assert(numAncilla == 120) ##THIS IS JUST A CHECK ON THE DISTANCE 11 CASE, BECAUSE I MADE A HYPERSPECIFIC THING BELOW WITH DIVIDES BY 2: IT ASSUMES WE ARE SAVING ON NUMBER OF ANCILLA FOR SWAPPING.
        max_ancilla = math.ceil((numAncilla/2) /trap_amount)
        if (ionSWAP):
            time_array.append((80*ions + 80*(ions-1) + 42)*max_ancilla + shuttling_times + layer_gcost) #ionSWAP model
            component_swap += (80*ions + 80*(ions-1) + 42)*max_ancilla #this is technically swap
            component_merge += 80
            component_split += 80
            component_gate += layer_gcost
            component_move += 5
        else:
            time_array.append(3*gcost*max_ancilla + shuttling_times + layer_gcost)
            component_swap += 3*gcost*max_ancilla #this is technically swap
            component_merge += 80
            component_split += 80
            component_gate += layer_gcost
            component_move += 5
        #cx_timings.append(cx_timing_temp)
        for cx_pair_layer in cx_timing_temp_layer:
            cx_timings.append(cx_pair_layer)
        current_mapping = rotate_ancillas(current_mapping, ancillas, trap_amount)
        #print("ancilla assignment", ancilla_assignment)
        #print("cx assignment dict here", cx_assignment)
    print("grand cx_assignment is", grand_cx_assignment)
    print("NOW DONE WITH X STABILIZER ITERATION")
    print("current mapping", current_mapping)
    assert(current_mapping == init_mapping) #in the case of the sensitivity analysis if we are hitting this it very well could mean we have more traps than we need, leading cyclone to be confused at start of x checks (hasn't reset to original position yet)
    circ.barrier()
    grand_cx_assignment = {}
    ancilla_assignment = {} #clear this as well
    for i in range(len(ancillas)):
        grand_cx_assignment[ancillas[i]] = []
    for i in range(len(z_arr)):
        if (len(z_arr[i]) > 0):
            grand_cx_assignment[ancillas[i % len(ancillas)]].append(z_arr[i])
    #grand_cx_assignment = copy.deepcopy(cx_assignment)
    print("grand cx_assignment after prepping Z is", grand_cx_assignment)
    while (checkEmpty(grand_cx_assignment) == False):
        ancillas_used = [] #tracker to keep track of how many gates/cycle we are doing
        #gcost = 20
        gcost = max(100, 13.33*ions-54) #we assume one gate will happen per trap and be parallelized (this is the ideal case, but depending on how much data/trap it could increase)
        cx_timing_temp_layer = [[]]
        print("current mapping", current_mapping)
        for x in current_mapping.keys():
            temp = current_mapping[x]
            assignment = []
            touching_ancillas = []
            for d in temp:
                 if (d not in ancillas):
                     assignment.append(d) #this is an array that includes the ancilla, it's looking for data anyway
                 elif (d in ancillas):
                     touching_ancillas.append(d)
                 else:
                     assert(0)
            for t in touching_ancillas:
                ancilla_assignment[t] = assignment
        print("ancilla assignment", ancilla_assignment)
        #So ancilla assignment is a dictionary of ancilla# --> list of data qubits it is touching within this iteration
        parallelism_level = 1 #added
        cx_in_traps_dict = {}
        for m in machine.traps:
            cx_in_traps_dict[m.id] = 0 #binary, indicates if a gate in that trap is fired
        print("cx in traps dict", cx_in_traps_dict)
        for a in ancilla_assignment.keys():
            print("actual parallelism level", parallelism_level)
            for q in current_mapping.keys():
                    if (a in current_mapping[q]):
                        curr_trap = q
            new_parallelism_level = 1 #added
            temp = ancilla_assignment[a]
            for d in temp:
                print("new parallelism", new_parallelism_level)
                cx_temp = grand_cx_assignment[a] #2D array of the data qubits in a's check (although we only use one dimension because deprecation)
                #find first nonempty
                cx_first = [] #if there is no ancilla left, then we shouldn't be looking for any data from it
                for t in range(len(cx_temp)):
                    if (len(cx_temp[t]) > 0):
                        cx_first = cx_temp[t]
                        break
                print("cx first - the ancilla's data left to check with", cx_first)
                print("temp (the ancilla assignment's touching data)", temp)
                if (d in cx_first):
                    circ.cx(d, a)
                    if (a not in ancillas_used):
                        cx_timing_temp_layer[0].append((d,a))
                        ancillas_used.append(a)
                        if (cx_in_traps_dict[curr_trap] == 0):
                            cx_in_traps_dict[curr_trap] = 1
                        else:
                            assert(cx_in_traps_dict[curr_trap] > 0)
                            cx_in_traps_dict[curr_trap] += 1
                            new_parallelism_level = cx_in_traps_dict[curr_trap]
                    else: #using the ancilla again in same layer - must add to parallelism level. Because this is an if/else it is not double counted with above case where its same trap
                        curr_timing_layer = []
                        for a_tup in cx_timing_temp_layer[0]:
                            curr_timing_layer.append(a_tup[0])
                            curr_timing_layer.append(a_tup[1])
                        len_tracker = 1
                        while (a not in curr_timing_layer and len_tracker < len(cx_timing_temp_layer)):
                            curr_timing_layer = []
                            for a_tup in cx_timing_temp_layer[len_tracker]:
                                curr_timing_layer.append(a_tup[0])
                                curr_timing_layer.append(a_tup[1])
                            len_tracker += 1
                        if (len_tracker == len(cx_timing_temp_layer)):
                            cx_timing_temp_layer.append([])
                        cx_timing_temp_layer[len_tracker].append((d, a))
                        cx_in_traps_dict[curr_trap] += 1
                        new_parallelism_level = cx_in_traps_dict[curr_trap]
                    #print("got here")
                    cx_first.remove(d)
                    grand_cx_assignment[a] = cx_temp #added line here 6/29/24

                    if (len(cx_first) == 0): #now we've completed a stabilizer, recheck for doing the max amount of stabilizers in parallel
                        print("got to this check")

        
            if new_parallelism_level > parallelism_level:
                parallelism_level = new_parallelism_level 
        print("new parallelism level", new_parallelism_level)
        print("old parallelism level", parallelism_level)
        layer_gcost = gcost * parallelism_level                     
        print("parallelism level", parallelism_level)                   
        print("grand is now", grand_cx_assignment)
        #print("real is now", real_cx_assignment)
        assert(d not in ancillas)
        max_ancilla = math.ceil((numAncilla/2) /trap_amount)
        if (ionSWAP):
            time_array.append((80*ions + 80*(ions-1) + 42) + shuttling_times + layer_gcost) #ionSWAP model
            component_swap += 80*ions + 80*(ions-1) + 42 #this is technically swap
            component_merge += 80
            component_split += 80
            component_gate += layer_gcost
            component_move += 5
        else:
            time_array.append(3*gcost*max_ancilla + shuttling_times + layer_gcost)
            component_swap += 3*gcost*max_ancilla #this is technically swap
            component_merge += 80
            component_split += 80
            component_gate += layer_gcost
            component_move += 5
        print("max ancilla is", max_ancilla)
        #cx_timings.append(cx_timing_temp)
        for cx_pair_layer in cx_timing_temp_layer:
            cx_timings.append(cx_pair_layer)
        current_mapping = rotate_ancillas(current_mapping, ancillas, trap_amount)
        #print("ancilla assignment", ancilla_assignment)
        #print("cx assignment dict here", cx_assignment)
    #when cx'ing assert that both of them are not ancillas
    #algorithm:
    #assign each ancilla to a stabilizer in cx array
    #for each iteration while all the stabilizers are incomplete, rotate the ancilla (this is the assumption that could add some more swaps/merges at the end but we have to make), then check if any of the trap qubits are in stabilizer, pop them (this might add some additional split/merges at the end)
    
    for i in range(len(cx_arr)): #ending h gates
        circ.h(i)
    time_array.append(100)
    print("time array", time_array)
    print("cx timings", cx_timings)
    print(sum(time_array))
    print("COMPONENTS:")
    print("Gate component", component_gate)
    print("Merge component", component_merge)
    print("Split component", component_split)
    print("Move component", component_move)
    print("Swap component", component_swap)
    return circ, time_array, cx_timings

def get_input_tuple_from_code_qldpc(code, css=True):
    hx, hz = code.hx, code.hz
    if (isinstance(code, css_code)):
        hx, hz = hx.toarray(), hz.toarray()
    n = code.hx.shape[1] # same as num data
    x_arr = []
    z_arr = []
    cx_arr = []
    for arraypos in range(len(hx)):
        x = hx[arraypos]
        temp = []
        for d in range(len(x)):
            if (x[d] == 1):
                temp.append(d)
        cx_arr.append(copy.deepcopy(temp))
        x_arr.append(copy.deepcopy(temp))
        #make z array catch up to x array here
        z_arr.append([])
    if (css):    
        # We want if CSS True: Cx: [[0,2,3,5,6], [1,2,5,6], [4,5,2], [0,2,3,5,6], [1,2,5,6], [4,5,2]], x_arr: [[0,2,3,5,6], [1,2,5,6], [4,5,2], [], [], []] 
        for arraypos in range(len(hz)):
            z = hz[arraypos]
            temp = []
            for d in range(len(x)):
                if (x[d] == 1):
                    temp.append(d)
            cx_arr.append(copy.deepcopy(temp))
            z_arr.append(copy.deepcopy(temp))
            x_arr.append([])
        assert(len(x_arr) == len(z_arr))
        assert(len(x_arr) == len(cx_arr))
    else:
        for arraypos in range(len(hz)):
            z = hz[arraypos]
            temp = []
            for d in range(len(z)):
                if (z[d] == 1):
                    temp.append(d)
                    if (d not in x_arr[arraypos]):
                        assert(d not in cx_arr[arraypos])
                        cx_arr[arraypos].append(d)
            z_arr.append(copy.deepcopy(temp))
    
    return (x_arr, z_arr, cx_arr, n)

def generate_code_qldpc(h):
    code = HgpCode(h, h)         # Define the HgpCode object
    code.build_graph(seed=22)
    return code

if (__name__ == "__main__"):
    codes = []

    #""" QLPDC RUNNABLE BLOCK
    h1 = np.loadtxt('../quits/parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt', dtype=int)
    qldpc1 = generate_code_qldpc(h1) 
    codes.append(qldpc1)
    """
    h2 = np.loadtxt('../quits/parity_check_matrices/n=20_dv=3_dc=4_dist=8.txt', dtype=int)
    qldpc2 = generate_code_qldpc(h2) 
    h = np.loadtxt('../quits/parity_check_matrices/n=28_dv=3_dc=4_dist=10.txt', dtype=int)
    qldpc3 = HgpCode(h2, h1)         # Define the HgpCode object
    qldpc3.build_graph(seed=22)
    #codes.append(qldpc1)
    #codes.append(qldpc2)
    codes.append(qldpc3)
    """
    
    """
    generators = [3,5,7]
    for number in generators:
        h=hamming_code(number) #Hamming code parity check matrix, going to use 3,5,7 for parameters
        #print("h type", type(h))
        codes.append(css_code(hx=h,hz=h))     
    """
    for code in codes:
        n_data_qubits = len(code.data_qubits) #if QLDPC this line
        #n_data_qubits = code.N ###if CSS this line
        
        space_efficient_mode = True
        sensitivity_mode = True
        num_logical = code.lz.shape[0]
        num_zcheck, num_data = code.hz.shape
        num_xcheck, num_data = code.hx.shape
        assert(n_data_qubits == num_data)
        num_ancilla = num_xcheck + num_zcheck
        #num_timesteps = sum(list(code.num_colors.values())) #taken from notebook
        name = f"{num_data}-{num_logical}-x"

        #CUSTOM_TRAP_AMOUNT = num_ancilla/num_timesteps
        #CUSTOM_TRAP_AMOUNT = num_ancilla/2
        CUSTOM_TRAP_AMOUNT = 121
        CUSTOM_ION_AMOUNT = math.ceil(n_data_qubits/CUSTOM_TRAP_AMOUNT) + math.ceil(num_ancilla/CUSTOM_TRAP_AMOUNT)
        COMBINED_SHUTTLE_TIME = 165
        ionSWAP = True
        #COMBINED_SHUTTLE_TIME = 165 #80*2 + 5 + 5 split + merge + move + degree 2 junction
        

        #input_tuple = parse_to_tuple(stabilizer_matrix, n_data_qubits)
        input_tuple = get_input_tuple_from_code_qldpc(code)
        max_weight = max_stabilizer_weight(input_tuple) #this doesn't matter anymore
        rows = len(input_tuple[2])
        #print("Max stabilizer weight", max_weight)
        print("Rows", rows)
        #temporary:
        trap_amount = rows
        num_cyclone_ancilla = rows
        #num_ions_per_region = int(n_data_qubits/max_weight) + 2 # a ballpark for not overcrowding
        num_ions_per_region = 4
        print("Max capacity used", num_ions_per_region)

        machStart = "C"
        i = 2
        mpar_model1 = MachineParams()
        mpar_model1.alpha = 0.003680029
        mpar_model1.beta = 39.996319971
        mpar_model1.split_merge_time = 80
        mpar_model1.shuttle_time = 5
        mpar_model1.junction2_cross_time = 5
        mpar_model1.junction3_cross_time = 100
        mpar_model1.junction4_cross_time = 120
        mpar_model1.gate_type = "FM"
        mpar_model1.swap_type = "GateSwap" #this is just for the machine, don't adjust the swap type here
        mpar_model1.ion_swap_time = 42
        num_ions_per_region = CUSTOM_ION_AMOUNT
        trap_amount = CUSTOM_TRAP_AMOUNT
        m = make_circle_machine_length_n(num_ions_per_region, mpar_model1, int(trap_amount))
        machString = machStart + str(int(trap_amount))
        print("machString", machString)
        circ, timing, cxs = cyclone_official_compiler(name=name, input_tuple=input_tuple, machine = m, machString=machString, numAncilla=num_cyclone_ancilla, ions=num_ions_per_region, shuttling_times=COMBINED_SHUTTLE_TIME, ionSWAP=ionSWAP)
        print("Successfully created the circuit")
        
        print("Code", name)
        print("FINAL TIMING: ", sum(timing))
        time.sleep(10)
        
