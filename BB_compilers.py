
import qiskit
from cyclone_compiler import rotate_ancillas, checkEmpty
import pickle as pkl
import math
from machine import Machine, MachineParams, Trap, Segment
from test_machines import *
import copy
import subprocess as sp
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def calculate_pauli_twirling(t, p):
    #honeywell trapped ion cite
    #here's ionQ: https://ionq.com/quantum-systems/forte
    """

    Cyclone Improvement range: (1e8, 1e7)
    ? Cyclone ONLY range: (1e7, 1e6) but at the small end of it (almost too flat for both - barely cyclone helps with error correction and baseline can't really tell)
    Non cyclone range: (1e9, 1e8) Gap kind of small, (1e10, 1e9) Gap too small - also best known machine 
    """
    # (1e-3, 1e7)
    #(1e-4, 1e10)
    #return 0, 0, 0
    
    
    #T1 = 1e8 #honeywell gives 1e12, ionq gives 1e7 #china paper = 1.2e10 t1, 4.2e9 t2 
    #T2 = 1e7 #honeywell gives 3e6, ionq gives 1e6 

    #REAL formula for log T1: ln(T1) = -0.99998246709278*ln(p) + 9.2104211120632
    #REAL formula for log T2: ln(T2) = -0.99998246709278x + 9.2104211120632 # let's inject equal t1, t2

    ###OLD formula for log T1: ln(T1) = -3.0791713714931*ln(p) – 5.1519920828629
    ####OLD formula for log T2: ln(T2) = -3.6230739164423*ln(p) – 11.211597692608
    if (p != 0):
        logT1 = -0.99998246709278*math.log(p) + 9.2104211120632
        logT2 = -0.99998246709278*math.log(p) + 9.2104211120632
        T1 = math.exp(logT1)
        T2 = math.exp(logT2)
        x = (1 - math.exp(-t/T1))/4
        y = x
        z = (1-math.exp(-t/T2))/2 - (1 - math.exp(-t/T1))/4
        return x, y, z
    else:
        return 0, 0, 0

def generate_BBcode_numbers(timing, code, output_file,baseline=False, GRIP=False, error_rates=[5e-4], num_shots=1000):
    """
    This method needs to take in timings, apply pauli twirling, run BB code for p phys 1e-3,2e-33e-3,4e3-5e-3
    When running multiple instances using subprocess, might want to make naming convention more specific for TMP and CODE
    Find fitting formula
    Return error rates 1e-4-1e-3

    """
    trials = num_shots
    arr = code.split("-")
    n = int(arr[0])
    k = int(arr[1])
    d = int(arr[2])
    path = f"CODE_{n}_{k}_{d}"
    os.makedirs(path, exist_ok=True)
    if (not(baseline)):
        if (GRIP): 
            file_path = os.path.join(path, "result_GRIP")
        else: #cyclone = plain result
            file_path = os.path.join(path, "result")
    else:
        file_path = os.path.join(path, "result_baseline")
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("error_rate\tnum_cycles\tn(E)\tlogical_error_rate\n")

    
    errors = error_rates
    for base_error in errors:
        print("On", base_error)
        injected_error, _, _ = calculate_pauli_twirling(timing, base_error)
        print("Injected error", injected_error)
        error = injected_error + base_error
        print("Adjusted error", error)
        print("Running setup...")
        sp.call(["python", "decoder_setup_batch.py", str(error), str(d), code], stdout=output_file)
        print("Running decoder...")
        if (baseline):
            sp.call(["python", "decoder_run_batch.py", str(base_error), str(error), str(d), code, str(trials), "Baseline"], stdout=output_file)
        else:
            if (GRIP):
                 sp.call(["python", "decoder_run_batch.py", str(base_error), str(error), str(d), code, str(trials), "GRIP"], stdout=output_file)
            else:
                sp.call(["python", "decoder_run_batch.py", str(base_error), str(error), str(d), code, str(trials), "Cyclone"], stdout=output_file)

def run_cyclone_BB_Compiler(code, trap_amount, error_rates=[5e-4], num_shots=1000):
    """
    Takes a BB circuit object and spits out qasm, timings
    Next, take those timings and inject error into BB decoder BPOSD
    """
    with open(f"singleCycleCircQASMobj_{code}.pkl", "rb") as file:
        circ = pkl.load(file)

    
    print("circ", circ)
    n = int(code.split("-")[0])
    input_tuple = generateInputTuplefromCirc(circ, n)
    ordered_cx_arr = input_tuple[2]
    appear_dict = {}
    for stab in ordered_cx_arr:
        for gate in stab:
            if gate in appear_dict.keys():
                appear_dict[gate] += 1
            else:
                appear_dict[gate] = 1
    print("appear dict", appear_dict)
    print("appear dict", min(appear_dict.values()))
    numAncilla = len(input_tuple[2])
    num_ions_per_region = math.ceil(input_tuple[3]/trap_amount) + math.ceil((numAncilla / 2) / trap_amount)

    machStart = "C"
    mpar_model1 = MachineParams()
    mpar_model1.alpha = 0.003680029
    mpar_model1.beta = 39.996319971
    mpar_model1.split_merge_time = 80
    mpar_model1.shuttle_time = 5
    mpar_model1.junction2_cross_time = 5
    mpar_model1.junction3_cross_time = 100
    mpar_model1.junction4_cross_time = 120
    mpar_model1.gate_type = "FM"
    mpar_model1.swap_type = "GateSwap"
    mpar_model1.ion_swap_time = 42
    m = make_circle_machine_length_n(num_ions_per_region, mpar_model1, trap_amount)
    machString = machStart + str(int(trap_amount))
    print("machString", machString)
    new_circ, time_array, cx_timings = cyclone_BB_codes(code, input_tuple, numAncilla, m, machString)
    log_file_string = "bb_cyclone_output.log"
    print("TIMINGS", sum(time_array))
    output_file = open(log_file_string,'w')
    generate_BBcode_numbers(sum(time_array), code, output_file, error_rates=error_rates, num_shots=num_shots)
    

def smart_baseline_Compiler(code):
    """
    Takes a BB circuit object and spits out gates, timings to a smart baseline with grid architecture
    """
    pass

def run_baseline_numbers(baseline_timings, error_rates=[5e-4], num_shots=1000):
    #baseline_timings = [96895, 125200, 160355]
    #assumes baseline timings has associated order of codes: 72-12-6, 90-8-10, and 144-12-12 like [100000, 200000, 300000]
    index = 0
    codes = ["72-12-6", "90-8-10", "144-12-12"]
    #codes = ["72-12-6", "72-12-6"]
    #codes = ["144-12-12"]
    for timing in baseline_timings: #really for code in baseline timings
        code = codes[index]
        log_file_string = f"{code}_baseline.log"
        output_file = open(log_file_string,'w')
        generate_BBcode_numbers(timing, code, output_file,baseline=True,GRIP=False,error_rates=error_rates,num_shots=num_shots)
        index += 1
    print("Successfully generated baseline BB numbers!")


def run_GRIP_numbers(GRIP_timings):
    #baseline_timings = [96895, 125200, 160355]
    #assumes baseline timings has associated order of codes: 72-12-6, 90-8-10, and 144-12-12 like [100000, 200000, 300000]
    index = 0
    codes = ["72-12-6", "90-8-10", "144-12-12"]
    #codes = ["144-12-12"]
    for timing in GRIP_timings: #really for code in baseline timings
        if (index  < 1):
            code = codes[index]
            log_file_string = f"{code}_GRIP.log"
            output_file = open(log_file_string,'w')
            generate_BBcode_numbers(timing, code, output_file,baseline=False, GRIP=True)
        index += 1
    print("Successfully generated baseline BB numbers!")



def generateInputTuplefromCirc(circ, n):
    gateSet = set()
    for x in circ:
        if (x[0] not in gateSet):
            gateSet.add(x[0])
    print("Circ", circ)
    print("Gate set", gateSet)
    x_arr = []
    z_arr = []
    cx_arr = []
    ancilla_len = n ###THIS IS A QUESTIONABLE LINE
    offset = int(ancilla_len/2)
    assert(ancilla_len//2==ancilla_len/2)

    for i in range(ancilla_len):
        cx_arr.append([])
        x_arr.append([])
        z_arr.append([])
    for x in circ:
        if (x[0] == "CNOT"):
            assert(len(x) == 3)
            if (x[1][0] == "Xcheck"):
                if (x[2][0] == "data_left"):
                    x_arr[x[1][1]].append(x[2][1])
                    cx_arr[x[1][1]].append(x[2][1])
                else:
                    assert(x[2][0] == "data_right")
                    x_arr[x[1][1]].append(x[2][1] + offset)
                    cx_arr[x[1][1]].append(x[2][1] + offset)
                assert(x[2][0] != "Zcheck")
                assert(x[2][0] != "Xcheck")
            elif (x[1][0] == "Zcheck"):
                #first = Zcheck
                assert(x[2][0] != "Xcheck")
                assert(x[2][0] != "Zcheck")
                print("shouldn't even happen")
                exit()
            elif (x[1][0] == "data_right"):
                #first = data_right
                assert(x[2][0] != "data_left")
                assert(x[2][0] != "data_right")
            elif (x[1][0] == "data_left"):
                #first = data_left
                assert(x[2][0] != "data_right")
                assert(x[2][0] != "data_left")

            else: #no options found
                assert(0)
            if (x[2][0] == "Xcheck"):
                #second = Xcheck
                assert(x[1][0] != "Zcheck")
                assert(x[1][0] != "Xcheck")
                print("shouldn't even happen")
                exit()
            elif (x[2][0] == "Zcheck"):
                #second = Zcheck
                if (x[1][0] == "data_left"):
                #print("x 2,1", x[2][1])
                    z_arr[x[2][1] + offset].append(x[1][1])
                    cx_arr[x[2][1]+ offset].append(x[1][1])
                else:
                    assert(x[1][0] == "data_right")
                    z_arr[x[2][1] + offset].append(x[1][1] + offset)
                    cx_arr[x[2][1]+ offset].append(x[1][1] + offset)
                assert(x[1][0] != "Xcheck")
                assert(x[1][0] != "Zcheck")
            elif (x[2][0] == "data_right"):
                #second = data_right
                assert(x[1][0] != "data_left")
                assert(x[1][0] != "data_right")
            elif (x[2][0] == "data_left"):
                #second = data_left
                assert(x[1][0] != "data_right")
                assert(x[1][0] != "data_left")
            else: #no options found
                assert(0)
            #qc.cx(first[x[1][1]], second[x[2][1]])
    print("x_arr", x_arr)
    print("z_arr", z_arr)
    print("cx_arr", cx_arr)
    x_count = 0
    z_count = 0
    for i in range(len(cx_arr)):
        if (len(x_arr[i]) > 0):
            x_count +=1
        if (len(z_arr[i]) > 0):
            z_count += 1
    print("counts", x_count, z_count)
    total_gates = 0
    for g in x_arr:
        for gate in g:
            total_gates += 1
    for g in z_arr:
        for gate in g:
            total_gates += 1
    print("Total gates (for figure in speedup of serialization plot)", total_gates)
    return (x_arr, z_arr, cx_arr, n)
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

def cyclone_BB_codes(name, input_tuple, numAncilla, machine, machString, shuttling_times=165, k=1): #shuttling times is 165 + gateSWAP + one gate at initial parallelization layer
    
    time_array = []
    trap_amount = len(machine.traps)

    machNumberArr = machString.split("C")
    machNumber = int(machNumberArr[1])
    #assert(trap_amount == machNumber) #the whole partitioning algorithm would break otherwise
    #assert(numAncilla == trap_amount)
    x_arr, z_arr, cx_arr, n = input_tuple
    ions = math.ceil(n/trap_amount) + math.ceil((numAncilla/2)/trap_amount)
    print("ions per trap is", ions)
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
        max_ancilla = math.ceil((numAncilla/2) /trap_amount)
        time_array.append(3*gcost*max_ancilla + shuttling_times + layer_gcost)
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
        time_array.append(3*gcost*max_ancilla + shuttling_times + layer_gcost)
        print("max ancilla is", max_ancilla)
        #cx_timings.append(cx_timing_temp)
        for cx_pair_layer in cx_timing_temp_layer:
            cx_timings.append(cx_pair_layer)
        current_mapping = rotate_ancillas(current_mapping, ancillas, trap_amount)
        #print("ancilla assignment", ancilla_assignment)
        #print("cx assignment dict here", cx_assignment)
    for i in range(len(cx_arr)): #ending h gates
        circ.h(i)
    time_array.append(100)
    print("time array", time_array)
    print("cx timings", cx_timings)
    print(sum(time_array))
    
    return circ, time_array, cx_timings



def generate_grand_and_ordered():
    with open(f"singleCycleCircQASMobj_90-8-10.pkl", "rb") as file:
        circ = pkl.load(file)
    x_arr, z_arr, cx_arr, n = generateInputTuplefromCirc(circ, 90)
    assert(len(x_arr) == len(z_arr))
    ordered_cx_arr = cx_arr
    grand_cx_arr = [x_arr, z_arr]
    with open("grand_cx_90-8-10.pkl", "wb") as file:
        pkl.dump(grand_cx_arr, file)
    with open("ordered_cx_90-8-10.pkl", "wb") as file:
        pkl.dump(ordered_cx_arr, file)
    print("Ordered", ordered_cx_arr)
    print("Grand", grand_cx_arr)
    print("Success!")


