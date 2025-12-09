from ldpc.codes import hamming_code
from bposd.css import css_code
from bposd.css_decode_sim import css_decode_sim
import numpy as np
from ldpc import bposd_decoder
import math
import qiskit
import copy
import scipy.sparse as sp


def get_input_tuple_from_code_qldpc(code, css=True):
    hx, hz = code.hx, code.hz
    #if (isinstance(code, css_code)):
        #hx, hz = hx.toarray(), hz.toarray()
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

def generate_circuit(name, input_tuple, numAncilla=1): #modify this later to be generate smart circuit
    x_arr, z_arr, cx_arr, n = input_tuple
    ancillas = []
    for i in range(numAncilla):
        ancilla = n + i
        ancillas.append(ancilla)
    print("ancillas", ancillas)
    circ = qiskit.QuantumCircuit(n + numAncilla)
    ancilla_pointer = 0
    assert(len(x_arr) == len(z_arr))
    for i in range(len(x_arr)):
        circ.h(i)
    for row in range(len(x_arr)):
        x_gates = x_arr[row]
        z_gates = z_arr[row]
        cx_gates = cx_arr[row]
        if (len(x_arr[row]) > 0):
            assert(len(z_arr[row]) == 0) #CSS Code condition, unless stabilizer circuits are being generated
            for c in cx_gates:
                circ.cx(ancillas[ancilla_pointer], c)
        else:
            assert(len(z_arr[row]) > 0) #CSS code condition, refer to above
            for c in cx_gates:
                circ.cx(c, ancillas[ancilla_pointer])
        ancilla_pointer += 1
        ancilla_pointer = ancilla_pointer % len(ancillas)
    for i in range(len(x_arr)):
        circ.h(i)
    smallName = name + ".qasm"
    #nameString = "no_cycles/circuits/" + name + "/" + smallName
    nameString = "CSSCodes/" + smallName
    circ.qasm(formatted=True, filename=nameString)
    print("depth", circ.depth())
    return circ

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

error_rates = [1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3]
generators = [3,5,7]
code_index = 0
times = [4090, 8850, 43177.4]
baseline_times = [7725, 41090, 262420]

names = ["[[7,1,x]]", "[[31,21,x]]", "[[127,113,x]]"]
num_stabilizers = [6, 10, 14] #purely to track num ancilla
code_LERs = []
code_bars = []
total_error_rates = []
for number in generators:
    
    h = np.array([
[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
[0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
[0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
[0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
[1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
[1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
])
    #print("h type", type(h))
    my_css_code=css_code(hx=h,hz=h) #create Steane code where both hx and hz are Hamming codes
    #print("Hx")
    # print(my_css_code.hx)
    # print("Hz")
    # print(my_css_code.hz)
    # print("Lx Logical")
    # print(my_css_code.lx)
    # print("Lz Logical")
    # print(my_css_code.lz)
    my_css_code.test()
    input_tuple = get_input_tuple_from_code_qldpc(my_css_code)
    print("Distance", my_css_code.compute_code_distance())
    exit()
    generate_circuit(names[code_index], input_tuple, num_stabilizers[code_index])
    #print("distance", my_css_code.compute_code_distance())
    LERS = []
    bars = []
    confirm_ps = []
    for error_index in range(len(error_rates)):
        p = error_rates[error_index]
        time = times[code_index]
        injected_p, _, _ = calculate_pauli_twirling(time, p) #X and Y and Z
        print("Injected p", injected_p)
        p = injected_p + p
        print("Total physical error rate", p)
        confirm_ps.append(p)



        osd_options = {
            "error_rate": p,
            "target_runs": 1000,
            "xyz_error_bias": [0, 0, 1],
            "output_file": "test.json",
            "bp_method": "ms",
            "ms_scaling_factor": 0,
            "osd_method": "osd_cs",
            "osd_order": 3,
            "channel_update": None,
            "seed": 42,
            "max_iter": 0,
            "output_file": "test.json",
        }

        lk = css_decode_sim(hx=my_css_code.hx, hz=my_css_code.hz, **osd_options)
        ler = lk.osdw_logical_error_rate
        bar = lk.osdw_logical_error_rate_eb
        print("Got LER", ler)
        LERS.append(ler)
        bars.append(bar)
    print("X axis", error_rates)
    print("LER for code", code_index)
    print(LERS)
    total_error_rates.append(confirm_ps)
    code_LERs.append(LERS)
    code_index += 1
    code_bars.append(bars)

print("All code LERS", code_LERs)
print("All code Bars", code_bars)
print("All errors injected", total_error_rates)
###THREE USED CSS CODES:
# [[7,1,nan]]
# [[31,21,nan]]
# [[127,113,nan]]