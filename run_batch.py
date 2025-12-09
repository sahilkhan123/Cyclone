###Baseline Setup

import subprocess as sp
import os
import datetime
import math
import json
PROG = []
code_indentifiers = [] #NOTE: relative order must be preserved throughout code
for x in os.listdir("QLDPC_Codes"):
    print(x)
    identifier = int(x.split("[[")[1].split(",")[0]) #the number of data qubits in the benchmark
    print("identifier", identifier)
    if (identifier != 1225): #skip this benchmark for time sake, too large when it gets to performing memory experiments on small p.
        PROG.append("QLDPC_Codes/" + x)
        code_indentifiers.append(identifier)
output_file = open('baseline_output.log','w')



#QLDPC BASELINE CAPACITIES = [5, 5, 6] #FOR 225, 625, 1225, RESPECTIVELY, DIRECT PLUG IN, NO MULTIPLY BY 2
#CSS BASELINE CAPACITIES - [4, 4, 4] $FOR 72,90,144 RESPECTIVELY, DIRECT PLUG IN, NO MULTIPLY BY 2
ion_constants = {225: 5, 375: 5, 625: 5, 72: 4, 90: 4, 144: 4}

mapper = "Greedy"
reorder = "Naive"
count = 0  
print("PROG", PROG)
baseline_values = {}
for p in PROG:
    n = code_indentifiers[count]
    MACHINE=[f"Gx{math.ceil(math.sqrt(n))}"] #substitute with sqrt(n) x sqrt(n) data qubits grid
    IONS = [str(ion_constants[n])]# capacities from above commented capacities
    for m in MACHINE:
        for i in IONS:
           print("Starting", p, datetime.datetime.now())
           code = sp.call(["python", "run.py", p, m, i, mapper, reorder, "1", "0", "0", "FM", "GateSwap"], stdout=output_file)
           if (code != 0):
               print("There was an error running the baseline. Please debug outside notebook in the run_batch.py (baseline) file")
           with open("tmp_output.json") as f:
               baseline_values[n] = json.load(f)[p]
    print("baseline values here", baseline_values)
           #sp.call(["python", "run_junction_network.py", p, m, i, mapper, reorder, "1", "0", "0", "FM", "GateSwap"])#, stdout=output_file)
    count += 1
           
