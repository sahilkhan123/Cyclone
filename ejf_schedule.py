'''
Schedules gates in topologically sorted order
Policy is called EJF Earliest Job First, because its similar to the famous job scheduling policy

done - handle case where both traps are full
done - handle junction traffic flow
todo - skip - handle preemption??
todo - skip - handle traffic based routing
'''

import networkx as nx
import numpy as np
from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction
from rebalance import *
from tqdm import tqdm
import random

class EJFSchedule:
    #Inputs are
    #1. gate dependency graph - IR
    #2. gate_info = what are the qubits used by a two-qubit gate?
    #3. M = machine object
    #4. init_map = initial qubit mapping
    def __init__(self, ir, gate_info, M, init_map, serial_trap_ops, serial_comm, global_serial_lock):
        self.ir = ir
        self.gate_info = gate_info
        self.machine = M
        self.init_map = init_map
        self.debug_counter = 0
        self.rebalance_counter = 0
        self.gate_roadblock_counter = 0
        self.split_roadblock_counter = 0
        self.merge_roadblock_counter = 0
        #Setup scheduler
        self.machine.add_comm_capacity(2)
        #Add space for 2 extra ions in each trap
        self.SerialTrapOps = serial_trap_ops # ---> serializes operations on a single trap zone
        self.SerialCommunication = serial_comm # ---> serializes all split/merge/move ops
        self.GlobalSerialLock = global_serial_lock #---> serialize gates and comm

        #SerialTrapOps enforces that all operations in a trap are serialized
        #i.e., no parallel gates in a single ion chain/region
        self.schedule = Schedule(M)
        self.router = BasicRoute(M)
        self.gate_finish_times = {}

        #Some scheduling statistics
        #Count the number of times we had to clear some traps because of traffic blocks
        self.count_rebalance = 0
        self.split_swap_counter = 0

        #Create the sys_stage object which is used to track system state
        #from the perspective of the scheduler
        trap_ions = {}
        seg_ions = {}
        #print("M.traps", [t.id for t in M.traps])
        for i in M.traps:
            if init_map[i.id]:
                trap_ions[i.id] = init_map[i.id][:]
            else:
                trap_ions[i.id] = []
        for i in M.segments:
            seg_ions[i.id] = []
        self.sys_state = MachineState(0, trap_ions, seg_ions)
        #self.sys_state.print_state()

    #Find the earliest time at which a gate can be scheduled
    #Earliest time = max(dependent gate times)
    def gate_ready_time(self, gate):
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            #Each in edge is of the form (in_gate_id, this_gate_id)
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
            else:
                print("Error: Finish time of depenedent gate not found", in_edge)
                assert 0
        return ready_time

    #Find the time at which a particular qubit/ion is ready for another operation
    def ion_ready_info(self, ion_id):
        s = self.schedule
        this_ion_ops = s.filter_by_ion(s.events, ion_id)
        this_ion_last_op_time = 0
        this_ion_trap = None
        #If there is some operation that has happened for this ion:
        if len(this_ion_ops):
            #The last operation on an ion is either a gate or a merge in a trap
            assert (this_ion_ops[-1][1] == Schedule.Gate) or (this_ion_ops[-1][1] == Schedule.Merge)
            #Pick up the time and location of last operation
            this_ion_last_op_time = this_ion_ops[-1][3]
            this_ion_trap = this_ion_ops[-1][4]['trap']
        else:
            #Find which trap originally held this ion
            #It shouldn't have changed because no ops have happened for this ion
            did_not_find = True
            for trap_id in self.init_map.keys():
                if ion_id in self.init_map[trap_id]:
                    this_ion_trap = trap_id
                    did_not_find = False
                    break
            if did_not_find:
                print("Did not find:", ion_id)
            assert (did_not_find == False)

        #Double checking ion location from sys_state object
        if this_ion_trap != self.sys_state.find_trap_id_by_ion(ion_id):
            print(ion_id, this_ion_trap, self.sys_state.find_trap_id_by_ion(ion_id))
            self.sys_state.print_state()
            assert 0
        return this_ion_last_op_time, this_ion_trap

    #Add a split operation to the current schedule
    def add_split_op(self, clk, src_trap, dest_seg, ion, barrier_time=-1):
        m = self.machine
        if self.SerialTrapOps == 1:
            if (self.machine.mparams.swap_type == "CustomSwap"):
                ###WE CAN IGNORE THE CLK INPUT VARIABLES TO ADD A SPLIT/MOVE/MERGE AND CONSIDER NOTHING ABOUT TRAFFIC, ONLY ABOUT WHAT WAS THE LAST OP
                assert(barrier_time != -1)
                split_start = max(self.schedule.last_ion_event(ion), barrier_time) #assumes infinite channel bandwidth, find max channel badnwidth at the end
                #print("ion", ion)
                #print("ion events trap", self.schedule.filter_by_ion(self.schedule.events, ion))
                #print("split start will be", split_start)
                
                """
                last_event_time_on_trap, event_type = self.schedule.last_event_time_on_trap_and_type(src_trap.id)
                if (event_type == self.schedule.Split or event_type == self.schedule.Merge):
                    split_start = clk
                    #print("got here", clk)
                    #print("last event time is", last_event_time_on_trap) #lk is last event time because clk is making sure all resources in path are available
                   
                else:
                
                    split_start = max(clk, last_event_time_on_trap)
                """
            else:
                last_event_time_on_trap = self.schedule.last_event_time_on_trap(src_trap.id) #we can try to do multiple splits at once in good SWAP
                if (last_event_time_on_trap > clk):
                    #we are waiting on a busy trap, roadblock must be happening
                    self.split_roadblock_counter += 1

                split_start = max(clk, last_event_time_on_trap)
        else:
            split_start = clk
      
        if self.SerialCommunication == 1:
            if (self.machine.mparams.swap_type == "CustomSwap"):
                print("never supposed to go down this path, flip serial communication variable off")
                assert(0)
                split_start = self.schedule.last_ion_event(ion) #assumes infinite channel bandwidth, find max channel badnwidth at the end
                #print("ion", ion)
                #print("ion events comm", self.schedule.filter_by_ion(self.schedule.events, ion))
                #print("split start due to serial communication will be", split_start)
            else:
                last_comm_time = self.schedule.last_comm_event_time()
                split_start = max(split_start, last_comm_time)

        if self.GlobalSerialLock == 1:
            if (self.machine.mparams.swap_type == "CustomSwap"):
                print("never supposed to go down this path, flip serial communication variable off")
                assert(0)
                split_start = self.schedule.last_ion_event(ion) #assumes infinite channel bandwidth, find max channel badnwidth at the end
                #print("ion", ion)
                #print("ion events lock", self.schedule.filter_by_ion(self.schedule.events, ion))
                #print("split start due to serial lock will be", split_start)
            else:
                last_event_time_in_system = self.schedule.get_last_event_ts()
                split_start = max(split_start, last_event_time_in_system)
        #print("split start is", split_start)
        #print("split start at this point (should be 200) for ion 474")
        #if (ion == 474):
         #   print("Schedule at this point", self.schedule.events)
          #  exit()
        split_duration, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops = m.split_time(self.sys_state, src_trap.id, dest_seg.id, ion)
        self.split_swap_counter += split_swap_count
        split_end = split_start + split_duration
        self.schedule.add_split_or_merge(split_start, split_end, [ion], src_trap.id, dest_seg.id, Schedule.Split, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops)
        return split_end

    #Add a merge operation to the current schedule
    def add_merge_op(self, clk, dest_trap, src_seg, ion, barrier_time=-1):
        m = self.machine
        if self.SerialTrapOps == 1:
            if (self.machine.mparams.swap_type == "CustomSwap"):
                assert(barrier_time != -1)
                merge_start = max(self.schedule.last_ion_event(ion), barrier_time)
                """
                last_event_time_on_trap, event_type = self.schedule.last_event_time_on_trap_and_type(dest_trap.id)
                if (event_type == self.schedule.Merge or event_type == self.schedule.Split):
                    print("got here", clk)
                    print("last event time is", last_event_time_on_trap)
                    merge_start = clk
                else:  
                    merge_start = max(clk, last_event_time_on_trap)
                """
            else:
                last_event_time_on_trap = self.schedule.last_event_time_on_trap(dest_trap.id) #we can try to do multiple splits at once in good SWAP
                merge_start = max(clk, last_event_time_on_trap)
                if (last_event_time_on_trap > clk):
                    #we are waiting on a busy trap, roadblock must be happening
                    self.merge_roadblock_counter += 1
        else:
            merge_start = clk

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            merge_start = max(merge_start, last_comm_time)

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            merge_start = max(merge_start, last_event_time_in_system)
        merge_end = merge_start + m.merge_time(dest_trap.id)
        self.schedule.add_split_or_merge(merge_start, merge_end, [ion], dest_trap.id, src_seg.id, Schedule.Merge, 0, 0, 0, 0, 0)
        return merge_end

    #Add a move operation to the current schedule
    #This is one segment to segment move i.e., move ion from src_seg to dest_seg
    def add_move_op(self, clk, src_seg, dest_seg, junct, ion):
        m = self.machine

        ###MODIFIED/COMMENTED THE TOP LINE IN 
        #move_start = clk
        move_start = self.schedule.last_ion_event(ion)
        #print("MOVE START FOR ION", ion, move_start)
        #print("CORREPONDING SCHEDULE", self.schedule.filter_by_ion(self.schedule.events, ion))
        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            move_start = max(move_start, last_event_time_in_system)

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            move_start = max(move_start, last_comm_time)

        move_end = move_start + m.move_time(src_seg.id, dest_seg.id) + m.junction_cross_time(junct)
        move_start, move_end = self.schedule.junction_traffic_crossing(src_seg, dest_seg, junct, move_start, move_end)
        self.schedule.add_move(move_start, move_end, [ion], src_seg.id, dest_seg.id)
        return move_end

    #Add a gate operation to the current schedule
    def add_gate_op(self, clk, trap_id, gate, ion1, ion2):
        fire_time = clk
        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(trap_id)
            fire_time = max(clk, last_event_time_on_trap)
            if (last_event_time_on_trap > clk):
                    #we are waiting on a busy trap, roadblock must be happening
                    self.gate_roadblock_counter += 1

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            fire_time = max(fire_time, last_event_time_in_system)
        gate_duration = self.machine.gate_time(self.sys_state, trap_id, ion1, ion2)
        self.schedule.add_gate(fire_time, fire_time + gate_duration, [ion1, ion2], trap_id)
        self.gate_finish_times[gate] = fire_time + gate_duration
        return fire_time + gate_duration

    #Heuristic to determine direction of shuttling for two traps
    def shuttling_direction(self, ion1_trap, ion2_trap):
        #Other Possible policies: lookahead, traffic/path based
        m = self.machine
        ss = self.sys_state
        excess_cap1 = m.traps[ion1_trap].capacity - len(ss.trap_ions[ion1_trap])
        excess_cap2 = m.traps[ion2_trap].capacity - len(ss.trap_ions[ion2_trap])
        #both excess capacities can be 0 if the traps are full
        frac_empty = float(excess_cap1)/m.traps[ion1_trap].capacity
        #Whichever trap has more excess capacity choose that as the destination
        if excess_cap1 > excess_cap2:
            dest_trap = ion1_trap
            source_trap = ion2_trap
        else:
            dest_trap = ion2_trap
            source_trap = ion1_trap

        if excess_cap1 <= 0 and excess_cap2 <= 0:
            print(ion1_trap, ion2_trap)
            print(ss.trap_ions)
            print("Both traps full", ion1_trap, m.traps[ion1_trap].capacity,  ss.trap_ions[ion1_trap])
            #sahil code commented below, half works
            ion1_trap_ft = self.schedule.last_event_time_on_trap(ion1_trap)
            ion2_trap_ft = self.schedule.last_event_time_on_trap(ion2_trap)
            self.do_rebalance_traps(max(ion1_trap_ft, ion2_trap_ft))
            self.rebalance_counter += 1
            #assert 0
        return source_trap, dest_trap

    #Fire an end-to-end shuttle operation from src_trap to dest_trap
    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, barrier_time, route=[]):
        s = self.schedule
        m = self.machine
        #If route is not specified in the function args, find a route using
        #the router object passed to the scheduler
        if len(route):
            rpath = route
        else:
            rpath = self.router.find_route(src_trap, dest_trap)

        #Find the time that it will take to do this entire shuttle
        #This is required to find a feasible time for scheduling this shuttle
        t_est = 0
        for i in range(len(rpath)-1):
            src = rpath[i]
            dest = rpath[i+1]
            if type(src) == Trap and type(dest) == Junction:
                my_seg = m.graph[src][dest]['seg']
                t_est += m.mparams.split_merge_time
                #split_time(self.sys_state, src.id, my_seg.id, ion)
            elif type(src) == Junction and type(dest) == Junction:
                t_est += m.move_time(src.id, dest.id)
            elif type(src) == Junction and type(dest) == Trap:
                t_est += m.merge_time(dest.id)

        #This is the traffic-unaware/conservative version where we wait for the full path to be available
        #print("ion is", )
        ##OVERRIDE BY SAHIL, WE WILL IGNORE CLK IN CYCLONE CODE AS INPUT TO ADDING SHUTTLING OPS BECAUSE WE DON'T NEED TO BE CONSERVATIVE WITH TRAFFIC ANYMORE
        clk = self.schedule.identify_start_time(rpath, gate_fire_time, t_est)

        #Add the shuttling operations to the schedule based on the identified start time
        clk = self._add_shuttle_ops(rpath, ion, clk, barrier_time=barrier_time)

        #self.sys_state.trap_ions[src_trap].remove(ion)
        #self.sys_state.trap_ions[dest_trap].append(ion)
        return clk

    #Helper function to implement a shuttle
    def _add_shuttle_ops(self, spath, ion, clk, barrier_time):
        #Decompose into trap-trap paths
        #For each trap to trap path call a split-move*-merge sequence
        trap_pos = []
        for i in range(len(spath)):
            if type(spath[i]) == Trap:
                trap_pos.append(i)
        for i in range(len(trap_pos)-1):
            idx0 = trap_pos[i]
            idx1 = trap_pos[i+1]+1
            clk = self._add_partial_shuttle_ops(spath[idx0:idx1], ion, clk, barrier_time=barrier_time)
            self.sys_state.trap_ions[spath[trap_pos[i]].id].remove(ion)
            last_junct = spath[trap_pos[i+1]-1]
            dest_trap = spath[trap_pos[i+1]]
            last_seg = self.machine.graph[last_junct][dest_trap]['seg']
            orient = dest_trap.orientation[last_seg.id]
            if orient == 'R':
                self.sys_state.trap_ions[spath[trap_pos[i+1]].id].append(ion)
            else:
                self.sys_state.trap_ions[spath[trap_pos[i+1]].id].insert(0, ion)
        return clk

    #Helper function to implement a shuttle
    def _add_partial_shuttle_ops(self, spath, ion, clk, barrier_time):
        assert len([item for item in spath if type(item) == Trap]) == 2
        seg_list = []
        for i in range(len(spath)-1):
            u = spath[i]
            v = spath[i+1]
            seg_list.append(self.machine.graph[u][v]['seg'])
        
        clk = self.add_split_op(clk, spath[0], seg_list[0], ion, barrier_time=barrier_time)
        for i in range(len(seg_list)-1):
            u = seg_list[i]
            v = seg_list[i+1]
            junct = spath[1+i]
            clk = self.add_move_op(clk, u, v, junct, ion)
        clk = self.add_merge_op(clk, spath[-1], seg_list[-1], ion, barrier_time=barrier_time)
        return clk

    #Main scheduling function for a gate
    def schedule_gate(self, gate, specified_time=0):
        s = self.schedule
        #Find time at which the gate can be fired
        ready = self.gate_ready_time(gate)
        ion1 = self.gate_info[gate][0]
        ion2 = self.gate_info[gate][1]
        #Find time at which ions are ready
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        fire_time = max(ready, ion1_time, ion2_time)
        fire_time = max(fire_time, specified_time)

        #print("Gate", gate, "I1", ion1, "I2", ion2, "IT1", ion1_trap, "IT2", ion2_trap, "Ready", ready, "FireTime:", fire_time)

        if ion1_trap == ion2_trap:
            #Ions are co-located in a trap, no shuttling required
            self.add_gate_op(fire_time, ion1_trap, gate, ion1, ion2)
        else:
            #Check if there is at least one path to shuttle from src to dest trap
            #rebalances the machine if needed i.e., clear traffic blocks
            rebal_flag, new_fin_time = self.rebalance_traps(focus_traps=[ion1_trap, ion2_trap], fire_time=fire_time)
            if not rebal_flag:
                source_trap, dest_trap = self.shuttling_direction(ion1_trap, ion2_trap)
                if source_trap == ion1_trap:
                    moving_ion = ion1
                else:
                    moving_ion = ion2
                clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, fire_time, barrier_time=-1)
                self.add_gate_op(clk, dest_trap, gate, ion1, ion2)
            else:
                print("GOT TO REBALANCING")
                #This is for the rebalancing case, trap_ids compute till this point may be stale
                self.schedule_gate(gate, specified_time=new_fin_time)

    
    def schedule_gate_enhancement(self, gate, ancillas, specified_time=0):
        s = self.schedule
        #Find time at which the gate can be fired
        #ready = self.gate_ready_time(gate)
        ion1 = self.gate_info[gate][0]
        ion2 = self.gate_info[gate][1]
        #Find time at which ions are ready
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        fire_time = max(ion1_time, ion2_time)
        fire_time = max(fire_time, specified_time)
        print("Ion, Fire Time", ion1, fire_time)

        #print("Gate", gate, "I1", ion1, "I2", ion2, "IT1", ion1_trap, "IT2", ion2_trap, "Ready", ready, "FireTime:", fire_time)

        if ion1_trap == ion2_trap:
            #Ions are co-located in a trap, no shuttling required
            self.add_gate_op(fire_time, ion1_trap, gate, ion1, ion2)
        else:
            #Check if there is at least one path to shuttle from src to dest trap
            #rebalances the machine if needed i.e., clear traffic blocks
            rebal_flag, new_fin_time = self.rebalance_traps(focus_traps=[ion1_trap, ion2_trap], fire_time=fire_time)
            if not rebal_flag:
                source_trap, dest_trap = self.shuttling_direction(ion1_trap, ion2_trap)
                """"
                if source_trap == ion1_trap:
                    moving_ion = ion1
                else:
                    moving_ion = ion2
                """
                if (ion1 in ancillas):
                    moving_ion = ion1
                    source_trap = ion1_trap
                    dest_trap = ion2_trap
                elif (ion2 in ancillas):
                    moving_ion = ion2
                    source_trap = ion2_trap
                    dest_trap = ion1_trap
                    #print("ERROR: they can't both be in ancillas")
                    assert(ion1 not in ancillas)
                else:
                    print("ERROR: neither of the moving ions are in ancillas?! This means a CX between data?")
                    assert(0)
                clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, fire_time, barrier_time=specified_time)
                self.add_gate_op(clk, dest_trap, gate, ion1, ion2)
            else:
                print("GOT TO REBALANCING")
                self.rebalance_counter += 1
                #This is for the rebalancing case, trap_ids compute till this point may be stale
                self.schedule_gate_enhancement(gate, ancillas, specified_time=new_fin_time)

    #Checks and rebalances the machine if necessary using MCMF
    def rebalance_traps(self, focus_traps, fire_time):
        m = self.machine
        ss = self.sys_state
        t1 = focus_traps[0]
        t2 = focus_traps[1]
        excess_cap1 = m.traps[t1].capacity - len(ss.trap_ions[t1])
        excess_cap2 = m.traps[t2].capacity - len(ss.trap_ions[t2])
        need_rebalance = False

        ftr = FreeTrapRoute(m, ss)
        status12, route12 = ftr.find_route(t1, t2)
        status21, route21 = ftr.find_route(t2, t1)
        self.sys_state.print_state()
        
        #If both traps are full
        if excess_cap1 == 0 and excess_cap2 == 0:
            need_rebalance = True
        else:
           #If no route exists either way
            if status12 == 1 and status21 == 1:
                need_rebalance = True
        if need_rebalance:
            #print("Rebalance procedure", "clk=", fire_time)
            finish_time = self.do_rebalance_traps(fire_time)
            #print("Rebalance procedure", "clk=", finish_time)
        if need_rebalance:
            return 1, finish_time
        else:
            return 0, fire_time

    def do_rebalance_traps(self, fire_time):
        self.count_rebalance += 1
        rebal = RebalanceTraps(self.machine, self.sys_state)
        flow_dict = rebal.clear_all_blocks()
        shuttle_graph = nx.DiGraph()
        used_flow = {}
        clk = fire_time
        for i in flow_dict:
            for j in flow_dict[i]:
                if flow_dict[i][j] != 0:
                    shuttle_graph.add_edge(i, j, weight=flow_dict[i][j])
                    used_flow[(i, j)] = 0
        fin_time = fire_time
        for node in shuttle_graph.nodes():
            if shuttle_graph.in_degree(node) == 0 and type(node) == Trap:
                #print("Starting computation from", node.show())
                updated_graph = shuttle_graph.copy()
                for edge in used_flow:
                    if used_flow[edge] == updated_graph[edge[0]][edge[1]]['weight']:
                        updated_graph.remove_edge(edge[0], edge[1])
                T = nx.dfs_tree(updated_graph, source=node)
                for tnode in T:
                    if T.out_degree(tnode) == 0:
                        shuttle_route = nx.shortest_path(T, node, tnode)
                        break
                for i in range(len(shuttle_route)-1):
                    e0 = shuttle_route[i]
                    e1 = shuttle_route[i+1]
                    if (e0, e1) in used_flow:
                        used_flow[(e0, e1)] += 1
                    elif (e1, e0) in used_flow:
                        used_flow[(e1, e0)] += 1
                moving_ion = self.sys_state.trap_ions[node.id][0]
                ion_time, _ = self.ion_ready_info(moving_ion)
                fire_time = max(fire_time, ion_time)
                #print("moving", moving_ion, "along path")
                #for item in shuttle_route:
                #    print(item.show())
                #print('path end')
                #Fire a shuttle along this route, move first ion in the source trap for now
                fin_time_new = self.fire_shuttle(node.id, tnode.id, moving_ion, fire_time, barrier_time=-2, route=shuttle_route) #this is ok because in rebalance it will never consider barriers, just last ion move
                fin_time = max(fin_time, fin_time_new)
        return fin_time
    """
    def run(self):
        self.gates = nx.topological_sort(self.ir)
        sorted_nodes = list(nx.topological_sort(self.ir))
        cnt = 0
        #self.sys_state.print_state()
        for g in sorted_nodes:
            self.schedule_gate(g)
            print("Scheduled", cnt, "of ", len(sorted_nodes), "gates")
            cnt += 1
        #self.schedule.print_events()
        #self.sys_state.print_state()
        """
    def run(self):
        sorted_nodes = list(nx.topological_sort(self.ir))
        print("gate info", self.gate_info)
        print("sorted nodes", sorted_nodes)
        self.gates = iter(sorted_nodes)  # Keep the original behavior of a generator if needed

        for g in tqdm(sorted_nodes, desc="Scheduling gates", unit="gate"):
            self.schedule_gate(g)

    #LEFT OFF HERE 5/28 MORNING, NEED TO IMPLEMENT RUN_GRIP SCHEDULING SO THAT THE TIME CAN BE WAY REDUCED. BUT THIS MIGHT
    #NOT BE THE ONLY REASON THERE IS A BUG. 15DATAPERTRAP SHOWS LESS OVERALL LINES THAN BASELINE BUT WAY MORE TIME???
    # ABOVE PRINTING OF GATE INFO AND GATES TO HELP FILL OUT THE METHOD BELOW 
    def run_GRIP_DFS(self, global_cx_arr, stab_to_ancilla_mapping):
        #self.gates shape: [0, 3, 6, 9, 10, 12, 15, 18, 21, 22, 24, 27...]
         #self.gate_info shape: 0 --> [5, 450] where the key is a gate in the list above left off here 6/19/25
        ancillas = []
        for x in stab_to_ancilla_mapping.keys():
            ancilla = stab_to_ancilla_mapping[x]
            ancillas.append(ancilla)
        sorted_gates = list(nx.topological_sort(self.ir))
        self.gates = iter(sorted_gates)
        start_time = 0
        print("global cx arr", global_cx_arr)
        self.sys_state.print_state()
        
        
        """
        if (len(global_cx_arr) == 2): #meaning its a BB code, arbitrary CSS case of 2 layers.
            intermediate_ancilla_x = {}
            intermediate_ancilla_z= {}
            print("Assuming THIS CODE IS BB, SO SHUFFLING THE SCHEDULE")
            ##HAVE TO CORRECT THIS TO BE 144 ANCILLA NOT 72 AND 72
            x = global_cx_arr[0]
            z = global_cx_arr[1]
            i = 0
            print("global cx array right here", global_cx_arr)
            print("length of x", len(x))
            print("x", x)
            print("z", z)
            print("stab to ancilla mapping", stab_to_ancilla_mapping)
            nonempty = 0
            while (i < len(x)):
                stabilizer = x[i]
                if (len(stabilizer) > 0):
                    original_ancilla = stab_to_ancilla_mapping[i]
                    intermediate_ancilla_x[original_ancilla] = stabilizer
                    nonempty += 1
                i += 1
            print("i", i)
            
            i = 0
            while (i < len(z)):
                stabilizer = z[i]
                if (len(stabilizer) > 0):
                    original_ancilla = stab_to_ancilla_mapping[i]
                    intermediate_ancilla_z[original_ancilla] = stabilizer
                i += 1
            print("intermediate ancilla x", intermediate_ancilla_x)
            print("intermediate ancilla z", intermediate_ancilla_z)
            assert(len(intermediate_ancilla_x) == len(intermediate_ancilla_z))
            new_stab_to_ancilla = {}
            real_global_cx_arr = []
            
            random.seed(52)
            random.shuffle(x)
            random.shuffle(z)
            for i in range(len(x)):
                stab = x[i]
                if (len(stab) > 0):
                    for key in intermediate_ancilla_x.keys():
                        if (intermediate_ancilla_x[key] == stab): #assumes no duplicate stabilizers WITHIN X, which is a fair assumption
                            assert(i not in new_stab_to_ancilla.keys())
                            new_stab_to_ancilla[i] = key
            print("new stab to ancilla after x", new_stab_to_ancilla)
            print("length", len(new_stab_to_ancilla))
            offset = len(x)
            for i in range(len(z)):
                stab = z[i]
                if (len(stab) > 0):
                    for key in intermediate_ancilla_z.keys():
                        if (intermediate_ancilla_z[key] == stab): #assumes no duplicate stabilizers WITHIN X, which is a fair assumption
                            assert(i + offset not in new_stab_to_ancilla.keys()) #shouldn't be overrwriting
                            new_stab_to_ancilla[i + offset] = key
            print("new stab to ancilla before renaming", new_stab_to_ancilla)
            print("new global cx", global_cx_arr)
            assert(len(new_stab_to_ancilla.keys()) == len(stab_to_ancilla_mapping))
            stab_to_ancilla_mapping = new_stab_to_ancilla
            print("new stab to ancilla", stab_to_ancilla_mapping)
            print(len(new_stab_to_ancilla.keys()))
            assert(len(new_stab_to_ancilla))
        """
        
            
        gate_set = set()
        print("global cx arr", global_cx_arr)
        for layer_num in range(len(global_cx_arr)): #can also incorporate BARRIER logic after each layer with the specified_time keyword
            layer = global_cx_arr[layer_num]
            for g in range(len(layer)):
                stab = layer[g]
                if (len(stab) > 0):
                    print("stabilizer", stab)
                    print("stab to mapping", stab_to_ancilla_mapping)
                    ancilla = stab_to_ancilla_mapping[g] #untabbed from else statement
                    #print("ancilla", ancilla)
                    for data in stab: #this is a kind of depth first approach within each time step. This may be bad! May want to do breadth first doing all first microsteps in each timestep in parallel, and then proceed to next
                        gate = self.find_gate(data, ancilla)
                        assert((data,ancilla) not in gate_set)
                        gate_set.add((data,ancilla))
                        print("scheduling gate", gate)
                        self.schedule_gate_enhancement(gate, ancillas, specified_time=start_time)
                    print("done with schedulingstabilizer", layer[g])
            start_time = self.schedule.last_comm_event_time()
            print("Last event for layer 1", start_time)
            #exit()
        
    def run_GRIP(self, global_cx_arr, stab_to_ancilla_mapping): #uses BFS policy for less collisions
        ancillas = []
        for x in stab_to_ancilla_mapping.keys():
            ancilla = stab_to_ancilla_mapping[x]
            ancillas.append(ancilla)
        sorted_gates = list(nx.topological_sort(self.ir))
        self.gates = iter(sorted_gates)
        start_time = 0
        print("global cx arr", global_cx_arr)
        self.sys_state.print_state()
    
        
            
        gate_set = set()
        print("global cx arr", global_cx_arr)
        for layer_num in range(len(global_cx_arr)): #can also incorporate BARRIER logic after each layer with the specified_time keyword
            layer = global_cx_arr[layer_num]
            highest_depth_stab = max(layer, key=len)
            for index in range(len(highest_depth_stab)):
                for g in range(len(layer)):
                    stab = layer[g]
                    if (len(stab) > 0):
                        print("stabilizer", stab)
                        print("stab to mapping", stab_to_ancilla_mapping)
                        ancilla = stab_to_ancilla_mapping[g] #untabbed from else statement
                        #print("ancilla", ancilla)
                        if (index < len(stab)):
                            data = stab[index]
                            gate = self.find_gate(data, ancilla)
                            assert((data,ancilla) not in gate_set)
                            gate_set.add((data,ancilla))
                            print("scheduling gate", gate)
                            self.schedule_gate_enhancement(gate, ancillas, specified_time=start_time)
                            print("done with schedulingstabilizer", layer[g])
                        else:
                            pass
                            #print("uh oh, non homogenous stabilizer depth. Might break code so wanted to stop and inform")
                            #assert(0)
            start_time = self.schedule.last_comm_event_time()
            print("Last event for layer 1", start_time)
    ###STOP IMPLEMENTING RUN_GRIP, FIRST CHECK TO SEE IF A SINGLE STABILIZER GIVEN INITIAL MAPPING AND GRAND CX COMPLETES IN THE AMOUNT OF TIME IT SHOULD
    #PRINT OUT GRAND CX AND INIT MAPPING 
    ##NOTE DEBUGGING WITH STABILIZER NUMBR 12 BASE ANCILLA + EXTRA MEANS [3,5] ON 453 IN FIRST TIME STEP


    def run_GRIP_stabilizer(self, global_cx_arr, stab_to_ancilla_mapping): #uses BFS policy for less collisions
        ancillas = []
        for x in stab_to_ancilla_mapping.keys():
            ancilla = stab_to_ancilla_mapping[x]
            ancillas.append(ancilla)
        sorted_gates = list(nx.topological_sort(self.ir))
        self.gates = iter(sorted_gates)
        start_time = 0
        print("global cx arr", global_cx_arr)
        self.sys_state.print_state()
    
        
        #gate_set = set()
        print("global cx arr", global_cx_arr)
        for layer_num in range(len(global_cx_arr)): #can also incorporate BARRIER logic after each layer with the specified_time keyword
            layer = global_cx_arr[layer_num]
            highest_depth_stab = max(layer, key=len)
            for index in range(len(highest_depth_stab)):
                for g in range(len(layer)):
                    stab = layer[g]
                    if (len(stab) > 0):
                        print("stabilizer", stab)
                        print("stab to mapping", stab_to_ancilla_mapping)
                        ancilla = stab_to_ancilla_mapping[g] #untabbed from else statement
                        #print("ancilla", ancilla)
                        if (index < len(stab)):
                            data = stab[index]
                            gate = self.find_gate_stabilizer(data, ancilla)
                            #assert((data,ancilla) not in gate_set)
                            #gate_set.add((data,ancilla))
                            print("scheduling gate", gate)
                            self.schedule_gate_enhancement(gate, ancillas, specified_time=start_time)
                            print("done with scheduling stabilizer", layer[g])
                        else:
                            pass
                            #print("uh oh, non homogenous stabilizer depth. Might break code so wanted to stop and inform")
                            #assert(0)
            start_time = self.schedule.last_comm_event_time()
            print("Last event for layer 1", start_time)
    ###STOP IMPLEMENTING RUN_GRIP, FIRST CHECK TO SEE IF A SINGLE STABILIZER GIVEN INITIAL MAPPING AND GRAND CX COMPLETES IN THE AMOUNT OF TIME IT SHOULD
    #PRINT OUT GRAND CX AND INIT MAPPING 
    ##NOTE DEBUGGING WITH STABILIZER NUMBR 12 BASE ANCILLA + EXTRA MEANS [3,5] ON 453 IN FIRST TIME STEP

    def find_gate(self, data, ancilla):
        found = False
        print("Gate info",self.gate_info)
        print("Data and ancilla", data, ancilla)
        for x in self.gate_info.keys():
            if (self.gate_info[x][0] == data and self.gate_info[x][1] == ancilla):
                assert(found == False)
                gate = x
                found= True
            elif (self.gate_info[x][1] == data and self.gate_info[x][0] == ancilla):
                assert(found == False)
                gate = x
                found = True
        assert(found == True)
        return gate
    
    def find_gate_stabilizer(self, data, ancilla):
        found = False
        print("Gate info",self.gate_info)
        print("Data and ancilla", data, ancilla)
        for x in self.gate_info.keys():
            if (self.gate_info[x][0] == data and self.gate_info[x][1] == ancilla):
                #assert(found == False)
                gate = x
                found= True
            elif (self.gate_info[x][1] == data and self.gate_info[x][0] == ancilla):
                #assert(found == False)
                gate = x
                found = True
        #assert(found == True)
        return gate


