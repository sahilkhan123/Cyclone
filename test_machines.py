from machine import Machine, Trap, Segment, MachineParams
import networkx as nx
import matplotlib.pyplot as plt

#ISCA Test machines Begin
def test_trap_2x3(capacity, mparams):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(6)]
    j = [m.add_junction(i) for i in range(3)]
    m.add_segment(0, t[0], j[0], 'R')
    m.add_segment(1, t[1], j[1], 'R')
    m.add_segment(2, t[2], j[2], 'R')
    m.add_segment(3, t[3], j[2], 'L')
    m.add_segment(4, t[4], j[1], 'L')
    m.add_segment(5, t[5], j[0], 'L')
    m.add_segment(6, j[0], j[1])
    m.add_segment(6, j[1], j[2])
    return m

def make_circle_machine_length_n(capacity, mparams, n):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(n)]
    j = [m.add_junction(i) for i in range(n)]

    #prints should resemble the above structures exactly
    for i in range(n):
        m.add_segment(i, t[i], j[i], 'R')
        #print("added " + str(i) + ", t[" +str(i) + "], j[" + str(i) + "], " + "R")
    for i in range(n):
        if (i == 0):
            m.add_segment(n + i, t[i], j[n-1], 'L') #links the circle by connecting tail to head
            #print("added " + str(n+i) + ", t[" +str(i) + "], j[" + str(n-1) + "], " + "L")
        else:
            m.add_segment(n + i, t[i], j[i-1], 'L')
            #print("added " + str(n+i) + ", t[" +str(i) + "], j[" + str(i-1) + "], " + "L")
    
    return m

def make_linear_machine(zones, capacity, mparams):
    m = Machine(mparams)
    traps = []
    junctions = []
    for i in range(zones):
        traps.append(m.add_trap(i, capacity))
    for i in range(zones-1):
        junctions.append(m.add_junction(i))
    for i in range(zones-1):
        m.add_segment(2*i,   traps[i], junctions[i], 'R') #t_i ---- j_i ---- t_i+1
        m.add_segment(2*i+1, traps[i+1], junctions[i], 'L')
    return m

def make_single_hexagon_machine(capacity, mparams):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(6)]
    j = [m.add_junction(i) for i in range(6)]
    m.add_segment(0, t[0], j[0], 'R')
    m.add_segment(1, t[1], j[1], 'R')
    m.add_segment(2, t[2], j[2], 'R')
    m.add_segment(3, t[3], j[3], 'R')
    m.add_segment(4, t[4], j[4], 'R')
    m.add_segment(5, t[5], j[5], 'R')
    m.add_segment(6, t[0], j[5], 'L')
    m.add_segment(7, t[1], j[0], 'L')
    m.add_segment(8, t[2], j[1], 'L')
    m.add_segment(9, t[3], j[2], 'L')
    m.add_segment(10, t[4], j[3], 'L')
    m.add_segment(11, t[5], j[4], 'L')
    return m

#ISCA Test machines End

def mktrap4x2(capacity):
    m = Machine()
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    j0 = m.add_junction(0)
    j1 = m.add_junction(1)
    m.add_segment(0, t0, j0)
    m.add_segment(1, t1, j0)
    m.add_segment(2, t2, j1)
    m.add_segment(3, t3, j1)
    m.add_segment(4, j0, j1)
    return m

def mktrap_4star(capacity):
    m = Machine()
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    j0 = m.add_junction(0)
    m.add_segment(0, t0, j0)
    m.add_segment(1, t1, j0)
    m.add_segment(2, t2, j0)
    m.add_segment(3, t3, j0)
    return m

def mktrap6x3(capacity):
    m = Machine()
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    t4 = m.add_trap(4, capacity)
    t5 = m.add_trap(5, capacity)
    j0 = m.add_junction(0)
    j1 = m.add_junction(1)
    j2 = m.add_junction(2)
    m.add_segment(0, t0, j0)
    m.add_segment(1, t1, j0)
    m.add_segment(2, t2, j1)
    m.add_segment(3, t3, j1)
    m.add_segment(4, t4, j2)
    m.add_segment(5, t5, j2)
    m.add_segment(6, j0, j1)
    m.add_segment(7, j1, j2)
    return m

def mktrap8x4(capacity):
    m = Machine()
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    t4 = m.add_trap(4, capacity)
    t5 = m.add_trap(5, capacity)
    t6 = m.add_trap(6, capacity)
    t7 = m.add_trap(7, capacity)

    j0 = m.add_junction(0)
    j1 = m.add_junction(1)
    j2 = m.add_junction(2)
    j3 = m.add_junction(3)

    m.add_segment(0, t0, j0)
    m.add_segment(1, t1, j0)
    m.add_segment(2, t2, j1)
    m.add_segment(3, t3, j1)
    m.add_segment(4, t4, j2)
    m.add_segment(5, t5, j2)
    m.add_segment(6, t6, j3)
    m.add_segment(7, t7, j3)

    m.add_segment(8, j0, j1)
    m.add_segment(9, j1, j2)
    m.add_segment(10, j2, j3)
    return m



def make_3x3_grid(capacity, mparams):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(9)]
    j = [m.add_junction(i) for i in range(6)]
    #add right segments
    m.add_segment(0, t[0], j[0])
    m.add_segment(1, t[1], j[1])
    m.add_segment(2, t[2], j[2])
    m.add_segment(3, t[3], j[3])
    m.add_segment(4, t[4], j[4])
    m.add_segment(5, t[5], j[5])
    #add left segments
    m.add_segment(6, t[3], j[0])
    m.add_segment(7, t[4], j[1])
    m.add_segment(8, t[5], j[2])
    m.add_segment(9,  t[6], j[3])
    m.add_segment(10, t[7], j[4])
    m.add_segment(11, t[8], j[5])

    #add vertical segments
    m.add_segment(12, j[0], j[1])
    m.add_segment(13, j[1], j[2])
    m.add_segment(14, j[3], j[4])
    m.add_segment(15, j[4], j[5])
    return m

def make_nxn_grid(N, capacity, mparams):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(N*N)]
    j = [m.add_junction(i) for i in range(N*(N-1))]
    s_number = 0
    for i in range(N * (N-1)):
        m.add_segment(i, t[i], j[i])
        print("add segment", i, i, i)
        s_number += 1
    for i in range(N, N*N):
        m.add_segment(s_number, t[i], j[i - N])
        print("add segment", s_number, i, i - N)
        s_number += 1
    for x in range(N - 1):
        start_num = x*N 
        for i in range(N - 1):
            m.add_segment(s_number, j[i + start_num], j[i+1+start_num])
            print("add segment", s_number, i + start_num, i+start_num+1)
            s_number += 1
    return m

def make_junction_network(N, capacity, mparams):
    #N = 5 for example
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(N*4)]
    j = [m.add_junction(i) for i in range(N*N)]
    j_index = 0
    s_number = 0
    for i in range(N):
        m.add_segment(s_number, t[i], j[j_index])
        s_number += 1
        for k in range(N - 1):
            m.add_segment(s_number, j[j_index + k], j[j_index + k + 1])
            s_number += 1
        m.add_segment(s_number, t[N + i], j[N-1 + j_index])
        j_index += N
        s_number += 1
    
    j_index = 0
    for i in range(N): #flipped vertically now, start at 2N because # 0-2N of the traps have already been used, the next one is exactly the number 2N
        m.add_segment(s_number, t[2*N + i], j[i])
        s_number += 1
        for k in range(N - 1):
            m.add_segment(s_number, j[k*N + i], j[(k+1)*N + i])
            s_number += 1
        m.add_segment(s_number, t[i + 3*N], j[i + (N-1)*N],)
        s_number += 1
    return m


def old_make_alternate_grid(N, capacity, mparams):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(N*N)]
    j = [m.add_junction(i) for i in range(N*N)]
    s_number = 0
    switch = 0
    offset = 0
    for row in range(N):
        for inner in range(N):
            m.add_segment(s_number, t[inner+offset], j[inner+offset])
            s_number += 1
        for inner in range(N - 1):
            if (switch % 2 == 0):
                m.add_segment(s_number, t[inner+offset], j[inner+offset+1])
                s_number += 1
            else:
                m.add_segment(s_number, t[inner+offset+1], j[inner+offset])
                s_number += 1
        offset += N
        switch += 1
    row_offset = 0
    for row in range(N - 1):
        for index in range(N):
            m.add_segment(s_number, t[index+N+row_offset], j[index+row_offset])
            s_number +=1
            m.add_segment(s_number, t[index+row_offset], j[index+N+row_offset])
            s_number += 1
        row_offset += N
    
def make_alternate_grid(N, capacity, mparams):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range((N*(N+1)) + (N*(N+1)))]
    j = [m.add_junction(i) for i in range((N+1)*(N+1))]
    s_number = 0
    switch = 0
    offset = 0
    for row in range(N + 1):
        for inner in range(N):
            m.add_segment(s_number, t[inner+offset], j[inner+offset])
            s_number += 1
        for inner in range(N - 1):
            m.add_segment(s_number, t[inner+offset], j[inner+offset+1])
            s_number += 1
         
        offset += N
        switch += 1
    #add last column of junctions
    base_junction_number = N*(N+1)
    base_marker = base_junction_number
    N_offset = 0
    for i in range(N+1):
        m.add_segment(s_number, t[N - 1 + N_offset], j[base_junction_number])
        s_number += 1
        N_offset += N
        base_junction_number += 1
    #now all the horizontal part is complete


    #for next part, set the counter of trap number to n*(n+1)
    base_trap_number = N*(N+1)
    #get inner square here of vertical, then save the last connecting vertical trap edge for last later
    vert_offset = 0
    for row in range(N):
        for inside in range(N):
            m.add_segment(s_number, t[base_trap_number], j[inside + vert_offset])
            s_number += 1
            m.add_segment(s_number, t[base_trap_number], j[inside + vert_offset + N])
            s_number += 1
            base_trap_number += 1
        vert_offset += N
    
    for i in range(N):
        m.add_segment(s_number, t[base_trap_number], j[base_marker + i])
        s_number += 1
        m.add_segment(s_number, t[base_trap_number], j[base_marker + i + 1])
        s_number += 1
        base_trap_number += 1


    #and now for last column of traps
    # row_offset = 0
    # for row in range(N - 1):
    #     for index in range(N):
    #         m.add_segment(s_number, t[index+N+row_offset], j[index+row_offset])
    #         s_number +=1
    #         m.add_segment(s_number, t[index+row_offset], j[index+N+row_offset])
    #         s_number += 1
    #     row_offset += N



    """
    labels = {node: node.id for node in m.graph.nodes}
    nx.draw_kamada_kawai(m.graph, labels=labels, with_labels=True, font_weight='bold', node_color='red', node_size=700)
    plt.show()
    """
    

    return m

def make_nxn_grid_WRONG(N, capacity, mparams):
    m = Machine(mparams)
    
    traps = [[m.add_trap(i * N + j, capacity) for j in range(N)] for i in range(N)]
    junctions_h = [[m.add_junction(i * (N - 1) + j) for j in range(N - 1)] for i in range(N)]  # horizontal junctions
    junctions_v = [[m.add_junction(i * N + j) for j in range(N)] for i in range(N - 1)]        # vertical junctions
    
    seg_id = 0

    # Connect each trap to the horizontal junctions (rightward)
    for i in range(N):
        for j in range(N - 1):
            m.add_segment(seg_id, traps[i][j], junctions_h[i][j])
            seg_id += 1
            m.add_segment(seg_id, traps[i][j + 1], junctions_h[i][j])
            seg_id += 1

    # Connect each trap to the vertical junctions (downward)
    for i in range(N - 1):
        for j in range(N):
            m.add_segment(seg_id, traps[i][j], junctions_v[i][j])
            seg_id += 1
            m.add_segment(seg_id, traps[i + 1][j], junctions_v[i][j])
            seg_id += 1

    # Optionally connect junctions horizontally and vertically (junction-junction links)
    for i in range(N):
        for j in range(N - 2):
            m.add_segment(seg_id, junctions_h[i][j], junctions_h[i][j + 1])
            seg_id += 1
    for i in range(N - 2):
        for j in range(N):
            m.add_segment(seg_id, junctions_v[i][j], junctions_v[i + 1][j])
            seg_id += 1

    return m

def make_9trap(capacity):
    m = Machine(alpha=0.005, inter_ion_dist=1, split_factor=5.0, move_factor=1.0)
    t = [m.add_trap(i, capacity) for i in range(9)]
    j = [m.add_junction(i) for i in range(9)]

    m.add_segment(0, t[0], j[0])
    m.add_segment(1, t[1], j[1])
    m.add_segment(2, t[2], j[2])

    m.add_segment(3, t[3], j[2])
    m.add_segment(4, t[4], j[5])
    m.add_segment(5, t[5], j[8])

    m.add_segment(6, t[6], j[8])
    m.add_segment(7, t[7], j[7])
    m.add_segment(8, t[8], j[6])

    m.add_segment(9, j[0], j[1])
    m.add_segment(10, j[0], j[3])
    m.add_segment(11, j[3], j[6])
    m.add_segment(12, j[3], j[4])
    m.add_segment(13, j[6], j[7])
    m.add_segment(14, j[1], j[4])
    m.add_segment(15, j[1], j[2])
    m.add_segment(16, j[4], j[7])
    m.add_segment(17, j[4], j[5])
    m.add_segment(18, j[7], j[8])
    m.add_segment(19, j[2], j[5])
    m.add_segment(20, j[5], j[8])
    return m
