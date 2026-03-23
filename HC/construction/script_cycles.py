import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pyvista as pv
import scipy.spatial.distance as sd
import scipy.sparse.csgraph as sps
from script_analysis import *



def count_cycles(Pos,Types,cube=14,threshold_Si=2,periodic=True,Lims=None):
    import networkx as nx
    lx,ly,lz = np.min(Pos,axis=0)
    Pos = Pos - np.array([0,0,lz])
    lz = 0
    Lx,Ly,Lz = np.max(Pos,axis=0)
    if not type(Lims) is None:
        Lz = Lims[2][1] - Lims[2][0]

    cube_dist = cube/2

    Nx = int((Lx - lx) // cube_dist + 1)
    Ny = int((Ly - ly) // cube_dist + 1)
    Nz = int((Lz - lz) // cube_dist + 1)

    Pos_added = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Si_or = np.sum(Types==1)

    ind_pos = np.arange(len(Pos[Types==1]))

    if periodic:
        Pos_add_z = Pos[:,2] > (Lz - threshold_Si)
        Pos_add_Lz = Pos[Pos_add_z] - np.array([[0,0,Lz]])

        Pos_added = np.append(Pos_added,Pos_add_Lz,axis=0)

        Types_added = np.append(Types_added,Types[Pos_add_z],axis=0)

        Pos_ind_si = Pos[Types==1][:,2] > (Lz - threshold_Si)
        ind_ext = ind_pos[Pos_ind_si]
        ind_pos = np.append(ind_pos,ind_ext)

    Num_Si = np.sum(Types_added==1)

    Cycles = []
    Cycles_mem = []
    L_cycles = []
    A,B,C,D = 0,0,0,0

    ind_tot = np.arange(len(Pos_added[Types_added==1]))

    # print(Nx,Ny,Nz)
    import time
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                a = time.time()

                Pos_trunc_x = (Pos_added[:,0] >= ((x-1)*cube_dist + lx)) * (Pos_added[:,0] <= ((x+1)*cube_dist + lx))
                Pos_trunc_y = (Pos_added[:,1] >= ((y-1)*cube_dist + ly)) * (Pos_added[:,1] <= ((y+1)*cube_dist + ly))
                Pos_trunc_z = (Pos_added[:,2] >= ((z-1)*cube_dist + lz)) * (Pos_added[:,2] <= ((z+1)*cube_dist + lz))
                Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                Pos_trunc = Pos_added[Pos_trunc_ind]
                Types_trunc = Types_added[Pos_trunc_ind]
                b=time.time()

                if (Types_trunc == 1).any() and (Types_trunc==2).any():

                    Bonds = compute_bonds_graph(Pos_trunc,Types_trunc,cube=cube,periodic=False,Lims=Lims)
                    ind_trunc = ind_tot[Pos_trunc_ind[Types_added==1]]



                    c=time.time()
                    G = nx.from_numpy_array(Bonds)
                    L = nx.minimum_cycle_basis(G)
                    # L = nx.cycle_basis(G)
                    # L = nx.simple_cycles(G)
                    d = time.time()
                    for cycle in L:
                        Cycle_l = []
                        for ind in cycle:
                            Cycle_l.append(ind_pos[ind_trunc[ind]])
                            Cycle_sort = Cycle_l[:]
                            Cycle_sort.sort()

                        if not Cycle_sort in Cycles_mem:
                            Cycles.append(Cycle_l)
                            Cycles_mem.append(Cycle_sort)

                    e=time.time()
                    A += b-a
                    B += c-b
                    C += d-c
                    D += e-d
    f = time.time()
    # Cycles = clean_cycles(Cycles)
    # G = nx.Graph()
    # K = [int(d) for c in Cycles for d in c]
    # G.add_nodes_from(K)
    # L = np.zeros((np.max(K)+1,np.max(K)+1))
    # LL = []
    # for c in Cycles:
    #     cy2 = np.roll(c,1)
    #     for c1,c2 in zip(c,cy2):
    #         if not L[c1,c2]:
    #             L[c1,c2] = True
    #             L[c2,c1] = True
    #             LL.append([int(c1),int(c2)])
    # G.add_edges_from(LL)
    #
    #
    # Min_Basis = nx.minimum_cycle_basis(G)
    # Basis = []
    # for k in Min_Basis:
    #     Basis.append(k)
    g = time.time()

    print(A,B,C,D,g-f)
    for cycle in Cycles:
        L_cycles.append(len(cycle))
    return Cycles, L_cycles


def count_cycles_test(Pos,Types,cube=14,threshold_Si=2,periodic=True,Lims=None):
    import networkx as nx
    lx,ly,lz = np.min(Pos,axis=0)
    Pos = Pos - np.array([0,0,lz])
    lz = 0
    Lx,Ly,Lz = np.max(Pos,axis=0)
    if not type(Lims) is None:
        Lz = Lims[2][1] - Lims[2][0]

    cube_dist = cube/2

    Nx = int((Lx - lx) // cube_dist + 1)
    Ny = int((Ly - ly) // cube_dist + 1)
    Nz = int((Lz - lz) // cube_dist + 1)

    Pos_added = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Si_or = np.sum(Types==1)

    ind_pos = np.arange(len(Pos[Types==1]))

    if periodic:
        Pos_add_z = Pos[:,2] > (Lz - threshold_Si)
        Pos_add_Lz = Pos[Pos_add_z] - np.array([[0,0,Lz]])

        Pos_added = np.append(Pos_added,Pos_add_Lz,axis=0)

        Types_added = np.append(Types_added,Types[Pos_add_z],axis=0)

        Pos_ind_si = Pos[Types==1][:,2] > (Lz - threshold_Si)
        ind_ext = ind_pos[Pos_ind_si]
        ind_pos = np.append(ind_pos,ind_ext)

    Num_Si = np.sum(Types_added==1)

    Cycles = []
    Cycles_mem = []
    L_cycles = []
    A,B,C,D = 0,0,0,0

    ind_tot = np.arange(len(Pos_added[Types_added==1]))

    # print(Nx,Ny,Nz)
    Num_plan = 0
    import time
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                a = time.time()

                Pos_trunc_x_b = (Pos_added[:,0] >= ((x-1)*cube_dist - threshold_Si + lx)) * (Pos_added[:,0] <= ((x+1)*cube_dist + threshold_Si + lx))
                Pos_trunc_y_b = (Pos_added[:,1] >= ((y-1)*cube_dist - threshold_Si + ly)) * (Pos_added[:,1] <= ((y+1)*cube_dist + threshold_Si + ly))
                Pos_trunc_z_b = (Pos_added[:,2] >= ((z-1)*cube_dist - threshold_Si + lz)) * (Pos_added[:,2] <= ((z+1)*cube_dist + threshold_Si + lz))
                Pos_trunc_ind_b = Pos_trunc_x_b * Pos_trunc_y_b * Pos_trunc_z_b
                Pos_trunc_b = Pos_added[Pos_trunc_ind_b]
                Types_trunc_b = Types_added[Pos_trunc_ind_b]

                Pos_trunc_x = (Pos_added[:,0] >= ((x-1)*cube_dist + lx)) * (Pos_added[:,0] <= ((x+1)*cube_dist + lx))
                Pos_trunc_y = (Pos_added[:,1] >= ((y-1)*cube_dist + ly)) * (Pos_added[:,1] <= ((y+1)*cube_dist + ly))
                Pos_trunc_z = (Pos_added[:,2] >= ((z-1)*cube_dist + lz)) * (Pos_added[:,2] <= ((z+1)*cube_dist + lz))
                Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                Pos_trunc = Pos_added[Pos_trunc_ind]
                Types_trunc = Types_added[Pos_trunc_ind]
                b=time.time()

                if (Types_trunc == 1).any() and (Types_trunc==2).any():

                    Bonds = compute_bonds_graph(Pos_trunc,Types_trunc,cube=cube,periodic=False,Lims=Lims)
                    Num_Bonds = compute_bonds(Pos_trunc,Types_trunc,periodic=False,Lims=Lims)[1]

                    ind_trunc = ind_tot[Pos_trunc_ind[Types_added==1]]
                    # Bonds_b = compute_bonds_graph(Pos_trunc_b,Types_trunc_b,cube=cube+threshold_Si,periodic=False,Lims=Lims)
                    Num_Bonds_b = compute_bonds(Pos_trunc_b,Types_trunc_b,periodic=False,Lims=Lims)[1]

                    # Num_Bonds_b = np.sum(Bonds_b,axis=0)

                    c=time.time()
                    G = nx.from_numpy_array(Bonds)


                    # L = nx.minimum_cycle_basis(G)
                    L = nx.cycle_basis(G)
                    d = time.time()
                    for cycle in L:
                        Cycle_l = []
                        is_kept = True
                        for ind in cycle:
                            ind_all = np.argmax((np.cumsum(Pos_trunc_ind[Types_added==1])-1) == ind)
                            ind_b = (np.cumsum(Pos_trunc_ind_b[Types_added==1])-1)[ind_all]

                            # print(np.sum(Bonds_b))
                            # print(np.shape(Num_Bonds),np.shape(Num_Bonds_b))
                            # print(ind,ind_all,ind_b,np.shape(Bonds),np.shape(Bonds_b))

                            if Num_Bonds_b[ind_b] != Num_Bonds[ind]:
                                is_kept=False
                                break

                            Cycle_l.append(ind_pos[ind_trunc[ind]])
                            Cycle_sort = Cycle_l[:]
                            Cycle_sort.sort()

                        if is_kept and not Cycle_sort in Cycles_mem:
                            Cycles.append(Cycle_l)
                            Cycles_mem.append(Cycle_sort)

                    e=time.time()
                    A += b-a
                    B += c-b
                    C += d-c
                    D += e-d
        # print(A,B,C,D,Nz)
    f = time.time()
    # Cycles = clean_cycles(Cycles)
    # G = nx.Graph()
    # K = [int(d) for c in Cycles for d in c]
    # G.add_nodes_from(K)
    # L = np.zeros((np.max(K)+1,np.max(K)+1))
    # LL = []
    # for c in Cycles:
    #     cy2 = np.roll(c,1)
    #     for c1,c2 in zip(c,cy2):
    #         if not L[c1,c2]:
    #             L[c1,c2] = True
    #             L[c2,c1] = True
    #             LL.append([int(c1),int(c2)])
    # G.add_edges_from(LL)
    #
    #
    # Min_Basis = nx.minimum_cycle_basis(G)
    # Basis = []
    # for k in Min_Basis:
    #     Basis.append(k)
    g = time.time()

    print(A,B,C,D,g-f)
    for cycle in Cycles:
        L_cycles.append(len(cycle))
    return Cycles, L_cycles

def plot_cycles(L_cycles,cube=0,s=3.5):
    # fig,ax = plt.subplots()
    fig,ax = plt.subplots(figsize=(8,6),dpi=200)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['svg.fonttype'] = 'none'
    max_graph = np.max(L_cycles) + 1
    maxx = max_graph


    #Max cycle corresponding to length, length is divided by 2 hence the following formula
    if cube != 0:
        max_cycle = np.pi / (np.arcsin(s/cube))
        max_cycle = int(max_cycle)
        maxx = max(max_graph,max_cycle)
        plt.axvline(max_cycle,linestyle="dashed",color="blue",linewidth=3,label="Precision Limit")

    purple = np.array([96,25,255])/255
    dark_purple = np.array([56,20,180])/255
    dark_dark_purple = np.array([34,10,120])/255



    plt.hist(L_cycles,bins=maxx-1,range=(1,maxx),color=purple,edgecolor=dark_purple,linewidth=1)
    plt.xticks([k+1/2 for k in range(1,maxx)],[k for k in range(1,maxx)])
    # plt.legend()
    plt.xticks(color=purple)
    plt.yticks(color=purple)
    plt.ylabel("Number of cycles",color=dark_purple)
    plt.xlabel("Length of cycles",color=dark_purple)
    # plt.show()
    plt.savefig("Cycles.png")


def clean_cycles_basis(Cycles):
    Cycles_S = [True] * len(Cycles)
    import networkx as nx
    G = nx.Graph()
    K = [int(d) for c in Cycles for d in c]
    G.add_nodes_from(K)
    L = np.zeros((np.max(K)+1,np.max(K)+1))
    D = []
    for c in Cycles:
        cy2 = np.roll(c,1)
        for c1,c2 in zip(c,cy2):
            if not L[c1,c2]:
                L[c1,c2] = True
                L[c2,c1] = True
                D.append([int(c1),int(c2)])
    G.add_edges_from(D)

    # nx.draw(G,with_labels=True)
    # plt.show()
    A = nx.minimum_cycle_basis(G)
    D = []
    for k in A:
        D.append(k)


    # return Cycles
    def find_cycle_same(cycle,Sub_Cycles):
        flag_there_is = False
        for cycle_2,ind_cycle_2 in zip(Sub_Cycles,range(len(Sub_Cycles))):
            if cycle[0] in cycle_2:
                ind = cycle_2.index(cycle[0])
                cycle_2 = np.roll(cycle_2,len(cycle_2)-ind)
                if cycle[1] == cycle_2[1]:
                    cycle_same = cycle_2
                    return cycle_same,  True
                elif cycle[1] == cycle_2[-1]:
                    cycle_same = np.roll(cycle_2[::-1],1)
                    return cycle_same,  True
        return [], False

    def is_min(cycle,cycle_index,Sub_Cycles,Cycles_S):
        tot_loop = 0
        cycle_basis = [cycle]
        cycle_basis_ind = [cycle_index]

        flag_break = False
        while tot_loop <= len(cycle):
            cycle_same, flag_there_is = find_cycle_same(cycle,Sub_Cycles)
            if flag_there_is:
                cycle_basis.append(cycle_same)
                loop = 0
                for c1,c2 in zip(cycle,cycle_same):
                    if c1!=c2:
                        break
                    loop += 1
                    tot_loop +=1
                cycle = np.roll(cycle,-loop+1)
                tot_loop-=1
                print(cycle,tot_loop,len(cycle))

            else:
                flag_break = True
                break
        # e=f

        if not flag_break:
            print("No break, cycle : {}".format(cycle_index))
            print(cycle_basis)
            Cycles_S[cycle_index] = False

    for cycle,index_cycle in zip(Cycles,range(len(Cycles))):
        Cycles_same = []
        Sub_Cycles = []


        for cycle_2,index_cycle_2 in zip(Cycles,range(len(Cycles))):
            if Cycles_S[index_cycle_2] and index_cycle != index_cycle_2:
                for ind in cycle:
                    if ind in cycle_2:
                        Sub_Cycles.append(cycle_2)
                        break
        # print(Sub_Cycles)
        is_min(cycle,index_cycle,Sub_Cycles,Cycles_S)
    Cycles_clean = []
    for cycle,is_kept in zip(Cycles,Cycles_S):
        if is_kept:
            Cycles_clean.append(cycle)

    return Cycles_clean


def clean_cycles(Cycles):
    import networkx as nx
    Cycles_S = [True] * len(Cycles)
    Cycles_set = []
    for cycle in Cycles:
        Cycles_set.append(set(cycle))

    def min_basis(Sub_Cycles):
        G = nx.Graph()
        K = [int(d) for c in Sub_Cycles for d in c]
        G.add_nodes_from(K)
        # print(G.number_of_nodes())
        L = np.zeros((np.max(K)+1,np.max(K)+1))
        D = []
        for c in Sub_Cycles:
            cy2 = np.roll(c,1)
            for c1,c2 in zip(c,cy2):
                if not L[c1,c2]:
                    L[c1,c2] = True
                    L[c2,c1] = True
                    D.append([int(c1),int(c2)])
        G.add_edges_from(D)


        Min_Basis = nx.minimum_cycle_basis(G)
        Basis = []
        Basis_sorted = []
        for k in Min_Basis:
            Basis.append(k)
            Basis_sorted.append(set(k))
        # print("Checkin",Sub_Cycles,Basis)
        return Basis, Basis_sorted

    def find_common(Cycles,cycle,Cycles_S):
        Common_edges = []
        cycle_rolled = np.roll(cycle,1)

        for ind_a,ind_b in zip(cycle,cycle_rolled):
            for cycle_2,index_cycle_2 in zip(Cycles,range(len(Cycles))):
                if Cycles_S[index_cycle_2]:
                    if ind_a in cycle_2:
                        index_2 = cycle_2.index(ind_a)
                        if index_2 == (len(cycle_2)-1):
                            if cycle_2[-2] == ind_b:
                                Common_edges.append(cycle_2)
                                break
                        elif ind_b == cycle_2[index_2+1] or ind_b == cycle_2[index_2-1]:
                            Common_edges.append(cycle_2)
                            break
        return Common_edges

    Cycles_index = [k for k in range(len(Cycles))]
    for cycle,index_cycle in zip(Cycles,Cycles_index):
        Cycles_same = []
        Sub_Cycles = find_common(Cycles,cycle,Cycles_S)
        # print(len(Sub_Cycles))
        # for cycle_2,index_cycle_2 in zip(Cycles,range(len(Cycles))):
        #     if Cycles_S[index_cycle_2]:
        #         Sub_Cycles += find_common(Cycles,cycle_2,Cycles_S)


        # print(cycle,Sub_Cycles)
        Basis,Basis_set = min_basis(Sub_Cycles)
        # print(len(Sub_Cycles),len(Basis))

        cycle_set = set(cycle)
        # print("BB",Basis)
        if not cycle_set in Basis_set:
            # print("Remove")
            # print("check",cycle_set,Basis_set)
            # print("AAA")
            Cycles_S[index_cycle] = False
            #
            for cycle_basis in Basis:
                cycle_basis_set = set(cycle_basis)
                if not cycle_basis_set in Cycles_set:
                    # print("Add")
                    # print(cycle,Sub_Cycles,Basis)
                    Cycles.append(cycle_basis)
                    Cycles_S.append([True])
                    Cycles_index.append(max(Cycles_index)+1)
                    Cycles_set.append(set(cycle_basis))


    Cycles_clean = []
    for cycle,is_kept in zip(Cycles,Cycles_S):
        if is_kept:
            Cycles_clean.append(cycle)


    return Cycles_clean


def xor_clean(Cycle_Basis,k):
    import time
    Cycle_lin = [ind for cycle in Cycle_Basis for ind in cycle]
    max_ind = np.max(Cycle_lin)


    def convert_back(cycle,max_ind):
        cycle_pairs = []
        for ind in cycle:
            ind_m = ind%max_ind
            ind_M = ind//max_ind
            cycle_pairs.append([ind_M,ind_m])
        return cycle_pairs

    def check_cycle(cycle_sym_diff):
        cycle = convert_back(cycle_sym_diff,max_ind)
        G = nx.Graph()
        Nodes = [int(c) for pairs in cycle for c in pairs]
        G.add_nodes_from(Nodes)
        # L = np.zeros((np.max(Nodes)+1,np.max(Nodes)+1))
        # D = []
        # for c in cycle:
        #     cy2 = np.roll(c,1)
        #     for c1,c2 in zip(c,cy2):
        #         if not L[c1,c2]:
        #             L[c1,c2] = True
        #             L[c2,c1] = True
        #             D.append([int(c1),int(c2)])
        # G.add_edges_from(D)
        G.add_edges_from(cycle)
        Basis_gen = nx.cycle_basis(G)
        Basis = []
        for k in Basis_gen:
            Basis.append(k)

        if len(Basis) == 1:
            return True
        return False

    a=time.time()
    Cycle_edges = []
    for cycle in Cycle_Basis:
        cycle_r = np.roll(cycle,1)
        cycle_add = []
        for ind,ind_r in zip(cycle,cycle_r):
            ind_M, ind_m = max(ind,ind_r),min(ind,ind_r)
            cycle_add.append(ind_M*max_ind + ind_m)
        Cycle_edges.append(set(cycle_add))
    b=time.time()


    Current_Basis = Cycle_edges[:]
    list_Cycle_sym_diff = Cycle_edges[:]

    A,B = 0,0
    for num_k in range(1,k+1):

        print(len(list_Cycle_sym_diff))
        Cycles_S = [True] * len(Current_Basis)
        Cycles_index = [k for k in range(len(list_Cycle_sym_diff))]
        list_Cycle_sym_diff_compil = []
        c = time.time()
        for cycle,ind_cycle in zip(Current_Basis,Cycles_index):
            # print("Doing_",ind_cycle)
            list_Cycle_sym_diff_this_cycle = []
            for cycle_2,ind_cycle_2 in zip(list_Cycle_sym_diff,Cycles_index):
                l=time.time()
                cycle_sym_diff = cycle.symmetric_difference(cycle_2)
                m=time.time()
                A+=m-l
                list_Cycle_sym_diff_this_cycle.append(cycle_sym_diff)

                if (len(cycle_sym_diff) < len(cycle)) and check_cycle(cycle_sym_diff):
                    #if new cycle shorter
                    Cycles_S[ind_cycle] = False
                    Cycles_S.append(True)
                    Current_Basis.append(cycle_sym_diff)
                    Cycles_index.append(len(Cycles_index))
                    break #Found one better, change basis


            list_Cycle_sym_diff_compil.append(list_Cycle_sym_diff_this_cycle)
        d = time.time()

        New_Basis = []
        list_Cycle_sym_diff_new = []
        for basis,Cycles_sym_diff,is_kept in zip(Current_Basis,list_Cycle_sym_diff_compil,Cycles_S):
            if is_kept:
                New_Basis.append(basis)
                for sym_diff in Cycles_sym_diff:
                    list_Cycle_sym_diff_new.append(sym_diff)
        Current_Basis = New_Basis[:]
        list_Cycle_sym_diff = list_Cycle_sym_diff_new[:]
        e=time.time()
        print(d-c,e-d,A)




    #Clean
    f = time.time()
    Cycles_clean = []
    print(len(Cycle_Basis),len(Cycle_edges),len(Current_Basis))
    # print(Current_Basis)

    for cycle in Current_Basis:
        # print(cycle)
        cycle_pairs = convert_back(cycle,max_ind)

        cycle_b = cycle_pairs[0][:]
        pairs = [False] + [True] * (len(cycle_pairs)-1)
        index_check = 1
        while True in pairs:
            # print(pairs)
            if pairs[index_check]:
                if cycle_b[-1] in cycle_pairs[index_check]:
                    i_a, i_b = cycle_pairs[index_check]
                    if i_a == cycle_b[-1]:
                        cycle_b.append(i_b)
                    else:
                        cycle_b.append(i_a)
                    pairs[index_check] = False

            index_check += 1
            if index_check >= len(cycle_pairs):
                index_check = 1
            # c+=1
            # if c == 1000:
            #     print(pairs)
            #     print(cycle_b)
            #     print(cycle_pairs)
            #     p=dfsd



        Cycles_clean.append(cycle_b[:-1])
    g = time.time()
    print(b-a,A,B,g-f)

    return Cycles_clean




def xor_clean_rec(Cycle_Basis,k,k_long,long=12):
    import time
    Cycle_lin = [ind for cycle in Cycle_Basis for ind in cycle]
    max_ind = np.max(Cycle_lin)+1


    def convert_back(cycle,max_ind):
        # print(cycle)
        cycle_pairs = []
        for ind in cycle:
            ind_m = ind%max_ind
            ind_M = ind//max_ind
            cycle_pairs.append([ind_M,ind_m])
        return cycle_pairs

    def convert_back_lin(cycle,max_ind):
        # print(cycle)
        cycle_lin = []
        for ind in cycle:
            ind_m = ind%max_ind
            ind_M = ind//max_ind
            cycle_lin.append(ind_M)
            cycle_lin.append(ind_m)
        return set(cycle_lin)

    def check_cycle(cycle_sym_diff):
        if len(cycle_sym_diff) == 0: return False
        cycle = convert_back(cycle_sym_diff,max_ind)
        G = nx.Graph()
        Nodes = [int(c) for pairs in cycle for c in pairs]
        G.add_nodes_from(Nodes)
        G.add_edges_from(cycle)
        return (G.number_of_edges() == G.number_of_nodes()) * nx.is_connected(G)


    def compo(cycle,cycle_init,Cycle_Basis,Cycles_S,num_k):
        if num_k == 0 or len(cycle) == 0:
            return -1, False
        for cycle_b,is_kept in zip(Cycle_Basis,Cycles_S):
            if is_kept:
                cycle_sym_diff = cycle.symmetric_difference(cycle_b)
                # cycle_sym_diff = np.setxor1d(cycle,cycle_b,assume_unique=True)
                # print(cycle,cycle_b,cycle_sym_diff)

                if len(cycle_sym_diff) < len(cycle_init) and check_cycle(cycle_sym_diff):
                    # print(cycle_sym_diff,cycle_b,cycle)
                    return cycle_sym_diff, True

                cycle_new_basis, is_new = compo(cycle_sym_diff,cycle_init,Cycle_Basis,Cycles_S,num_k-1)
                if is_new:
                    return cycle_new_basis, is_new
        return -1, False


    def find_common_e(Cycles,cycle,Cycles_S):
        Common_edges = []
        for cycle_2,is_kept in zip(Cycles,Cycles_S):
            if is_kept and len(cycle.intersection(cycle_2)):
                Common_edges.append(cycle_2)

        return Common_edges

    def find_common_n(Cycles,cycle,Cycles_S):
        Common_edges = []
        cycle_lin = convert_back_lin(cycle,max_ind)
        for cycle_2,is_kept in zip(Cycles,Cycles_S):
            cycle_2_lin = convert_back_lin(cycle_2,max_ind)
            if is_kept and len(cycle_lin.intersection(cycle_2_lin)):
                Common_edges.append(cycle_2)

        return Common_edges

    def find_common_n_2(Cycles,cycle,Cycles_S):
        Cycles_kept = [False] * len(Cycles)
        cycle_lin = convert_back_lin(cycle,max_ind)
        for cycle_2,ind_cycle_2,is_kept in zip(Cycles,range(len(Cycles)),Cycles_S):
            cycle_2_lin = convert_back_lin(cycle_2,max_ind)
            if is_kept and len(cycle_lin.intersection(cycle_2_lin)):
                Cycles_kept[ind_cycle_2] = True
                for cycle_3,ind_cycle_3,is_kept_2 in zip(Cycles,range(len(Cycles)),Cycles_S):
                    cycle_3_lin = convert_back_lin(cycle_3,max_ind)
                    if is_kept and len(cycle_2_lin.intersection(cycle_3_lin)):
                        Cycles_kept[ind_cycle_3] = True


        Common_edges = []
        for cycle_2,is_kept,is_taken in zip(Cycles,Cycles_S,Cycles_kept):
            if is_kept and is_taken:
                Common_edges.append(cycle_2)


        return Common_edges



    a=time.time()
    Cycle_edges = []

    for cycle in Cycle_Basis:
        cycle_r = np.roll(cycle,1)
        cycle_add = []
        for ind,ind_r in zip(cycle,cycle_r):
            ind_M, ind_m = max(ind,ind_r),min(ind,ind_r)
            cycle_add.append(ind_M*max_ind + ind_m)
        # Cycle_edges.append(set(cycle_add))
        Cycle_edges.append(set(cycle_add))
    b=time.time()


    Current_Basis = Cycle_edges[:]
    Cycles_S = [True for k in range(len(Current_Basis))]
    Cycles_index = [k for k in range(len(Current_Basis))]

    for cycle,ind_cycle in zip(Current_Basis,Cycles_index):
        if ind_cycle != Cycles_index[-1]:
            Common_edges=Current_Basis[:ind_cycle] + Current_Basis[ind_cycle+1:]
        else: Common_edges = Current_Basis[:-1]
        # Common_edges = Current_Basis[:]
        # Common_edges = find_common_n_2(Current_Basis,cycle,Cycles_S)
        # print(A,B,len(Common_edges),len(Current_Basis),ind_cycle,len(Cycles_index))
        # Common_edges = find_common_n(Current_Basis,cycle,Cycles_S)
        # Common_edges = find_common_e(Current_Basis,cycle,Cycles_S)
        # print("B",len(Common_edges))
        # cycle_new_basis, is_new = compo(cycle,cycle,Current_Basis,Cycles_S,k)
        if len(cycle) < long:
            cycle_new_basis, is_new = compo(cycle,cycle,Common_edges,Cycles_S,k)
        else:
            cycle_new_basis, is_new = compo(cycle,cycle,Common_edges,Cycles_S,k_long)
        # print(cycle_new_basis,is_new)
        if is_new:
            Current_Basis.append(cycle_new_basis)
            Cycles_index.append(len(Cycles_index))
            Cycles_S[ind_cycle] = False
            Cycles_S.append(True)




    #Clean
    Cycles_clean = []
    print(len(Cycle_Basis),len(Cycle_edges),len(Current_Basis))
    # print(Current_Basis)

    for cycle,is_kept in zip(Current_Basis,Cycles_S):
        # print(cycle)
        if is_kept:
            cycle_pairs = convert_back(cycle,max_ind)

            cycle_b = cycle_pairs[0][:]
            pairs = [False] + [True] * (len(cycle_pairs)-1)
            index_check = 1
            while True in pairs:
                # print(pairs)
                if pairs[index_check]:
                    if cycle_b[-1] in cycle_pairs[index_check]:
                        i_a, i_b = cycle_pairs[index_check]
                        if i_a == cycle_b[-1]:
                            cycle_b.append(i_b)
                        else:
                            cycle_b.append(i_a)
                        pairs[index_check] = False

                index_check += 1
                if index_check >= len(cycle_pairs):
                    index_check = 1

            # print(cycle_pairs,cycle_b)


            Cycles_clean.append(cycle_b[:-1])
    g = time.time()

    return Cycles_clean





def xor_clean_rm(Cycle_Basis,k,k_long,long=12):
    import time

    Cycle_lin = [ind for cycle in Cycle_Basis for ind in cycle]
    max_ind = np.max(Cycle_lin)+1


    def convert_back(cycle,max_ind):
        cycle_pairs = []
        for ind in cycle:
            ind_m = ind%max_ind
            ind_M = ind//max_ind
            cycle_pairs.append([ind_M,ind_m])
        return cycle_pairs

    def convert_back_lin(cycle,max_ind):
        cycle_lin = []
        for ind in cycle:
            ind_m = ind%max_ind
            ind_M = ind//max_ind
            cycle_lin.append(ind_M)
            cycle_lin.append(ind_m)
        return set(cycle_lin)

    def check_cycle(cycle_sym_diff):
        if len(cycle_sym_diff) == 0: return False
        cycle = convert_back(cycle_sym_diff,max_ind)
        G = nx.Graph()
        Nodes = [int(c) for pairs in cycle for c in pairs]
        G.add_nodes_from(Nodes)
        G.add_edges_from(cycle)
        return (G.number_of_edges() == G.number_of_nodes()) * nx.is_connected(G)


    def compo(cycle,cycle_init,Cycle_Basis,Cycle_Basis_check,num_k):
        if num_k == 0:
            return False, -1
        for cycle_b in Cycle_Basis:
            if cycle_init != cycle_b:
                cycle_sym_diff = cycle.symmetric_difference(cycle_b)
                # Cycle_Basis_check = [cycle_sym_diff]

                if cycle_sym_diff in Cycle_Basis_check:
                    if len(cycle_sym_diff) > len(cycle_init):
                        return True, Cycle_Basis.index(cycle_sym_diff)
                    else: return True, -1

                is_end, index = compo(cycle_sym_diff,cycle_init,Cycle_Basis,Cycle_Basis_check,num_k-1)
                if is_end:
                    return True, index
            else: return True, index
        return False, -1


    def find_common_e(Cycles,cycle,Cycles_S):
        Common_edges = []
        for cycle_2,is_kept in zip(Cycles,Cycles_S):
            if is_kept and len(cycle.intersection(cycle_2)) and cycle_2 != cycle:
                Common_edges.append(cycle_2)

        return Common_edges

    def find_common_n(Cycles,cycle,Cycles_S):
        Common_edges = []
        cycle_lin = convert_back_lin(cycle,max_ind)
        for cycle_2,is_kept in zip(Cycles,Cycles_S):
            cycle_2_lin = convert_back_lin(cycle_2,max_ind)
            if is_kept and len(cycle_lin.intersection(cycle_2_lin)) and cycle_2_lin != cycle_lin:
                Common_edges.append(cycle_2)

        return Common_edges


    a=time.time()
    Cycle_edges = []

    for cycle in Cycle_Basis:
        cycle_r = np.roll(cycle,1)
        cycle_add = []
        for ind,ind_r in zip(cycle,cycle_r):
            ind_M, ind_m = max(ind,ind_r),min(ind,ind_r)
            cycle_add.append(ind_M*max_ind + ind_m)
        Cycle_edges.append(set(cycle_add))
    b=time.time()


    Current_Basis = np.array(Cycle_edges[:])
    Cycles_S = [True for k in range(len(Current_Basis))]
    Cycles_index = [k for k in range(len(Current_Basis))]

    for cycle,ind_cycle,is_kept in zip(Current_Basis,Cycles_index,Cycles_S):
        if is_kept:
            a=time.time()
            Cycles_Kept = Cycles_S[:]
            Cycles_Kept[ind_cycle] = False
            Common_edges = Current_Basis[Cycles_Kept]
            Common_edges = Common_edges.tolist()

            Cycle_Basis_check = find_common_e(Current_Basis,cycle,Cycles_S)
            # Common_edges = find_common_n(Current_Basis,cycle,Cycles_S)
            # print(len(Common_edges))
            b = time.time()
            if len(cycle) < long:
                is_rm, index = compo(cycle,cycle,Common_edges,Cycle_Basis_check,k)
            else:
                is_rm, index = compo(cycle,cycle,Common_edges,Cycle_Basis_check,k_long)
            # print(len(Common_edges),b-a,c-b,ind_cycle,len(Cycles_index
            # print(is_rm,index )
            if is_rm:
                if index == -1:
                    Cycles_S[ind_cycle] = False
                else:
                    index = index + (index < ind_cycle)
                    Cycles_S[index] = False
            c = time.time()



    #Clean
    Cycles_clean = []
    print(len(Cycle_Basis),len(Cycle_edges),len(Current_Basis))
    # print(Current_Basis)

    for cycle,is_kept in zip(Current_Basis,Cycles_S):
        # print(cycle)
        if is_kept:
            cycle_pairs = convert_back(cycle,max_ind)

            cycle_b = cycle_pairs[0][:]
            pairs = [False] + [True] * (len(cycle_pairs)-1)
            index_check = 1
            while True in pairs:
                # print(pairs)
                if pairs[index_check]:
                    if cycle_b[-1] in cycle_pairs[index_check]:
                        i_a, i_b = cycle_pairs[index_check]
                        if i_a == cycle_b[-1]:
                            cycle_b.append(i_b)
                        else:
                            cycle_b.append(i_a)
                        pairs[index_check] = False

                index_check += 1
                if index_check >= len(cycle_pairs):
                    index_check = 1


            Cycles_clean.append(cycle_b[:-1])
    g = time.time()
    # print(b-a,A,B,g-f)

    return Cycles_clean

def xor_rm(Cycles):
    import time
    # def convert_back(cycle,max_ind):
    #     cycle_pairs = []
    #     for ind in cycle:
    #         ind_m = ind%max_ind
    #         ind_M = ind//max_ind
    #         cycle_pairs.append([ind_M,ind_m])
    #     return cycle_pairs
    a=time.time()

    Cycles_S = [True for k in range(len(Cycles))]

    Cycle_lin = [ind for cycle in Cycles for ind in cycle]
    max_ind = np.max(Cycle_lin)+1

    #Convert to edges
    Edges_set = []
    for cycle in Cycles:
        cycle_r = np.roll(cycle,1)
        cycle_add = []
        for ind,ind_r in zip(cycle,cycle_r):
            ind_M, ind_m = max(ind,ind_r),min(ind,ind_r)
            cycle_add.append(ind_m*max_ind + ind_M)
        Edges_set.append(set(cycle_add))

    A = 0
    B = 0
    for cycle_set,ind_cycle_set in zip(Edges_set,range(len(Edges_set))):

        Edges_set_red = []
        Edges_set_red_ind = []
        for edge_set,edge_set_ind in zip(Edges_set,range(len(Edges_set))):
            #Get cycles that have at least one edge in common
            if len(cycle_set.intersection(edge_set)):
                Edges_set_red.append(edge_set)
                Edges_set_red_ind.append(edge_set_ind)

        for cycle_set_2,ind_cycle_set_red_2 in zip(Edges_set_red,range(len(Edges_set_red))):
            ind_cycle_set_2 = Edges_set_red_ind[ind_cycle_set_red_2]

            if Cycles_S[ind_cycle_set] and Cycles_S[ind_cycle_set_2]:
                new_cycle = cycle_set.symmetric_difference(cycle_set_2)

                if new_cycle in Edges_set_red:
                # if new_cycle in Edges_set:
                    # a = convert_back(cycle_set,max_ind)
                    # b = convert_back(cycle_set_2,max_ind)
                    # c = convert_back(new_cycle,max_ind)
                    # if 39 in np.array(a).flatten() or 39 in np.array(b).flatten() or 39 in np.array(c).flatten():
                    #     print("##############")
                    #     print(len(a),a)
                    #     print(len(b),b)
                    #     print(len(c),c)
                    #     print(len(cycle_set) >= len(cycle_set_2) and len(cycle_set) >= len(new_cycle),len(cycle_set_2) >= len(new_cycle))
                    #     print(ind_cycle_set)
                    #     print(ind_cycle_set_2)
                    #     print(Edges_set.index(new_cycle))
                    #     print(Cycles_S[2])
                    #     print("##############")

                    if len(cycle_set) >= len(cycle_set_2) and len(cycle_set) >= len(new_cycle): Cycles_S[ind_cycle_set] = False
                    elif len(cycle_set_2) >= len(new_cycle): Cycles_S[ind_cycle_set_2] = False
                    else: Cycles_S[Edges_set.index(new_cycle)] = False

    Cycles_return = []
    print(np.sum(Cycles_S),len(Cycles_S),Cycles_S[:5])
    for cycle,is_kept in zip(Cycles,Cycles_S):
        if is_kept: Cycles_return.append(cycle)
    return Cycles_return




def find_cycles(Bonds):
    G = nx.from_numpy_array(Bonds)
    import time
    a=time.time()
    Cycles = []
    Cycles_set = []
    L_cycles = []

    A = 0
    B = 0
    C = 0

    for node in G.nodes:
        neighbors = list(nx.neighbors(G,node))
        for n1 in neighbors:
            for n2 in neighbors:
                if n1!=n2:
                    G.remove_edges_from([(node,n1),(node,n2)])
                    try:
                        APSP = nx.bidirectional_shortest_path(G,n1,n2)
                        # APSP = nx.shortest_path(G,n1,n2)
                        cycle = [node] + APSP
                        cycle_set = set(cycle)
                        if not cycle_set in Cycles_set:
                            Cycles.append(cycle)
                            Cycles_set.append(cycle_set)
                            L_cycles.append(len(cycle))
                    except: pass
                    G.add_edges_from([(node,n1),(node,n2)])

    print(time.time()-a)
    return Cycles,L_cycles


def visualize_cycles(Pos,Types,Cycles):
    a=time.time()
    Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
    print(time.time()-a)

    colors = ["","","","green","red","pink","blue","purple","black","orange","magenta"]
    colors = colors+colors[3:]+colors[3:]+colors[3:]+colors[3:]+colors[3:]+colors[3:]+colors[3:]
    plotter = pv.Plotter()

    Pos_Si = Pos[Types==1]
    def slide(value):
        value = int(round(value))
        L_poly = []
        a=0
        A = []
        # for cycle in Cycles:
        #     # if 255 in cycle:
        #     #     a+=1
        #     #     print(a)
        #     #     A = A + cycle
        #     #     print(set(A))
        #
        #     if len(cycle)== value:
        #
        #         Pos_cycle = Pos_Si[cycle]
        #         Pos_cycle = np.append(Pos_cycle,[Pos_cycle[0]],axis=0)
        #
        #         x,y,z = Pos_cycle.transpose()
        #         pv_poly = pv.StructuredGrid(x,y,z)
        #         L_poly.append(pv_poly)
        # pv_poly = L_poly[0].merge(L_poly[1:])
        # plotter.add_mesh(pv_poly,color=colors[value],line_width=5,name="cycles")
        for value in range(3,np.max(L_cycles)):
            L_poly = []
            for cycle in Cycles:
                if np.random.random()<0.2:
                    if len(cycle)== value:

                        Pos_cycle = Pos_Si[cycle]
                        Pos_cycle = np.append(Pos_cycle,[Pos_cycle[0]],axis=0)

                        x,y,z = Pos_cycle.transpose()
                        pv_poly = pv.StructuredGrid(x,y,z)
                        L_poly.append(pv_poly)
            if len(L_poly) > 2:
                pv_poly = L_poly[0].merge(L_poly[1:])
                plotter.add_mesh(pv_poly,color=colors[value],line_width=8,name="cycles"+str(value))
                # plotter.add_mesh(pv_poly,color=colors[value],line_width=2,name="cycles"+str(value))



    # print("A")
    a = time.time()
    N_Si = len(Pos_Si)
    Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_Si).reshape((1,N_Si,1))*np.array([0,1])).reshape((N_Si*N_Si,2))
    Indices = Indices[Bonds.ravel()!=0]
    print(time.time()-a)

    # plotter.add_point_labels(Pos_Si,np.arange(len(Pos_Si)),always_visible=True)


    # poly = [pv.PolyData(np.array([Pos_Si[i_si_1],Pos_Si[i_si_2]])) for i_si_1,i_si_2 in Indices]
    # mesh = poly[0].merge(poly[1:])
    # plotter.add_mesh(mesh,opacity=0.1,color="gray")
    # print("B")

    b = time.time()
    tubes = [pv.Tube(Pos_Si[i_si_1],Pos_Si[i_si_2],n_sides=5,radius=0.2) for i_si_1,i_si_2 in Indices]
    print(time.time()-b)
    b = time.time()
    mesh = tubes[0].merge(tubes[1:])
    print(time.time()-b)
    b = time.time()
    plotter.add_mesh(mesh,opacity=0.05)
    print(time.time()-b)
    L_cycles = [len(cycle) for cycle in Cycles]

    plotter.add_slider_widget(slide, [3,np.max(L_cycles)],value=3,title="Length Cycles", fmt="%3.0f")
    plotter.show()


def save_cycles(Pos,Types,Cycles,file="cycles.txt"):
    L_cycles = [len(cycle) for cycle in Cycles]
    Pos_Si = Pos[Types==1]

    max_cycle = np.max(L_cycles)

    with open(file,'w') as file:
        file.write("# File containing the cycles computed using SHAnC\n")
        file.write("# The format is length of cycle, ids, positions (x,y,z)\n")
        for j in range(3,max_cycle+1):
            file.write("#cycles of {}\n".format(j))
            for cycle in Cycles:
                if len(cycle) == j:
                    file.write("{}".format(j))
                    for index in cycle:
                        file.write(" {}".format(index))

                    for index in cycle:
                        pos_cycle = Pos_Si[index]
                        file.write(" {:3.3f} {:3.3f} {:3.3f}".format(pos_cycle[0],pos_cycle[1],pos_cycle[2]))
                    file.write("\n")
    return



def read_cycles(file="cycles.txt"):
    Cycles = []
    L_cycles = []
    Pos_cycles = []

    for line in open(file,"r"):
        if "#" in line: pass
        else:
            lsplit = line.split()
            length = int(lsplit[0])
            ids_str = lsplit[1:length+1]
            ids = []
            for id_str in ids_str: ids.append(int(id_str))

            positions_str = lsplit[length+1:]
            positions = []
            for pos_str in positions_str: positions.append(float(pos_str))
            pos_x = positions[0::3]
            pos_y = positions[1::3]
            pos_z = positions[2::3]
            pos = np.array([pos_x,pos_y,pos_z]).transpose()
            L_cycles.append(length)
            Cycles.append(ids)
            Pos_cycles.append(pos)

    return Cycles, L_cycles, Pos_cycles







if __name__=="__main__":

    # file = "quartz_dupl.data"
    # list_BOX,list_ATOMS = read_data(file,do_scale=False)

    # file = "demo/dump_last_oh.lammpstrj"
    # file = "dummp_trimmed.lammpstrj"
    # file = "demo/dummp_128.lammpstrj"
    # file = "demo/dummp_256.lammpstrj"
    # file = "demo/dummp_512.lammpstrj"
    # file = "demo/dummp_1024.lammpstrj"
    # file = "dummps_snad_last.lammpstrj"
    # file = "demo/dummps_round_2_last.lammpstrj"
    file = "demo/dummps_round_3_last.lammpstrj"
    # file = "dummps_long_last.lammpstrj"
    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)

    list_TSTEP=[0]
    list_Pos = list_ATOMS[:,:,2:]
    list_Types = list_ATOMS[:,:,1]
    Pos = list_ATOMS[-1][:,2:]
    Types = list_ATOMS[-1][:,1]




    import time
    import networkx as nx


    if False:
        Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
        a = time.time()
        Cycles,L_cycles = find_cycles(Bonds)
        Cycles = xor_rm(Cycles)
        print(time.time()-a)
    else:
        Cycles,L_cycles,Pos_cycles = read_cycles(file="cycles.txt")

    L_cycles = [len(cycle) for cycle in Cycles]

    C = np.zeros((np.max(L_cycles)+3),dtype="int")
    for k in L_cycles:
        C[k]+=1
    print(C)

    visualize_cycles(Pos,Types,Cycles)
    # save_cycles(Pos,Types,Cycles,file="cycles_60ps.txt")
    # plot_cycles(L_cycles)
    # G = nx.from_numpy_array(Bonds)
    # print(len(Cycles),(G.number_of_edges() - G.number_of_nodes() + 1))

    if False:
        a = time.time()
        G = nx.from_numpy_array(Bonds)
        L = nx.minimum_cycle_basis(G)
        C2 = []
        for k in L:
            C2.append(len(k))
        b=time.time()
        print(b-a)
        A = np.zeros(np.max(C2)+1)
        for k in C2:
            A[k]+=1
        print(A)
        print(time.time()-a)


    # A = [ 0,  0.,  0., 14., 27. ,42., 19.,  3.] # 128
    # A = [  0.,   0.,   0.,  19.,  35.,  54., 102. ,  2.,   1.] #256
    # A = [  0.,   0.,   0.,  30.,  34.,  71., 231.,   5.,   2.] #512
    # A = [  0. ,  0. ,  0. , 39.  ,20.,  67., 581. ,  2.,   0. ,  1.  , 0. ,  1.] #1024
    # Com = np.array(A + [0] * (np.max(L_cycles)+3-len(A)))
    # C = np.array(C.tolist() + [0] * (len(A)-len(C)))
    #
    # D = C-Com
    # print(abs(np.sum(D[D<0])))
    # print((np.sum(D[D>0])))

    # cube = 9
    # Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
    # G = nx.from_numpy_array(Bonds)
    # a = time.time()
    # Cycles, L_cycles = count_cycles(Pos,Types,cube=cube,Lims=list_BOX[-1],periodic=False)
    # b= time.time()
    # D = xor_clean_rec(Cycles,1,1,12)
    # c=time.time()
    # # C = xor_clean_rm(D,1,1,12)
    # C = xor_clean_rm_2(Pos,D,1,1,12)
    # d= time.time()
    # print(d-c)
    # C2 = []
    # for k in C:
    #     C2.append(len(k))
    #
    # L = np.zeros((np.max(C2)+1))
    # for k in C2:
    #     L[k] += 1
    # L = L * (G.number_of_edges() - G.number_of_nodes() + 1) / len(C2)
    #
    # C2_2 = []
    # for k,l in zip(L,range(len(L))):
    #     C2_2 = C2_2 + [l]*int(k)
    # plot_cycles(C2_2)
    # e = time.time()
    # print(e-a)


    #
    # a=time.time()
    # Cube = np.linspace(16,40,20)
    # Cube = np.linspace(8,16,8)
    # Cube = [9.2]
    #
    # L_m = []
    # L_p = []
    #
    # Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
    # G = nx.from_numpy_array(Bonds)
    # #
    # for cube in Cube:
    #     #
    #     # Cycles, L_cycles = count_cycles_test(Pos,Types,cube=cube,Lims=list_BOX[-1],periodic=True)
    #     Cycles, L_cycles = count_cycles(Pos,Types,cube=cube,Lims=list_BOX[-1],periodic=False)
    #     # D = xor_clean_rec(Cycles,2,2,12)
    #     D = xor_clean_rec(Cycles,1,1,12)
    #     C = xor_clean_rm(D,1,1,12)
    #     # C = xor_clean_rm_2(Pos,D,1,1,12)
    #     # print("C",len(C))
    #     # C = xor_clean_rm(Cycles,1,1,12)
    #     # C = Cycles[:]
    #
    #     # Cycles = nx.cycle_basis(G)
    #     # # D = xor_clean_rec(Cycles,2,2,12)
    #     # C = xor_clean_rec(Cycles,1,1,12)
    #     # # C = xor_clean_rm(D,1,1,12)
    #
    #     C2 = []
    #     for k in C:
    #         C2.append(len(k))
    #     # plot_cycles(C2)
    #     b=time.time()
    #     print(b-a)
    #     # print(G.number_of_edges() - G.number_of_nodes() + 1, len(C2), len(C2) - (G.number_of_edges() - G.number_of_nodes() + 1))
    #     # print(np.sum(np.array(C2)!=6))
    #
    #     C = np.zeros((np.max(C2)+1))
    #     for k in C2:
    #         C[k] += 1
    #     print(len(C2),(G.number_of_edges() - G.number_of_nodes() + 1))
    #     C = C * (G.number_of_edges() - G.number_of_nodes() + 1) / len(C2)
    #     C = C.round()
    #     # A = [ 0,  0.,  0., 14., 27. ,42., 19.,  3.] # 128
    #     # A = [  0.,   0.,   0.,  19.,  35.,  54., 102. ,  2.,   1.] #256
    #     # A = [  0.,   0.,   0.,  30.,  34.,  71., 231.,   5.,   2.] #512
    #     A = [  0. ,  0. ,  0. , 39.  ,20.,  67., 581. ,  2.,   0. ,  1.  , 0. ,  1.] #1024
    #     Com = np.array(A + [0] * (np.max(C2)+1-len(A)))
    #     C = np.array(C.tolist() + [0] * (len(A)-len(C)))
    #
    #     D = C-Com
    #     L_m.append(abs(np.sum(D[D<0])))
    #     L_p.append(np.sum(D[D>0]))
    #
    # plt.plot(Cube,L_m,"o-r")
    # plt.plot(Cube,L_p,"o-b")
    # plt.show()





    # a=time.time()
    # Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
    # G = nx.from_numpy_array(Bonds)
    # L = nx.minimum_cycle_basis(G)
    # # C = []
    # C2 = []
    # for k in L:
    #     # C.append(k)
    #     C2.append(len(k))
    # # print(len(C))
    # b=time.time()
    # print(b-a)
    # plot_cycles(C2)
    # A = np.zeros(np.max(C2)+1)
    # for k in C2:
    #     A[k]+=1
    # print(A)






    # a=time.time()
    # Bonds = compute_bonds_graph(Pos,Types,cube=60,periodic=False,Lims=list_BOX[-1])
    # b=time.time()
    # print(b-a)
    # # G = min_basis(Cycles)
    #
    # G = nx.from_numpy_array(Bonds)
    # print("Edges ", G.number_of_edges(), " Nodes ", G.number_of_nodes())
    # c=time.time()
    # print(c-b)
    # # L = nx.minimum_cycle_basis(G)
    # L = nx.cycle_basis(G)
    # #
    # d=time.time()
    # print(d-c)
    # C = []
    # C2 = []
    # for k in L:
    #     C.append(k)
    #     C2.append(len(k))
    # print(len(C))
    # e=time.time()
    # print(e-d)
    # # D = xor_clean(C,2)
    # # D = xor_clean_rec(C,1,2,20)
    # D = xor_clean_rec(C,1,1,12)
    # # G2 = min_basis(D)
    # # L2 = nx.minimum_cycle_basis(G2)
    # # C3 = []
    # # C4 = []
    # # for k in L2:
    # #     C3.append(len(k))
    # #     C4.append(k)
    # # print("NEW",len(C3))
    # f = time.time()
    #
    # E = []
    # for k in D:
    #     E.append(len(k))
    #
    # print(len(C2))
    # plot_cycles(C2)
    # print(len(E))
    # plot_cycles(E)
    # # plot_cycles(L_cycles,cube=16)
    # print(f-e)
    # print(f-a)
    # print(np.sum(np.array(E)!=6))
    # C2 = E
    # #
    # # # #


    # Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
    # G = nx.from_numpy_array(Bonds)
    # a = time.time()
    # S = nx.spanner(G,1)
    # b = time.time()
    # print(S.number_of_nodes(),G.number_of_nodes())
    # print(S.number_of_edges(),G.number_of_edges())
    # nx.draw(S)
    # plt.show()


    # # # #
    # # #
    #
    #
    # A = [0.0, 0.0, 0.0, 47.0, 37.0, 90.0, 548.0, 3.0, 1.0]
    # A = [  0.,   0.,   0.,  30.,  34.,  71., 231.,   5.,   2.]
    #
    #
    # Com = np.array(A + [0] * (np.max(C2)+1-len(A)))
    #
    # # C = np.zeros((np.max(C2)+1))
    # # for k in C2:
    # #     C[k]+=1
    # print(C.tolist())
    # print(C-Com)
    # D = C-Com
    # print(np.sum(D[D<0]))
    # print(np.sum(D[D>0]))

    # analyze(Pos,Types,periodic=True,Lims=list_BOX[-1])
    # plot_syst(Pos,Types,Lims=list_BOX[-1])
    # analyze_mult(list_TSTEP,list_Pos,list_Types,periodic=True,Lims=list_BOX[-1])
    #
    # A=[]
    # B=[]
    #
    # LL=[]
    # # L = np.linspace(9,30,30,endpoint=True)
    # L = [k for k in range(5,20)]
    # # L = [11]
    # # L = [13]
    # # import networkx as nx
    # for k in L:
    #     # Cycles, L_cycles = count_cycles(Pos,Types,cube=k,Lims=list_BOX[-1],periodic=True)
    #     Cycles, L_cycles = count_cycles_test(Pos,Types,cube=k,Lims=list_BOX[-1],periodic=False)
    # #     # for j in Cycles:
    # #     #     j.sort()
    # #     #     print(j)
    # #     # print("\n\n")
    #     A.append(len(L_cycles))
    #     B.append(L_cycles)
    #     # LL.append(np.max(L_cycles))
    # #
    # #
    # #     B = [k for C in Cycles for k in C]
    # #     Bonds = np.zeros((np.max(B)+1,np.max(B)+1))
    # #     for c in Cycles:
    # #         c_2 = np.roll(c,1)
    # #         for c1,c2 in zip(c,c_2):
    # #             Bonds[c1,c2] = 1
    # #             Bonds[c2,c1] = 1
    # #     New_cycles=[]
    # #     New_G = nx.from_numpy_array(Bonds)
    # #     L2 = nx.minimum_cycle_basis(New_G)
    # #     for cycle_2 in L2:
    # #         New_cycles.append(cycle_2)
    # #     # Cycles = New_cycles[:]
    # #     LL.append(len(New_cycles))
    #
    #
    #
    #
    #
    #
    # plt.plot(L,A,"o-r")
    # plt.hlines(len(C),5,20,"b")
    # plt.xlabel("Length of Box (A)")
    # plt.ylabel("Number of Cycles")
    # # plt.plot(L,LL,"o-b")
    # plt.show()
    # # analyze_plot_syst(Pos,Types,periodic=False,Lims=list_BOX[-1],Cycles=Cycles,L_cycles=L_cycles,draw_limit=10,compute_limit=12)
    # # analyze_mult(list_TSTEP,list_Pos,list_Types,periodic=True,Lims=list_BOX[-1])
    # # plot_syst(Pos,Types,Cycles=Cycles)
    # # plot_syst(Pos,Types,Cycles=Cycles,L_cycles=L_cycles)
    # plot_cycles(L_cycles,cube=16)
    # # print(len(L_cycles))