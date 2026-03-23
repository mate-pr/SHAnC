import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from read_write import *
import pyvista as pv
import scipy as sp
import scipy.spatial.distance as sd
import scipy.signal as sps




def compute_bonds(Pos,Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,do_count_type_3=True):
    """
    compute_bonds(Pos,Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,do_count_type_3=True)

    Compute the Bonds of a silica system.
    |!| This version must only be used for small systems. It takes too much memory otherwise.
    The types are the following : 1 : Si, 2: O, 3: Oh, 4: H

    Parameters
    ----------
    Pos : array
        The position of the atoms
    Types : array
        The type of the Atoms
    threshold_Si : float, optional
        The threshold used to consider if Si and O are bonding. 2 by default
    threshold_O : float, optional
        The threshold used to consider if O and Si are bonding. 2 by default
    threshold_H : float, optional
        The threshold used to consider if O and H are bonding. 1.3 by default
    do_count_type_3 : bool, optional
        If one wants to consider the Oh in the calculations.

    Returns
    -------
        Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O
    """

    Pos_Si = Pos[Types==1]
    if do_count_type_3:
        Pos_O = Pos[((Types==2) + (Types==3)).astype("bool")]
    else:
        Pos_O = Pos[ ((Types==2)).astype("bool")]
    Pos_H = Pos[Types==4]
    if Pos_H.any(): H_present = True
    else: H_present = False

    num_at = len(Types)
    num_Si = len(Pos_Si)
    num_O = len(Pos_O)

    # There are multiple ways to compute the distance, but this one is the fastest I found
    Dist = sd.cdist(Pos_Si,Pos_O)

    Si_Nearest_O = np.min(Dist,axis=1).reshape((len(Dist),1))
    O_Nearest_Si = np.min(Dist,axis=0)

    Bonds_Si = (Dist<(threshold_Si))
    Bonds_O = (Dist<(threshold_O))
    Bonds =  Bonds_Si + Bonds_O


    Si_count_O = np.sum(Bonds,axis=1)
    O_count_Si = np.sum(Bonds,axis=0)

    if H_present:
        Dist_OH = sd.cdist(Pos_H,Pos_O[:num_O])

        Bonds_OH = Dist_OH<threshold_H
        O_count_H = np.sum(Bonds_OH,axis=0)
        H_count_O = np.sum(Bonds_OH,axis=1)
    else : O_count_H, H_count_O = np.array([]),np.array([])

    return Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O


def compute_bonds_graph(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[],rdf_max=5):
    """
    compute_bonds_graph(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[])

    Compute the bonds that can be used to create a graph.
    The output is a neighbor matrix array taht can be converted to a graph using networkx nx.from_numpy_array

    Parameters
    ----------

    Pos : array
        The position of the atoms of the system
    Types: array
        The types of the atoms of the system
    cube: float, optional
        The edge of the cubes used to divide the system, 30 by default. Larger cubes are faster but are more memory intensive
    threshold_Si : float, optional
        The threshold used to consider if Si and O are bonding. 2 by default
    threshold_O : float, optional
        The threshold used to consider if O and Si are bonding. 2 by default
    threshold_H : float, optional
        The threshold used to consider if O and H are bonding. 1.3 by default
    periodic : bool, optional
        Compute as if the system was periodic, True by default
    Lims : list, optional
        The limits of the system. It is necessary for periodic computations

    Returns
    -------
        Neighbor matrix of the system

    """

    Lx,Ly,Lz = np.max(Pos,axis=0)
    lx,ly,lz = np.min(Pos,axis=0)
    if periodic:
        if len(Lims) == 0:
            print("No limits were provided, the system will be taken as NON PERIODIC. Use the Lim keyword to add limits")
            periodic = False
        else:
            lz,Lz = Lims[2]


    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Si_or = np.sum(Types==1)


    if periodic:
        #Adds atoms to the system periodically to account for the periodicity
        Dz = Lz-lz

        Pos_add_z = Pos[:,2] > (Lz - rdf_max)
        Pos_remove_z = Pos[:,2] < (lz + rdf_max)

        Pos_add_Lz = Pos[Pos_add_z] - np.array([[0,0,Dz]])
        Pos_remove_Lz = Pos[Pos_remove_z] + np.array([[0,0,Dz]])

        Pos_add = np.append(Pos_add_Lz,Pos_remove_Lz,axis=0)
        Pos_added = np.append(Pos_added,Pos_add,axis=0)

        Types_add = np.append(Types[Pos_add_z],Types[Pos_remove_z],axis=0)
        Types_added = np.append(Types_added,Types_add,axis=0)

    # print("MIN")
    # print(np.min(Pos_added[:,0]),np.max(Pos_added[:,0]))
    # print(np.min(Pos_added[:,1]),np.max(Pos_added[:,1]))
    # print(np.min(Pos_added[:,2]),np.max(Pos_added[:,2]))

    Num_Si = np.sum(Types_added==1)
    Bonds_tot = np.zeros((Num_Si,Num_Si))
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                # print("LIMS")
                # print(((x*cube + lx - threshold_Si - 0.2)),((x+1)*cube + lx + threshold_Si + 0.2))
                # print(((y*cube + ly - threshold_Si - 0.2)),((y+1)*cube + ly + threshold_Si + 0.2))
                # print(((z*cube + lz - threshold_Si - 0.2)),((z+1)*cube + lz + threshold_Si + 0.2))

                #Slice the system inside this cube
                Pos_trunc_x = (Pos_added[:,0] > (x*cube + lx - threshold_Si - 0.2)) * (Pos_added[:,0] < ((x+1)*cube + lx + threshold_Si + 0.2))
                Pos_trunc_y = (Pos_added[:,1] > (y*cube + ly - threshold_Si - 0.2)) * (Pos_added[:,1] < ((y+1)*cube + ly + threshold_Si + 0.2))
                Pos_trunc_z = (Pos_added[:,2] > (z*cube + lz - threshold_Si - 0.2)) * (Pos_added[:,2] < ((z+1)*cube + lz + threshold_Si + 0.2))
                Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                Pos_trunc = Pos_added[Pos_trunc_ind]
                Types_trunc = Types_added[Pos_trunc_ind]
                if (Types_trunc == 1).any() and (Types_trunc==2).any():

                    Bonds = compute_bonds(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H)[0]
                    Bonds = Bonds.astype("float")
                    #Get Bonds Si
                    Bonds = Bonds.dot(Bonds.transpose())
                    #Set distance to 1
                    Bonds = Bonds / (Bonds + (Bonds==0)*1)
                    Pos_trunc_ind_Si = Pos_trunc_ind[Types_added==1]

                    Pos_trunc_ind_Si = np.matmul(Pos_trunc_ind_Si.reshape((len(Pos_trunc_ind_Si),1)),Pos_trunc_ind_Si.reshape((1,len(Pos_trunc_ind_Si))))

                    Bonds_tot[Pos_trunc_ind_Si] = Bonds_tot[Pos_trunc_ind_Si] + Bonds.ravel()
    Bonds_tot_or = Bonds_tot[:Num_Si_or]


    if periodic:
        Pos_add_z_Si = Pos_add_z[Types==1]
        Pos_remove_z_Si = Pos_remove_z[Types==1]
        num_add = np.sum(Pos_add_z_Si)

        Bonds_tot_or[Pos_add_z_Si] = Bonds_tot_or[Pos_add_z_Si] + Bonds_tot[Num_Si_or:Num_Si_or+num_add]
        Bonds_tot_or[Pos_remove_z_Si] = Bonds_tot_or[Pos_remove_z_Si] + Bonds_tot[Num_Si_or+num_add:]
        Bonds_tot_or = Bonds_tot_or[:,:Num_Si_or]

    Bonds_tot_or = Bonds_tot_or + Bonds_tot_or.transpose()
    Bonds_tot_or = Bonds_tot_or / (Bonds_tot_or + (Bonds_tot_or==0)*1)
    Bonds_tot_or = Bonds_tot_or - np.eye(len(Bonds_tot_or)) * Bonds_tot_or
    return Bonds_tot_or

def compute_hist_neighbors(Pos,Types,cube=100,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[],rdf_max=5):
    """
    compute_hist_neighbors(Pos,Types,cube=100,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[],rdf_max=5)

    Computes the Bonds and RDF by slicing the system in multiple subsystems

    Parameters
    ----------

    Pos : array
        The position of the atoms of the system
    Types: array
        The types of the atoms of the system
    cube: float, optional
        The edge of the cubes used to divide the system, 30 by default. Larger cubes are faster but are more memory intensive
    threshold_Si : float, optional
        The threshold used to consider if Si and O are bonding. 2 by default
    threshold_O : float, optional
        The threshold used to consider if O and Si are bonding. 2 by default
    threshold_H : float, optional
        The threshold used to consider if O and H are bonding. 1.3 by default
    periodic : bool, optional
        Compute as if the system was periodic, True by default
    Lims : list, optional
        The limits of the system. It is necessary for periodic computations
    rdf_max : float, optional
        The maximum distance for the RDF. 5 by default

    Returns
    -------
        Dist_list, Si_count_O_tot, O_count_Si_tot

    """

    Lx,Ly,Lz = np.max(Pos,axis=0)
    lx,ly,lz = np.min(Pos,axis=0)

    if periodic:
        if len(Lims) == 0:
            print("No limits were provided, the system will be taken as NON PERIODIC. Use the Lim keyword to add limits")
            periodic = False
        else:
            lz,Lz = Lims[2]



    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Si_or = np.sum(Types==1)


    if periodic:
        Pos_add_z = Pos[:,2] > (Lz - rdf_max)
        Pos_remove_z = Pos[:,2] < (lz + rdf_max)

        Pos_add_Lz = Pos[Pos_add_z] - np.array([[0,0,Lz-lz]])
        Pos_remove_Lz = Pos[Pos_remove_z] + np.array([[0,0,Lz-lz]])

        Pos_add = np.append(Pos_add_Lz,Pos_remove_Lz,axis=0)
        Pos_added = np.append(Pos_added,Pos_add,axis=0)

        Types_add = np.append(Types[Pos_add_z],Types[Pos_remove_z],axis=0)
        Types_added = np.append(Types_added,Types_add,axis=0)


    Num_at = len(Types)
    Num_Si = np.sum(Types==1)
    Num_O = np.sum(Types==2)
    In_trunc = np.array([1]*Num_at + [0]*(len(Pos_added)-Num_at),dtype="bool")


    Si_count_O_tot = np.zeros((Num_Si))
    O_count_Si_tot = np.zeros((Num_O))
    Dist_list = []
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                #the first slicing is for the system that will the computation will be done to
                Pos_trunc_x_u = (Pos_added[:,0] >= (x*cube + lx)) * (Pos_added[:,0] < ((x+1)*cube + lx))
                Pos_trunc_y_u = (Pos_added[:,1] >= (y*cube + ly)) * (Pos_added[:,1] < ((y+1)*cube + ly))
                Pos_trunc_z_u = (Pos_added[:,2] >= (z*cube + lz)) * (Pos_added[:,2] < ((z+1)*cube + lz))
                Ind_trunc_uniq = Pos_trunc_x_u * Pos_trunc_y_u * Pos_trunc_z_u
                Pos_trunc_uniq = Pos_added[Ind_trunc_uniq]
                Types_trunc_uniq = Types_added[Ind_trunc_uniq]

                #The second slicing is for the RDF, as it needs a larger distance
                Pos_trunc_x = (Pos_added[:,0] >= (x*cube + lx - rdf_max)) * (Pos_added[:,0] < ((x+1)*cube + lx + rdf_max))
                Pos_trunc_y = (Pos_added[:,1] >= (y*cube + ly - rdf_max)) * (Pos_added[:,1] < ((y+1)*cube + ly + rdf_max))
                Pos_trunc_z = (Pos_added[:,2] >= (z*cube + lz - rdf_max)) * (Pos_added[:,2] < ((z+1)*cube + lz + rdf_max))
                Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                Pos_trunc = Pos_added[Pos_trunc_ind]
                Types_trunc = Types_added[Pos_trunc_ind]

                Ind_trunc_uniq_in_trunc = (Ind_trunc_uniq * In_trunc)[Pos_trunc_ind]

                if (Types_trunc == 1).any() and (Types_trunc == 2).any():

                    Si_count_O = compute_bonds(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,do_count_type_3=True)[1]
                    O_count_Si = compute_bonds(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,do_count_type_3=False)[2]


                    Si_count_O = Si_count_O[Ind_trunc_uniq_in_trunc[(Types_trunc==1).astype("bool")]]
                    Si_index = Ind_trunc_uniq[:Num_at][Types==1]
                    Si_count_O_tot[Si_index] = Si_count_O
                    O_count_Si = O_count_Si[Ind_trunc_uniq_in_trunc[(Types_trunc==2).astype("bool")]]
                    O_index = Ind_trunc_uniq[:Num_at][((Types==2)).astype("bool")]
                    O_count_Si_tot[O_index] = O_count_Si


                    Pos_Si = Pos_trunc_uniq[Types_trunc_uniq==1]
                    Pos_O = Pos_trunc_uniq[((Types_trunc_uniq==2) + (Types_trunc_uniq==3)).astype("bool")]
                    Dist_Si_O = sd.cdist(Pos_Si,Pos_O)
                    # Dist_Si_O = sd.cdist(Pos_Si,Pos_Si)
                    Dist_Si_O = Dist_Si_O[:Num_Si]
                    Dist_Si_O = Dist_Si_O + (Dist_Si_O==0)*100
                    Dist_Si_O = Dist_Si_O[Dist_Si_O<rdf_max].ravel()
                    Dist_list.append(Dist_Si_O)

    Si_count_O_tot = Si_count_O_tot[:Num_Si]
    O_count_Si_tot = O_count_Si_tot[:Num_O]

    return Dist_list, Si_count_O_tot, O_count_Si_tot



def plot_syst(Pos,Types,do_bonds=True,Cycles=None,L_cycles=None):
    """
    plot_syst(Pos,Types,Cycles=None,L_cycles=None)

    Plots a whole system with cycles if given.
    This should only be used for small systems as it has no clustering

    Parameters
    ----------
        Pos : list
            The position of the atoms
        Types : list
            The type of the atoms
        Cycles : list
            The cycles computed using the script_cycles file
        L_cycle : list
            The length of the cycles computed using the script_cycles file

    Returns
    -------
        None
            A graph is produced

    """
    plotter = pv.Plotter()
    plotter.add_axes()



    sp = pv.Sphere(radius=0.6)
    Colors_Si = "dodgerblue"
    data = pv.PolyData(Pos[Types==1])
    pc = data.glyph(scale=False,geom=sp,orient=False)
    plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color=Colors_Si)

    sp = pv.Sphere(radius=0.4)
    Colors_O = "red"
    data = pv.PolyData(Pos[Types==2])
    pc = data.glyph(scale=False,geom=sp,orient=False)
    plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color=Colors_O)

    if (Types==3).any():
        sp = pv.Sphere(radius=0.2)
        Colors_H = "gray"
        data = pv.PolyData(Pos[Types==3])
        pc = data.glyph(scale=False,geom=sp,orient=False)
        plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color=Colors_H)

    if do_bonds:

        Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O = compute_bonds(Pos,Types)

        N_Si,N_O = np.shape(Bonds)
        Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))

        Pos_Si = Pos[Types==1]
        Pos_O = Pos[Types==2]

        Indices = Indices[Bonds.ravel()!=0]

        tubes = [pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2) for i_si,i_o in Indices]
        mesh = tubes[0].merge(tubes[1:])
        plotter.add_mesh(mesh,opacity=0.3,pbr=True,roughness=.5,metallic=.2,color="blue")

    if not type(Cycles) is type(None) and type(L_cycles) is type(None):
        Pos_Si = Pos[Types==1]
        print(len(Pos_Si))
        tubes = []


        for cycle in Cycles:
            cycle2 = np.roll(cycle,1)

            for cycle_1,cycle_2 in zip(cycle,cycle2):
                tubes.append(pv.Tube(Pos_Si[cycle_1],Pos_Si[cycle_2],n_sides=5,radius=0.5))

        mesh = tubes[0].merge(tubes[1:])
        plotter.add_mesh(mesh,opacity=0.5,pbr=True,roughness=.5,metallic=.8,color="red")

    elif not type(Cycles) is type(None) and not type(L_cycles) is type(None):
        Pos_Si = Pos[Types==1]
        tubes = []

        for cycle in Cycles:
            cycle2 = np.roll(cycle,1)
            for cycle_1,cycle_2 in zip(cycle,cycle2):
                tubes.append(pv.Tube(Pos_Si[cycle_1],Pos_Si[cycle_2],n_sides=5,radius=0.5))

        mesh = tubes[0].merge(tubes[1:])
        plotter.add_mesh(mesh,opacity=0.5,pbr=True,roughness=.5,metallic=.8,color="red")



    plotter.show()



def analyze_plot_syst(Pos,Types,periodic=False,Lims=[],draw_limit=5,compute_limit=7,Cycles=None,L_cycles=None):
    """
    analyze_plot_syst(Pos,Types,periodic=False,Lims=[],draw_limit=5,compute_limit=7,Cycles=None,L_cycles=None)

    Plot slices of the system highlighting the atoms with wrong number of saturation
    A slider is implemented to move the visualization through the helix

    Parameters
    ----------
        Pos : list
            The position of the atoms
        Types : list
            the types of the atoms
        periodic : bool, optional
            Is the system is periodic in z. False by default
        Lims : list, optional
            The boundaries of the system used for the periodicity. [] by default
        draw_limit : float, optional
            The height of the slices that are drawn. 5 by default
        compute_limit : float, optional
            the height of the slices used for computation. 7 by default
        Cycles : list, optional
            The cycles of the system computed using script_cycles
        L_cycles : list, optional
            The length of the cycles computed using script_cycles

    Returns
    -------
        None
            A plot of the system is shown
    """

    def slide(value):
        """Function used for the slider which slides the computation"""

        if periodic:
            Pos_cut_l = (Pos[:,2] >= value-compute_limit) * (Pos[:,2] < value+compute_limit) + ((Pos[:,2] + Lims[2][1]) >= value-compute_limit) * ((Pos[:,2] + Lims[2][1]) < value+compute_limit) + ((Pos[:,2] - Lims[2][1]) > value-compute_limit) * ((Pos[:,2] - Lims[2][1]) < value+compute_limit)
            Pos_cut_l = Pos_cut_l > 0
        else:
            Pos_cut_l = (Pos[:,2] > value-compute_limit) * (Pos[:,2] < value+compute_limit)
        Pos_cutted_l = Pos[Pos_cut_l]
        Types_cutted_l = Types[Pos_cut_l]
        Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O = compute_bonds(Pos_cutted_l,Types_cutted_l)
        Pos_cut = (Pos_cutted_l[:,2] > value-draw_limit) * (Pos_cutted_l[:,2] < value+draw_limit)

        #Remove outside
        Pos_cutted = Pos_cutted_l[Pos_cut]
        Types_cutted = Types_cutted_l[Pos_cut]
        select_Si = Pos_cut[Types_cutted_l==1]
        select_O = Pos_cut[((Types_cutted_l==2) + (Types_cutted_l==3)).astype("bool")]


        Si_count_O = Si_count_O[select_Si]
        O_count_Si = O_count_Si[select_O]
        if O_count_H.any(): O_count_H = O_count_H[select_O]



        #Add Si with colors
        sp = pv.Sphere(radius=0.6)
        Colors_Si = ["white","powderblue","lightsteelblue","dodgerblue","blue","navy","black"]
        for number_bonds in range(0,7):
            op = 1.0
            if number_bonds == 4: op = 0.05
            Si_bonds = (Si_count_O==number_bonds)
            if Si_bonds.any():
                data = pv.PolyData(Pos_cutted[Types_cutted==1][Si_bonds])
            else: data = pv.PolyData()
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_Si[number_bonds],name="Si_{}".format(number_bonds))

        sp = pv.Sphere(radius=0.4)
        Colors_O = ["white","orange","red","darkred","black"]
        for number_bonds in range(0,5):
            op = 1.0
            if number_bonds == 2: op = 0.05
            if O_count_H.any():
                O_bonds = ((O_count_Si + O_count_H)==number_bonds)
            else:
                O_bonds = (O_count_Si==number_bonds)

            if O_bonds.any():
                data = pv.PolyData(Pos_cutted[((Types_cutted==2)+(Types_cutted==3)).astype("bool")][O_bonds])
            else: data = pv.PolyData()
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_O[number_bonds],name="O_{}".format(number_bonds))


        sp = pv.Sphere(radius=0.2)
        Colors_H = ["white","gray","black"]
        data = pv.PolyData(Pos_cutted[Types_cutted==4])
        pc = data.glyph(scale=False,geom=sp,orient=False)
        plotter.add_mesh(pc,opacity=0.05,pbr=True,roughness=.5,metallic=.2,color="gray",name="H")

        if periodic:
            Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O = compute_bonds(Pos_cutted_l,Types_cutted_l)
        Bonds = Bonds[select_Si]
        Bonds = Bonds[:,select_O]

            #Recompute bonds without periodicity otherwise bonds crossing
        N_Si,N_O = np.shape(Bonds)
        Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))

        Pos_Si = Pos_cutted[Types_cutted==1]
        Pos_O = Pos_cutted[((Types_cutted==2) + (Types_cutted==3)).astype("bool")]

        Indices = Indices[Bonds.ravel()!=0]
        tubes = pv.MultiBlock()
        for i_si,i_o in Indices:
            t = pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2)
            tubes.append(t)

        mesh = tubes.combine()
        plotter.add_mesh(mesh,opacity=0.1,pbr=True,roughness=.5,metallic=.2,color="blue",name="Bonds")



        if not type(Cycles) is type(None) and not type(L_cycles) is type(None):
            Pos_Si_l = Pos_cutted_l[Types_cutted_l==1]
            Ind_Si_cut = Pos_cut_l[Types==1]
            Ind_sum = np.cumsum(Ind_Si_cut)-1
            tubes_cycles = [pv.MultiBlock() for k in range(np.max(L_cycles)+1)]

            for cycle,l_cycle in zip(Cycles,L_cycles):
                cycle2 = np.roll(cycle,1)
                for cycle_1,cycle_2 in zip(cycle,cycle2):
                    if Ind_Si_cut[cycle_1] and Ind_Si_cut[cycle_2]:

                        # print(cycle_1,Ind_sum[cycle_1])
                        cycle_1 = Ind_sum[cycle_1]
                        cycle_2 = Ind_sum[cycle_2]
                        tubes_cycles[l_cycle].append(pv.Tube(Pos_Si_l[cycle_1],Pos_Si_l[cycle_2],n_sides=5,radius=0.5))

            colors = ["","","white","gray","green","red","pink","blue","purple","yellow","black","orange","magenta"]
            if len(L_cycles) > len(colors):
                colors = colors + colors[2:]

            for multblock,color,n in zip(tubes_cycles,colors,np.arange(len(tubes_cycles))):
                if multblock.n_blocks:
                    mesh = multblock.combine()
                    if n==6: opacity = 0.2
                    else: opacity = 0.3
                    plotter.add_mesh(mesh,opacity=opacity,pbr=True,roughness=.5,metallic=.8,color=color,name="cycles_{}".format(n))
                else: plotter.remove_actor("cycles_{}".format(n))


        # plotter.add_point_labels(Pos_Si_l,np.arange(len(Pos_Si_l)),always_visible=True)



    z_min, z_max = np.min(Pos[:,2]), np.max(Pos[:,2])
    # print(np.shape(O_count_H))
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_slider_widget(slide, [z_min,z_max],value=(z_max+z_min)/2,title="Pos", fmt="%3.3e")

    plotter.show()




def compute_analysis(Pos,Types,hist_Dens,hist_Si,hist_O,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=False,Lims=[],rdf_max=5,density=True):
    """
    compute_analysis(Pos,Types,hist_Dens,hist_Si,hist_O,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=False,Lims=[],rdf_max=5)

    Function used by analyze_mult. It computes the histogram of the RDF, and bonds of the system.
    """
    Dist_list, Si_count_O, O_count_Si = compute_hist_neighbors(Pos,Types,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,periodic=periodic,Lims=Lims,rdf_max=rdf_max)
    #
    Dist_list = [k for j in Dist_list for k in j]

    num_inf_15 = np.sum(np.array(Dist_list) < 1.5)
    num_inf_155 = np.sum(np.array(Dist_list) < 1.55)
    num_si_3 = np.sum(Si_count_O==3)
    num_si_5 = np.sum(Si_count_O==5)
    num_si_6 = np.sum(Si_count_O==6)
    num_o_1 = np.sum(O_count_Si==1)
    num_o_3 = np.sum(O_count_Si==3)
    N_at = len(Pos)
    print("Bonds < 1.5 A : {}, {:3.3f}%".format(num_inf_15,num_inf_15/N_at*4/3*100))
    print("Bonds < 1.55 A : {}, {:3.3f}%".format(num_inf_155,num_inf_155/N_at*4/3*100))
    print("3 Bonds Si : {}, {:3.3f}%".format(num_si_3,num_si_3/N_at*3*100))
    print("5 Bonds Si : {}, {:3.3f}%".format(num_si_5,num_si_5/N_at*3*100))
    print("6 Bonds Si : {}, {:3.3f}%".format(num_si_6,num_si_6/N_at*3*100))
    print("1 Bonds O : {}, {:3.3f}%".format(num_o_1,num_o_1/N_at*3/2*100))
    print("3 Bonds O : {}, {:3.3f}%".format(num_o_3,num_o_3/N_at*3/2*100))



    hist_Dens, radius = np.histogram(Dist_list,hist_Dens[1])

    if density:
        #get the sliding average of the edges
        radius = ((np.roll(radius,1) + radius) / 2)[1:]
        dr = radius[1] - radius[0]
        hist_Dens = hist_Dens / (4*np.pi*radius**2*dr) / np.sum(Types==1)
    else:
        # hist_Dens = hist_Dens / np.sum(Types==1)
        # hist_Dens = np.histogram(Dens_trunc,hist_Dens[1])[0]
        pass
    hist_Si = np.histogram(Si_count_O,hist_Si[1])[0]
    hist_O = np.histogram(O_count_Si,hist_O[1])[0]
    # hist_H = np.histogram(O_count_H,hist_H[1])[0]


    # Counts = [np.sum(Si_count_O), np.sum(O_count_Si), np.sum(O_count_H)]
    Counts = [np.sum(Si_count_O), np.sum(O_count_Si)]
    print(Counts)

    return [hist_Dens,hist_Si,hist_O],Counts



def plot_analysis(Counts_Hists,Counts,hist_Dens,hist_Si,hist_O):
    """
    plot_analysis(Counts_Hists,Counts,hist_Dens,hist_Si,hist_O)

    Function used by analyze_mult. It updates the histogram for the RDF, and saturation of atoms
    """
    Counts_Hist_Dens = Counts_Hists[0]
    for count, rect in zip(Counts_Hist_Dens,hist_Dens[2].patches):
        rect.set_height(count)

    # Si_count_O, O_count_Si, O_count_H = Counts
    Si_count_O, O_count_Si = Counts


    Counts_Hist_Si = Counts_Hists[1]
    for count, rect in zip(Counts_Hist_Si,hist_Si[2].patches):
        rect.set_height(count)
    hist_Si[2].set_label("Si-O Bonds {}".format(Counts[0]))

    # if Counts[2]>0:
    if 0:
        Counts_Hist_O = Counts_Hists[2]
        for count, rect in zip(Counts_Hist_O,hist_O[2].patches):
            rect.set_height(count)

        Counts_Hist_H = Counts_Hists[3]
        Counts_Hist_H[0]=0
        for count, rect in zip(Counts_Hist_H,hist_H[2].patches):
            rect.set_height(count)

        hist_O[2].set_label("O-Si Bonds {}".format(Counts[1]))
        hist_H[2].set_label("O-H Bonds {}".format(Counts[2]))

    else:
        Counts_Hist_O = Counts_Hists[2]
        for count, rect in zip(Counts_Hist_O,hist_O[2].patches):
            rect.set_height(count)
        hist_O[2].set_label("O-Si Bonds {}".format(Counts[1]))


def analyze_mult(list_Tstep,list_Pos,list_Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,rdf_max=5,periodic=False,Lims=[],anim=False,save=False,vline=1.609,density=True):
    """
    analyze_mult(list_Tstep,list_Pos,list_Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,rdf_max=5,periodic=False,Lims=[],anim=False,save=False,vline=1.609)

    This functions computes the RDF and saturation and plots them on a graph. It supports multiple data, and there is a slider to navigate between the different timesteps

    Parameters
    ----------
        list_Tstep : list
            The list of the timesteps to plot multiple graphs
        list_Pos : list
            The list of the position for all the timesteps
        list_Types : list
            The list of the type for all the timesteps
        threshold_Si : float, optional
            The threshold used to consider if Si and O are bonding. 2 by default
        threshold_O : float, optional
            The threshold used to consider if O and Si are bonding. 2 by default
        threshold_H : float, optional
            The threshold used to consider if O and H are bonding. 1.3 by default
        rdf_max : float, optional
            The maximum radius computed for the rdf. 5 by default
        periodic : bool, optional
            Is the system periodic in z. False by default
        Lims : list, optional
            The boundaries of the system used for the periodicity. [] by default
        anim : bool, optional
            Do an animation of the rdf and saturation with respect to the timestep. False by default
        save : bool, optional
            Save the graph in a svg graph instead of showing it. will only save the last timestep. False by default
        vline : float, optional
            Adds a vertical line to the graph. 1.609 by default

    Returns
    -------
        None
            A graph is produced

    """
    fig,ax = plt.subplots()
    if save:
        fig,ax = plt.subplots(figsize=(8,6),dpi=200)
        plt.rcParams.update({'font.size': 15})
        plt.rcParams['svg.fonttype'] = 'none'

    plt.axis("off")

    if (not save) and (len(list_Tstep)!=1):

        fig.subplots_adjust(bottom=0.25)
        ax_slider = fig.add_axes([0.25,0.1,0.65,0.03])
        slider = Slider(ax=ax_slider,label="Timestep",valmin=list_Tstep[0],valmax=list_Tstep[-1],valinit=list_Tstep[0],valfmt="%d",valstep=list_Tstep)
    num_O = np.sum(list_Types[0] == 2)
    num_Si = np.sum(list_Types[0] == 1)

    purple = np.array([96,25,255])/255
    dark_purple = np.array([56,20,180])/255
    dark_dark_purple = np.array([34,10,120])/255

    plt.subplot(2,1,1)
    hist_Dens = plt.hist(np.array([]),bins=100,range=(0,5),color=purple,edgecolor=dark_purple,linewidth=1,label="A")
    plt.title("RDF Si-O",color=dark_purple)
    plt.ylabel("Number",color=dark_purple)
    plt.xlabel("Distance (A)",color=dark_purple)
    plt.xticks([k for k in range(int(rdf_max))]+[vline],[k for k in range(int(rdf_max))]+[vline],color=purple)
    plt.yticks(color=purple)
    # vline = (int(vline*40) +1)/40
    plt.axvline(vline,color=dark_dark_purple)
    plt.xlim(0,5)
    if density: plt.ylim(0,2)
    else: plt.ylim(0,num_O*1.5)

    plt.subplot(2,2,3)
    hist_Si = plt.hist(np.array([]),bins=12,range=(0,6),color=purple,edgecolor=dark_purple,linewidth=1,label="A")
    plt.title("Number of Bonds for Si",color=dark_purple)
    plt.xlabel("Number of Bonds",color=dark_purple)
    plt.ylabel("Number of Si",color=dark_purple)
    plt.xticks([k+0.25 for k in range(7)],[k for k in range(7)],color=purple)
    plt.yticks(color=purple)
    plt.ylim(0,num_Si*1.2)


    plt.subplot(2,2,4)
    hist_O = plt.hist(np.array([]),bins=12,range=(0,6),color=purple,edgecolor=dark_purple,linewidth=1,label="A")
    # hist_H = plt.hist(np.array([]),bins=12,range=(0,6),color="blue",edgecolor="darkblue",linewidth=1,label="A")
    plt.xticks([k+0.25 for k in range(7)],[k for k in range(7)],color=purple)
    plt.yticks(color=purple)
    plt.title("Number of Bonds for O",color=dark_purple)
    plt.xlabel("Number of Bonds",color=dark_purple)
    plt.ylabel("Number of O",color=dark_purple)
    plt.ylim(0,num_O*1.2)

    if anim:
        plt.show(block=False)



    list_Counts_Hists,list_Counts = [], []
    for tstep in range(len(list_Tstep)):
        Hist_Counts,Counts = compute_analysis(list_Pos[tstep],list_Types[tstep],hist_Dens,hist_Si,hist_O,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,periodic=periodic,Lims=Lims,rdf_max=rdf_max,density=density)
        # Hist_Counts,Counts = compute_analysis(list_Pos[tstep],list_Types[tstep],hist_Dens,hist_Si,hist_O,hist_H,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,periodic=periodic,Lims=Lims,rdf_max=rdf_max)
        if anim:
            slider.set_val(list_Tstep[tstep])
            # plot_analysis(Hist_Counts,Counts,hist_Dens,hist_Si,hist_O,hist_H)
            plot_analysis(Hist_Counts,Counts,hist_Dens,hist_Si,hist_O)
            plt.pause(0.02)

        list_Counts_Hists.append(Hist_Counts)
        list_Counts.append(Counts)


    def update(val):
        index = list_Tstep.index(val)
        # plot_analysis(list_Counts_Hists[index],list_Counts[index],hist_Dens,hist_Si,hist_O,hist_H)
        plot_analysis(list_Counts_Hists[index],list_Counts[index],hist_Dens,hist_Si,hist_O)
        plt.draw()

    if not anim and not save and len(list_Tstep)!=1:
        slider.on_changed(update)
    if not anim and not save:
        update(list_Tstep[0])
        plt.show()

    if save:
        update(list_Tstep[-1])
        plt.tight_layout()
        plt.savefig("analysis.svg")



def analyze_defects(Pos,Types,periodic=False,Lims=[],Cycles=None,L_cycles=None,d_spacing=5,isovalue=1.,alpha=2.,prec=20,d=10,length_box=20,smoothing=1000,N_th=8):
    """
    analyze_defects(Pos,Types,periodic=False,Lims=[],Cycles=None,L_cycles=None)


    """
    D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=periodic,Lims=Lims,rdf_max=5)

    plotter = pv.Plotter()
    plotter.add_axes()

    sp = pv.Sphere(radius=0.6)
    Colors_Si = ["white","powderblue","lightsteelblue","dodgerblue","blue","navy","black"]
    for number_bonds in range(0,7):
        op = 1.0
        if number_bonds == 4: pass
        else:
            Si_bonds = (Si_count_O==number_bonds)
            if Si_bonds.any():
                data = pv.PolyData(Pos[Types==1][Si_bonds])
                pc = data.glyph(scale=False,geom=sp,orient=False)
                plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_Si[number_bonds],name="Si_{}".format(number_bonds))
            else: plotter.remove_actor("Si_{}".format(number_bonds))


    sp = pv.Sphere(radius=0.4)
    Colors_O = ["white","orange","red","darkred","black"]
    for number_bonds in range(0,5):
        op = 1.0
        if number_bonds==2: pass
        else:
            # if O_count_H.any():
            #     O_bonds = ((O_count_Si + O_count_H)==number_bonds)
            # else:
            O_bonds = (O_count_Si==number_bonds)
            # print(np.shape(O_count_Si),np.shape(Pos[((Types==2)+(Types==3)).astype("bool")]))

            if O_bonds.any():
                data = pv.PolyData(Pos[((Types==2)).astype("bool")][O_bonds])
                pc = data.glyph(scale=False,geom=sp,orient=False)
                plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_O[number_bonds],name="O_{}".format(number_bonds))
            else: plotter.remove_actor("O_{}".format(number_bonds))

    num_si_3 = np.sum(Si_count_O==3)
    num_si_5 = np.sum(Si_count_O==5)
    num_si_6 = np.sum(Si_count_O==6)
    num_o_1 = np.sum(O_count_Si==1)
    num_o_3 = np.sum(O_count_Si==3)


    print("3 Bonds Si : {}".format(num_si_3))
    print("5 Bonds Si : {}".format(num_si_5))
    print("6 Bonds Si : {}".format(num_si_6))
    print("1 Bonds O : {}".format(num_o_1))
    print("3 Bonds O : {}".format(num_o_3))

    # sp = pv.Sphere(radius=0.2)
    # Colors_H = ["white","gray","black"]
    # data = pv.PolyData(Pos_cutted[Types_cutted==4])
    # pc = data.glyph(scale=False,geom=sp,orient=False)
    # plotter.add_mesh(pc,opacity=0.05,pbr=True,roughness=.5,metallic=.2,color="gray",name="H")


    if not type(Cycles) is type(None) and not type(L_cycles) is type(None):
        Pos_Si_l = Pos[Types_cutted_l==1]
        Ind_Si_cut = Pos[Types==1]
        Ind_sum = np.cumsum(Ind_Si_cut)-1
        tubes_cycles = [pv.MultiBlock() for k in range(np.max(L_cycles)+1)]

        for cycle,l_cycle in zip(Cycles,L_cycles):
            cycle2 = np.roll(cycle,1)
            for cycle_1,cycle_2 in zip(cycle,cycle2):
                if Ind_Si_cut[cycle_1] and Ind_Si_cut[cycle_2]:

                    # print(cycle_1,Ind_sum[cycle_1])
                    cycle_1 = Ind_sum[cycle_1]
                    cycle_2 = Ind_sum[cycle_2]
                    tubes_cycles[l_cycle].append(pv.Tube(Pos_Si_l[cycle_1],Pos_Si_l[cycle_2],n_sides=5,radius=0.5))

        colors = ["","","white","gray","green","red","pink","blue","purple","yellow","black","orange","magenta"]
        if len(L_cycles) > len(colors):
            colors = colors + colors[2:]

        for multblock,color,n in zip(tubes_cycles,colors,np.arange(len(tubes_cycles))):
            if multblock.n_blocks:
                mesh = multblock.combine()
                if n==6: opacity = 0.2
                else: opacity = 0.3
                plotter.add_mesh(mesh,opacity=opacity,pbr=True,roughness=.5,metallic=.8,color=color,name="cycles_{}".format(n))
            else: plotter.remove_actor("cycles_{}".format(n))

    Lx,Ly,Lz = np.max(Pos,axis=0) + d
    lx,ly,lz = np.min(Pos,axis=0) - d


    Nx = int(round((Lx-lx+2*d)/d_spacing))+1
    Ny = int(round((Ly-ly+2*d)/d_spacing))+1
    Nz = int(round((Lz-lz+2*d)/d_spacing))+1

    grid = pv.ImageData(dimensions=(Nx,Ny,Nz),origin=(lx-d,ly-d,lz-d),spacing=(d_spacing,d_spacing,d_spacing))
    Lims = [[Lx,lx],[Ly,ly],[Lz,lz]]

    cube = compute_quick_surface(Pos,grid,Lims,alpha=alpha,prec=prec,d=d,length_box=length_box,N_th=8)
    contour = grid.contour(isosurfaces=(isovalue),scalars=cube)
    if smoothing != 0:
        smooth = contour.smooth(n_iter=int(smoothing))
    else: smooth = contour
    curv = smooth.curvature(curv_type="mean")

    curv_sorted_trunc = np.sort(curv)[len(curv)//10:len(curv) - len(curv)//10]
    L = np.arange(len(curv_sorted_trunc))
    a,b = np.polyfit(L,curv_sorted_trunc,1)
    cmax = a * len(curv) + b

    plotter.add_mesh(smooth,name="contour",opacity=0.1,pbr=True,roughness=.5,metallic=.2,color="red")

    plotter.show()


def save_defects(file_name,Pos,Types,periodic=False,Lims=[]):
    """
    analyze_defects(Pos,Types,periodic=False,Lims=[],Cycles=None,L_cycles=None)


    """
    D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=periodic,Lims=Lims,rdf_max=5)


    Count_Si_0 = (Si_count_O==0)
    Count_Si_1 = (Si_count_O==1)
    Count_Si_2 = (Si_count_O==2)
    Count_Si_3 = (Si_count_O==3)
    Count_Si_4 = (Si_count_O==4)
    Count_Si_5 = (Si_count_O==5)
    Count_Si_6 = (Si_count_O==6)
    Count_Si_7 = (Si_count_O==7)
    Count_Si_8 = (Si_count_O==8)

    Count_O_0 = (O_count_Si==0)
    Count_O_1 = (O_count_Si==1)
    Count_O_2 = (O_count_Si==2)
    Count_O_3 = (O_count_Si==3)
    Count_O_4 = (O_count_Si==4)
    Count_O_5 = (O_count_Si==5)
    Count_O_6 = (O_count_Si==6)

    Types[Types==1] = Count_Si_4 * 1 + Count_Si_0 * 5 + Count_Si_1 * 6 + Count_Si_2 * 7 + Count_Si_3 * 8 + Count_Si_5 * 9 + Count_Si_6 * 10 + Count_Si_7 * 11 + Count_Si_8 * 12
    Types[Types==2] = Count_O_2 * 2 + Count_O_0 * 13 + Count_O_1 * 14 + Count_O_3 * 15 + Count_O_4 * 16 + Count_O_5 * 17 + Count_O_6 * 18

    write_xyz(file_name,Pos,Types)






def compute_quick_surface(Pos,grid,Lims,alpha=2,prec=20,d=10,length_box=20,N_th=8):
    """
    compute_quick_surface(Pos,grid,Lims,alpha=2,prec=20,d=10,length_box=20,N_th=8)

    This function is used to compute surface like in the quicksurf of VMD
    The surface is computed using sum of gaussians centered on each atom
    \\rho = \\sum e^{-\\frac{\\abs{r-r_i}^2}{2\\alpha^2}}
    For more information see : https://doi.org/10.2312/PE/EuroVisShort/EuroVisShort2012/067-071

    To accelerate the computation, the system is divided in multiple boxes in which the distance between
    the atoms and the grid is computed. Each box is larger for the atoms than for the grid in order not
    to have border effects.

    It also supports multithreading

    Parameters
    ----------
        Pos : list
            The position of the atoms
        grid : pyvista.core.grid.ImageData
            The grid on which the surface will be computed
        Lims : list
            The boundaries of the system
        alpha : float, optional
            the coefficient for the radius of the guassian function. 2 by default
        prec : float, optional
            the amount by which each box is enlargened for the atoms. 20 by default
        d : float, optional
            The amount by which the limit coordinates of the system are expanded. 10 by default
        length_box : float, optional
            The larger of the box used to divide the system. 20 by default
        N_th : int, optional
            The number of threads used to compute the surface. 8 by default

    Returns
    -------
        cube : list
            The cube of the surface
    """

    import threading as th
    x,y,z = grid.points.T
    Pos_Grid = grid.points



    Lx,lx = Lims[0]
    Ly,ly = Lims[1]
    Lz,lz = Lims[2]

    Nx_box = int((Lx-lx+2*d)/length_box) + 1
    Ny_box = int((Ly-ly+2*d)/length_box) + 1
    Nz_box = int((Lz-lz+2*d)/length_box) + 1

    # import time
    def evaluate_surface(cube,Nx_box,Ny_box,Nz_box_list):
        # A,B = 0, 0
        for z_box in range(Nz_box_list[0],Nz_box_list[1]):
            for y_box in range(Ny_box):
                for x_box in range(Nx_box):
                    # at=time.time()

                    Pos_box_x = ((x) >= (x_box * length_box+lx-d)) * ((x) <= ((x_box+1) * length_box+lx-d))
                    Pos_box_y = ((y) >= (y_box * length_box+ly-d)) * ((y) <= ((y_box+1) * length_box+ly-d))
                    Pos_box_z = ((z) >= (z_box * length_box+lz-d)) * ((z) <= ((z_box+1) * length_box+lz-d))
                    Ind_Box = Pos_box_x * Pos_box_y * Pos_box_z

                    Pos_Box = Pos_Grid[Ind_Box]

                    Pos_trunc_x = ((Pos[:,0]) >= (x_box * length_box - prec+lx-d)) * ((Pos[:,0]) <= ((x_box+1)*length_box + prec+lx-d))
                    Pos_trunc_y = ((Pos[:,1]) >= (y_box * length_box - prec+ly-d)) * ((Pos[:,1]) <= ((y_box+1)*length_box + prec+ly-d))
                    Pos_trunc_z = ((Pos[:,2]) >= (z_box * length_box - prec+lz-d)) * ((Pos[:,2]) <= ((z_box+1)*length_box + prec+lz-d))

                    Ind_trunc = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                    Pos_trunc = Pos[Ind_trunc]
                    # bt=time.time()

                    Dist = sd.cdist(Pos_Box, Pos_trunc)

                    cube[Ind_Box] = np.einsum("ij->i",np.exp(-Dist**2/2/alpha**2))
                    # ct=time.time()
        #             A+=bt-at
        #             B+=ct-bt
        # print(A,B,Nz_box_list)

    cube = np.zeros((len(Pos_Grid)))

    Nz_list = [int(Nz_box/N_th) for k in range(N_th+1)]

    for j in range(Nz_box%N_th):
        Nz_list[j] += 1
    # print(Nz_list)

    Nz_list_list = []
    count = 0
    for k in range(len(Nz_list)-1):
        Nz_list_list.append([count ,count + Nz_list[k]])
        count = count + Nz_list[k]
    # print(Nz_list_list)
    list_th = []
    for Nz_box_list in Nz_list_list:
        if Nz_box_list[0] != Nz_box_list[1]:
            thread = th.Thread(target=evaluate_surface,args=(cube,Nx_box,Ny_box,Nz_box_list))
            list_th.append(thread)
            thread.start()

    for thread in list_th:
        thread.join()
    return cube


def curvature_analysis(Pos,prec=20,d=10,length_box=20):
    """
    Putting prec too low and/or length box too high might result in an apparition of "cubes" in the shape

    """
    Lx,Ly,Lz = np.max(Pos,axis=0) + d
    lx,ly,lz = np.min(Pos,axis=0) - d


    class _slider:
        def __init__(self,func,d_spacing,alpha,isovalue,smoothing,lims):
            self.func = func
            self.d_spacing = float(d_spacing)
            self.alpha = float(alpha)
            self.isovalue = float(isovalue)
            self.smoothing = float(smoothing)
            self.lims = float(lims)
            self.cube = None

        def __call__(self,called,value):
            if called=="d_spacing":
                self.d_spacing = value
                self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims)
            elif called=="alpha":
                self.alpha = value
                self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims)
            elif called=="isovalue":
                self.isovalue = value
                self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,cube=self.cube)
            elif called == "smoothing":
                self.smoothing = value
                self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,cube=self.cube)
            elif called == "lims":
                self.lims = value
                self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,cube=self.cube)


    def slider_precision(d_spacing,alpha,isovalue,smoothing,lims,cube=None):

        Nx = int(round((Lx-lx+2*d)/d_spacing))+1
        Ny = int(round((Ly-ly+2*d)/d_spacing))+1
        Nz = int(round((Lz-lz+2*d)/d_spacing))+1

        grid = pv.ImageData(dimensions=(Nx,Ny,Nz),origin=(lx-d,ly-d,lz-d),spacing=(d_spacing,d_spacing,d_spacing))
        Lims = [[Lx,lx],[Ly,ly],[Lz,lz]]
        if type(cube) is type(None):
            #No cube were provided, therefore compute the whole cube again
            #This is used to not recompute the cube everytime you change the isovalue
            cube = compute_quick_surface(Pos,grid,Lims,alpha=alpha,prec=prec,d=d,length_box=length_box,N_th=8)
        contour= grid.contour(isosurfaces=(isovalue),scalars=cube)
        if smoothing != 0:
            smooth = contour.smooth(n_iter=int(smoothing))
            # smooth = contour.smooth_taubin(n_iter=int(smoothing),pass_band=0.01)

        else: smooth = contour
        # curv = smooth.curvature(curv_type="minimum")
        # curv = smooth.curvature(curv_type="maximum")
        curv_min = smooth.curvature(curv_type="minimum")
        curv_max = smooth.curvature(curv_type="maximum")


        curv_max_sorted_trunc = np.sort(curv_max)[len(curv_max)//10:len(curv_max) - len(curv_max)//10]
        L = np.arange(len(curv_max_sorted_trunc))
        a,b = np.polyfit(L,curv_max_sorted_trunc,1)
        c_max_max = a * len(curv_max) + b

        curv_min_sorted_trunc = np.sort(curv_min)[len(curv_min)//10:len(curv_min) - len(curv_min)//10]
        L = np.arange(len(curv_min_sorted_trunc))
        a,b = np.polyfit(L,curv_min_sorted_trunc,1)
        c_min_max = a * len(curv_min) + b


        norm = max([c_max_max,c_min_max]) * lims

        curv_min = curv_min / norm
        curv_max = curv_max / norm

        colors = np.zeros((len(curv_min),3))


        for index,c_min,c_max in zip(range(len(curv_min)),curv_min,curv_max):
            if c_max > 1: c_max = 1
            if c_min > 1: c_min = 1
            if c_max < -1: c_max = -1
            if c_min < -1: c_min = -1

            colors[index,0] = 1 - abs((c_max*2 + abs(c_min) - c_min)/4) - (abs(c_min+c_max) + (c_min + c_max))/4
            colors[index,1] = 1 - abs((abs(c_min+c_max)-(c_min+c_max))/4) - (abs(c_min+c_max) + (c_min + c_max))/4
            colors[index,2] = 1 - abs((abs(c_min+c_max)-(c_min+c_max))/4) - abs((c_max*2 + abs(c_min) - c_min)/4)

            # if np.sum(colors[index]) < 0:
            #     print(c_min,c_max)

        #Remove <0 and >1
        colors = (colors > 0) * (colors < 1) * colors + (colors>=1) * 1

        # C = np.sum(colors,axis=1)
        # print(np.min(C),np.max(C))
        import time
        a=time.time()
        colors_blur = np.zeros(np.shape(colors))
        Pos_points_surface = smooth.points
        x,y,z = Pos_points_surface.transpose()

        for pos, color, color_index in zip(Pos_points_surface,colors,range(len(colors))):
            pos_x,pos_y,pos_z = pos
            Pos_box_x = ((x) >= (pos_x - length_box/2)) * ((x) <= ((pos_x) + length_box/2))
            Pos_box_y = ((y) >= (pos_y - length_box/2)) * ((y) <= ((pos_y) + length_box/2))
            Pos_box_z = ((z) >= (pos_z - length_box/2)) * ((z) <= ((pos_z) + length_box/2))

            Pos_close = Pos_points_surface[Pos_box_x*Pos_box_y*Pos_box_z]
            Colors_close = colors[Pos_box_x*Pos_box_y*Pos_box_z]

            Dist_close =  sd.cdist([pos],Pos_close)[0]
            # print(Dist_close)
            Dist_close = (Dist_close<1)*1 + (Dist_close>1)*Dist_close


            colors_blur[color_index] = np.sum(Colors_close.transpose()/Dist_close**2,axis=1) / np.sum(1/Dist_close**2)
        print(time.time()-a)



        # plotter.add_mesh(smooth,name="contour",opacity=1.0,scalars=colors,rgb=True)
        plotter.add_mesh(smooth,name="contour",opacity=1.0,scalars=colors_blur,rgb=True)
        # plotter.add_mesh(smooth,name="contour",opacity=1.0,scalars=curv_min,clim=(-norm,norm))

        return cube


    plotter = pv.Plotter()
    plotter.add_background_image("lims.png",auto_resize=False)


    slider = _slider(slider_precision,5,3,1,0,1)

    plotter.add_slider_widget(lambda value: slider("d_spacing",value), [0.5, 10],value=5.,title="Spacing", fmt="%1.1f",pointa=(0.01,.9),pointb=(0.2,.9),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("alpha",value), [0.5, 10],value=3.,title="Alpha", fmt="%1.1f",pointa=(0.01,.8),pointb=(0.2,.8),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("isovalue",value), [0.5, 20],value=1.,title="Isovalue", fmt="%1.1f",pointa=(0.01,.7),pointb=(0.2,.7),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("smoothing",value), [0, 1000],value=0,title="Smoothing", fmt="%1.1f",pointa=(0.01,.6),pointb=(0.2,.6),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("lims",value), [0.5, 2],value=1.,title="Lims", fmt="%1.1f",pointa=(0.01,.5),pointb=(0.2,.5),slider_width=0.02,title_height=0.02)


    plotter.show(full_screen=False)

def analyze_density(Pos,periodic=False,Lims=[],d_spacing=5,isovalue=5.,alpha=2.,prec=20,d=10,length_box=20,smoothing=1000,N_th=8,rdf_max=20):
    """
    analyze_defects(Pos,Types,periodic=False,Lims=[],Cycles=None,L_cycles=None)


    """

    plotter = pv.Plotter()
    plotter.add_axes()


    Lx,Ly,Lz = np.max(Pos,axis=0) + d
    lx,ly,lz = np.min(Pos,axis=0) - d


    Nx = int(round((Lx-lx+2*d)/d_spacing))+1
    Ny = int(round((Ly-ly+2*d)/d_spacing))+1
    Nz = int(round((Lz-lz+2*d)/d_spacing))+1

    grid = pv.ImageData(dimensions=(Nx,Ny,Nz),origin=(lx-d,ly-d,lz-d),spacing=(d_spacing,d_spacing,d_spacing))
    Lims = [[Lx,lx],[Ly,ly],[Lz,lz]]

    cube = compute_quick_surface(Pos,grid,Lims,alpha=alpha,prec=prec,d=d,length_box=length_box,N_th=N_th)
    contour = grid.contour(isosurfaces=(isovalue),scalars=cube)
    if smoothing != 0:
        smooth = contour.smooth(n_iter=int(smoothing))
    else: smooth = contour

    Pos_Surface = smooth.points
    Density = np.zeros((len(Pos_Surface)))
    for point,density_index in zip(Pos_Surface,range(len(Density))):
        x,y,z = point
        # Pos_trunc = Pos[(Pos[:,0] > x-rdf_max) * (Pos[:,0] < x+rdf_max) * (Pos[:,1] > y-rdf_max) * (Pos[:,1] < y+rdf_max) * (Pos[:,2] > z-rdf_max) * (Pos[:,2] < z+rdf_max)]
        Pos_trunc = (Pos[:,0] > x-rdf_max) * (Pos[:,0] < x+rdf_max) * (Pos[:,1] > y-rdf_max) * (Pos[:,1] < y+rdf_max) * (Pos[:,2] > z-rdf_max) * (Pos[:,2] < z+rdf_max)

        # Dist = sd.cdist([point],Pos_trunc)
        # Density[density_index] = np.einsum("ij->",4*np.pi*Dist**2)
        Density[density_index] = np.sum(Pos_trunc)



    print(np.min(Density),np.max(Density))
    Density  = Density / np.max(Density)
    plotter.add_mesh(smooth,name="contour",opacity=1.0,cmap="cool",scalars=Density,scalar_bar_args={"title":"Density"})

    plotter.show()



def transfo_inv(Pos_transfo,D,P,slice_thickness=5):
    """
    This function computes the inverse transformation of a list of transformed position
    """

    #Because of the sliding due to the issue with periodicity
    #The basis of the helix is often not at z = 0
    Pos_slice = Pos_transfo[Pos_transfo[:,2] < (slice_thickness + np.min(Pos_transfo[:,2]))]
    mean_slice = np.mean(Pos_slice)


    mean = np.mean(Pos_transfo,axis=0)
    # print(mean)
    mean[2] = 0
    u,v,w = (Pos_transfo - mean).transpose()
    # u,v,w = (Pos_transfo).transpose()

    # u,v,w = (Pos_transfo).transpose()
    #if unlucky and a point has u=0. Should not happen because if would mean that there are no holes inside
    u = u + (u==0) * 0.00001
    P_t = P/2/np.pi
    R = D
    print('R',R)
    # R = 29.133407802862774
    # P_t = 32.677991490301196
    # R=40




    Lim = R/P_t**2 * (v**2+u**2)**(1/2)
    Lim_m = w/P_t - Lim
    Lim_M = w/P_t + Lim
    Nz = 500
    z = np.linspace(Lim_m,Lim_M, Nz)
    # z = z%(2*np.pi)



    #z is found by finding the roots of this equation
    equation_z = abs(P_t*z - R/P_t * (np.sign(u) * (v**2+u**2)**(1/2) * np.cos((z + np.arctan(v/u)))) - w)
    roots_z_ind = []
    for curv in equation_z.transpose():
        ind_mins = np.array(sps.argrelmin(curv,mode="wrap")[0])
        values_mins = curv[ind_mins]

        # ind_mins = ind_mins[values_mins<0.5]
        #
        # if len(ind_mins)==0:
        #     plt.plot(curv,"or")
        #     plt.show()

        ind_mins_center = abs(ind_mins-Nz//2)
        # print(ind_mins,ind_mins_center)
        roots_z_ind_curv = ind_mins[np.argmin(ind_mins_center)]
        roots_z_ind.append(roots_z_ind_curv)

    roots_z_ind = np.array(roots_z_ind)



    # equation_z = abs(P_t*z + R/P_t * (np.sin(z)*v - np.cos(z)*u)  - w)
    # roots_z_ind = np.argmin(equation_z,axis=0)
    # print(np.shape(roots_z_ind))
    # print("Max err",np.max(np.min(equation_z,axis=0)))


    roots_z = np.take_along_axis(z,np.expand_dims(roots_z_ind,0),axis=0)[0]
    print("Max err",np.max(abs(P_t*roots_z - R/P_t * (np.sign(u) * (v**2+u**2)**(1/2) * np.cos((roots_z + np.arctan(v/u)))) - w)))

    # roots_z = roots_z%(2*np.pi)
    # roots_z = roots_z + np.min(roots_z) + np.pi

    N = (R**2 + P_t**2)**(1/2)
    print("N",N)
    print("P_t",P_t)

    # print(np.min(z),np.max(z))
    x = N/P_t * (-np.cos(roots_z)*u + np.sin(roots_z)*v)
    y = R - np.sin(roots_z)*u - np.cos(roots_z)*v
    # y = R - np.sin(roots_z)*u/2 - np.cos(roots_z)*v/2

    # print(np.max(u),np.max(v))
    # print(np.max(x),np.max(y))

    d = np.argmax(x)
    # print(np.shape(equation_z))
    # plt.plot(z[:,d],equation_z[:,d],"or")
    # plt.show()

    return x,y,roots_z * P_t





def evaluate_dimenions(Pos_transfo,Types,slice_thickness=5,threshold_cut=0.01,threshold_lr=2,show=False):
    def Rz(alpha):
        """3D Rotation matrix around the z axis"""
        return np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
    def Ry(beta):
        """3D Rotation matrix around the y axis"""
        return np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
    def Rx(gamma):
        """3D Rotation matrix around the x axis"""
        return np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])
    def Rota(alpha):
        return np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])



    def calc_dist(Pos,xa,ya,xb,yb):
        """Compute the distance between positions and the segment drawn by two points"""
        Vec = np.array([xb-xa,yb-ya])

        Vec_d_a = Pos - np.array([xa,ya])
        Vec_d_b = Pos - np.array([xb,yb])

        D_a = np.linalg.norm(Vec_d_a,axis=1)
        D_b = np.linalg.norm(Vec_d_b,axis=1)

        Vec = Vec.reshape((1,2))

        D_a_sup_b = (D_a > D_b).reshape((len(D_a),1))
        D_b_sup_a = (D_a <= D_b).reshape((len(D_b),1))

        Vec_calc = Vec * D_b_sup_a - Vec * D_a_sup_b
        Vec_d_calc = Vec_d_a * D_b_sup_a + Vec_d_b * D_a_sup_b

        #Compute the angle to know if the orthogonal of the line containing the Pos is in the segment or not
        cos_angle = np.sum(Vec_d_calc * Vec_calc,axis=1) / np.linalg.norm(Vec_d_calc,axis=1) / np.linalg.norm(Vec_calc,axis=1)
        cos_angle = (cos_angle > 1) * 1 - (cos_angle < -1) * 1 + cos_angle * (cos_angle <= 1) * (cos_angle >= -1)
        angle = np.arccos(cos_angle)

        Dist = (angle < np.pi/2) * np.linalg.norm(Vec_d_calc,axis=1) * abs(np.sin(angle)) + (angle >= np.pi/2) * np.min(np.array([D_a,D_b]),axis=0)


        return Dist

    ##Compute the pitch as the difference between max z - min z
    N_points = len(Pos_transfo)
    min_z = np.min(Pos_transfo[:,2])
    pitch = np.max(Pos_transfo[:,2]) - np.min(Pos_transfo[:,2]) #Not the perfect formula as empty space, but close enough
    N_z = int((pitch // slice_thickness)+1)


    if False:
        #Compute the diameter through the max distance of the flattened helix
        Pos_centered = Pos_transfo - np.mean(Pos_transfo,axis=0)
        Pos_centered[:,2] = 0
        Dist = np.sort(np.sum(Pos_centered**2,axis=1)**(1/2))
        Dist = Dist[int(threshold_cut * N_points):int(1-threshold_cut * N_points)]
        diameter = np.max(Dist)*2
        thickness = np.max(Dist) - np.min(Dist)


        #Compute the diameters using a linear regression
        a,b = np.polyfit(np.arange(len(Dist)),Dist,1)
        diameter = (a*N_points+b) * 2
        thickness = a*N_points


    ##Recenter the positions
    mean = np.mean(Pos_transfo,axis=0)
    mean[2] = 0
    Pos_transfo = Pos_transfo - mean
    Means = []

    ##Compute the factor D as the average radius with the center
    for slice_helix in range(N_z):
        Pos_slice = Pos_transfo[(Pos_transfo[:,2] >= slice_helix*slice_thickness + min_z) * (Pos_transfo[:,2] < (slice_helix+1)*slice_thickness + min_z)]
        radius = np.sum(Pos_slice[:,:2]**2,axis=1)**(1/2)
        mean = np.mean(radius,axis=0)
        Means.append(np.sum(mean**2)**(1/2))


    D = np.mean(Means)

    ##Do the invert transformation to get the slices correctly for the width
    x,y,z = transfo_inv(Pos_transfo,D,pitch)
    Pos = np.array([x,y,z]).transpose()

    center = np.mean(Pos_transfo,axis=0)


    #Do translation to have whole slices using periodic conditions
    Pos_under_z = (Pos[:,2] < min_z).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos + np.array([0,0,pitch]))
    Pos_over_z = (Pos[:,2] > (min_z+pitch)).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos - np.array([0,0,pitch]))
    Pos_middle_z = ((Pos[:,2] >= min_z) * (Pos[:,2] <= (min_z + pitch))).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos)

    Pos_transfo_under_z = (Pos[:,2] < min_z).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos_transfo + np.array([0,0,pitch]))
    Pos_transfo_over_z = (Pos[:,2] > (min_z+pitch)).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos_transfo - np.array([0,0,pitch]))
    Pos_transfo_middle_z = ((Pos[:,2] >= min_z) * (Pos[:,2] <= (min_z + pitch))).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos_transfo)

    Pos_transfo = Pos_transfo_under_z + Pos_transfo_middle_z + Pos_transfo_over_z
    Pos = Pos_under_z + Pos_middle_z + Pos_over_z
    x,y,z = Pos.transpose()


    W_int, W_ext = [],[]
    T_int, T_ext = [],[]
    W = []
    Surf = []
    Surf_tot = []
    T = []

    for slice_helix in range(N_z):
    # def slide(value):
    #     slice_helix = int(value)
        ##Get slices and center them
        Pos_slice = Pos_transfo[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]
        Diff_slice_mean = center - np.mean(Pos_slice,axis=0)
        Pos_slice = Pos_slice - np.mean(Pos_slice,axis=0)

        ## rotate to face the same direction
        angle_z = np.arctan2(Diff_slice_mean[1],Diff_slice_mean[0])
        Pos_slice = Pos_slice.dot(Rz(angle_z))

        Num_S = len(Pos_slice)

        ##get the surface to rotate to be on the xy plane
        def plane_surf(X,a,b):
            x,y = X.transpose()
            return a*x + b*y

        a,b = sp.optimize.curve_fit(plane_surf,Pos_slice[:,:2],Pos_slice[:,2])[0]

        Vec_1 = np.array([1,0,plane_surf(np.array([1,0]),a,b)])
        ar = pv.Arrow([0,0,0],Vec_1,scale=10)

        Vec_2 = np.array([0,1,plane_surf(np.array([0,1]),a,b)])
        ar = pv.Arrow([0,0,0],Vec_2,scale=10)
        angle_y = np.arctan2(Vec_1[0],Vec_1[2])
        Pos_slice = Pos_slice.dot(Ry(angle_y-np.pi/2))

        angle_x = np.arctan2(Vec_2[2],Vec_2[1])
        Pos_slice = Pos_slice.dot(Rx(angle_x))


        angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
        dist = np.linalg.norm(Pos_slice,axis=1)





        def square_fit_T(Pos,l,L,phi):
            Pos_corners = np.array([[-l/2, -L/2],[l/2,-L/2],[l/2,L/2],[-l/2,L/2]])
            Pos_corners_rota = Pos_corners.dot(Rota(phi))
            x1,y1,x2,y2,x3,y3,x4,y4 = Pos_corners_rota.ravel()

            Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
            Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
            Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
            Dist_4 = calc_dist(Pos,x4,y4,x1,y1)

            Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4])
            Min_dists = np.min(Dists,axis=0)
            return Min_dists


        angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0])+np.pi
        N_angles=64
        d_angle = 2*np.pi/N_angles
        mean_pos_angle=[]
        angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
        for angle in angles:
            selection = (angles >= angle) * (angles < (angle + d_angle))
            mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
        mean_pos_angle = np.array(mean_pos_angle)

        x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
        l,L,phi = x_max-x_min,y_max-y_min,0
        Bounds = np.array([[0,l],[0,L],[0,np.pi*2]]).transpose()

        l,L,phi = sp.optimize.curve_fit(square_fit_T,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=(l,L,phi),bounds=Bounds)[0]


        Pos_slice = Pos_slice.dot(Rz(-phi))


        Pos_x = Pos_slice[:,0][int(Num_S*threshold_cut):int(Num_S*(1-threshold_cut))]
        width = np.max(Pos_slice[:,1]) - np.min(Pos_slice[:,1])
        W.append(width)
        T.append(np.max(Pos_slice[:,0]) - np.min(Pos_slice[:,0]))





        Pos_slice = Pos_slice[:,:2]



        if False:
            def points(Pos,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x5,y5)
                Dist_5 = calc_dist(Pos,x5,y5,x6,y6)
                Dist_6 = calc_dist(Pos,x6,y6,x7,y7)
                Dist_7 = calc_dist(Pos,x7,y7,x8,y8)
                Dist_8 = calc_dist(Pos,x8,y8,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4,Dist_5,Dist_6,Dist_7,Dist_8])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists

            def points_arg(Pos,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x5,y5)
                Dist_5 = calc_dist(Pos,x5,y5,x6,y6)
                Dist_6 = calc_dist(Pos,x6,y6,x7,y7)
                Dist_7 = calc_dist(Pos,x7,y7,x8,y8)
                Dist_8 = calc_dist(Pos,x8,y8,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4,Dist_5,Dist_6,Dist_7,Dist_8])
                arg_dists = np.argmin(Dists,axis=0)
                return arg_dists

            x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
            d = 3
            dx = (x_max - x_min)/d
            dy = (y_max - y_min)/d
            x1,y1,x2,y2 = x_min,y_min+dy, x_min+dx, y_min
            x3,y3,x4,y4 = x_max-dx,y_min, x_max, y_min+dy
            x5,y5,x6,y6 = x_max,y_max-dy, x_max-dx,y_max
            x7,y7,x8,y8 = x_min+dx,y_max, x_min,y_max-dy
            p0 = (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
            bounds_x_m = (x_min,0)
            bounds_x_M = (0,x_max)

            bounds_y_m = (y_min,0)
            bounds_y_M = (0,y_max)

            Bounds = np.array( [bounds_x_m,bounds_y_m,bounds_x_m,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_M,bounds_x_M,bounds_y_M,bounds_x_m,bounds_y_M,bounds_x_m,bounds_y_M]).transpose()


            angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
            N_angles=64
            d_angle = 2*np.pi/N_angles
            mean_pos_angle=[]
            angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
            for angle in angles:
                selection = (angles >= angle) * (angles < (angle + d_angle))
                mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
            mean_pos_angle = np.array(mean_pos_angle)



            x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8 = sp.optimize.curve_fit(points,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=p0,bounds=Bounds)[0]

            Vecs_ext = np.array([[x3-x4,y3-y4],[x5-x4,y5-y4],[x5-x6,y5-y6]])
            Vecs_int = np.array([[x7-x8,y7-y8],[x8-x1,y8-y1],[x1-x2,y1-y2]])

            arg_dists = points_arg(Pos_slice[:,:2],x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)

            ext_points = Pos_slice[((arg_dists == 3) + (arg_dists == 4) + (arg_dists == 5)).astype("bool")][:,:2]
            dists_ext = points(ext_points,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
            thickness_ext = np.sort(dists_ext)
            thickness_ext = thickness_ext[:int((1-threshold_cut)*len(thickness_ext))][-1]

            int_points = Pos_slice[((arg_dists == 1) + (arg_dists == 7) + (arg_dists == 8)).astype("bool")][:,:2]
            dists_int = points(int_points,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
            thickness_int = np.sort(dists_int)
            thickness_int = thickness_int[:int((1-threshold_cut)*len(thickness_int))][-1]

            # T_ext.append(thickness_ext*2)
            # T_int.append(thickness_int*2)
            # W_ext.append(np.sum(np.linalg.norm(Vecs_ext,axis=1)))
            # W_int.append(np.sum(np.linalg.norm(Vecs_int,axis=1)))


            # print("t_ext",thickness_ext*2)
            # print("t_int",thickness_int*2)
            # print("w_ext",np.sum(np.linalg.norm(Vecs_ext,axis=1)))
            # print("w_int",np.sum(np.linalg.norm(Vecs_int,axis=1)))

        # plt.plot(Pos_slice[:,0],Pos_slice[:,1],"or")
        # plt.show()
        # c=d
            # print(np.array([x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8]))

        N_y = 20
        if True:

            y_int = np.linspace(y_min,y_max,N_y,endpoint=False)
            dy = y_int[1] - y_int[0]
            L_T_int = []
            for y in y_int:
                Pos_extr = Pos_slice[(Pos_slice[:,0] > 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                L_T_int.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
            T_int.append(np.median(L_T_int))



            y_ext = np.linspace(y_min,y_max,N_y,endpoint=False)
            dy = y_ext[1] - y_ext[0]
            L_T_ext = []
            for y in y_ext:
                Pos_extr = Pos_slice[(Pos_slice[:,0] < 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                L_T_ext.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
            T_ext.append(np.median(L_T_ext))


        ##FUNC IMG
        if False:
            # Pos_slice = Pos[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]
            # mean_slice = np.mean(Pos_slice,axis=0)
            # Pos_slice = (Pos_slice - mean_slice).dot(Rz(-angle_x))
            Pos_slice = Pos_transfo[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]

            Diff_slice_mean = center - np.mean(Pos_slice,axis=0)
            Pos_slice = Pos_slice - np.mean(Pos_slice,axis=0)

            angle_z = np.arctan2(Diff_slice_mean[1],Diff_slice_mean[0])

            Pos_slice = Pos_slice.dot(Rz(angle_z))

            mean_slice = np.mean(Pos_slice,axis=0)
            Num_S = len(Pos_slice)
            Pos_slice = Pos_slice - mean_slice


            def plane_surf(X,a,b):
                x,y = X.transpose()
                return a*x + b*y

            a,b = sp.optimize.curve_fit(plane_surf,Pos_slice[:,:2],Pos_slice[:,2])[0]

            Vec_1 = np.array([1,0,plane_surf(np.array([1,0]),a,b)])
            Vec_2 = np.array([0,1,plane_surf(np.array([0,1]),a,b)])

            if False:
                ar = pv.Arrow([0,0,0],Vec_1,scale=10)
                plotter.add_mesh(ar,name="ar1",color="red")
                ar = pv.Arrow([0,0,0],Vec_2,scale=10)
                plotter.add_mesh(ar,name="ar2",color="blue")
            if True:
                ar = pv.Arrow([0,0,0],[1,0,0],scale=10)
                plotter.add_mesh(ar,name="ar3",color="Black")
                ar = pv.Arrow([0,0,0],[0,1,0],scale=10)
                plotter.add_mesh(ar,name="ar4",color="Black")
                ar = pv.Arrow([0,0,0],[0,0,1],scale=10)
                plotter.add_mesh(ar,name="ar5",color="Black")

            angle_y = np.arctan2(Vec_1[0],Vec_1[2])
            Pos_slice = Pos_slice.dot(Ry(angle_y-np.pi/2))

            angle_x = np.arctan2(Vec_2[2],Vec_2[1])
            Pos_slice = Pos_slice.dot(Rx(angle_x))


            # angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
            # dist = np.linalg.norm(Pos_slice,axis=1)
            #
            # def coss(x,a,phi,b,w):
            #     return -a*np.cos(w*x+phi) + b
            # a, phi, b, w = (np.max(dist)-np.min(dist)) /2, 0, np.mean(dist), 2
            # a, phi, b, w = sp.optimize.curve_fit(coss,angles,dist,p0=(a,phi,b,w))[0]
            # # Pos_slice = Pos_slice.dot(Rz(-phi+np.pi/2))
            # print("PHI",phi)
            # x = np.linspace(0,2*np.pi,100)
            # plt.plot(x,coss(x,a,phi,b,w),"o-r")
            # print("PHI_av",phi)
            #
            # angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
            # dist = np.linalg.norm(Pos_slice,axis=1)
            # a, phi, b, w = (np.max(dist)-np.min(dist)) /2, 0, np.mean(dist), 2
            # a, phi, b, w = sp.optimize.curve_fit(coss,angles,dist,p0=(a,phi,b,w))[0]
            # print("PHI_ap",phi)
            # plt.plot(x,coss(x,a,phi,b,w),"o-b")
            # plt.show()

            def points(Pos,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):
                X,Y = Pos.transpose()


                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x5,y5)
                Dist_5 = calc_dist(Pos,x5,y5,x6,y6)
                Dist_6 = calc_dist(Pos,x6,y6,x7,y7)
                Dist_7 = calc_dist(Pos,x7,y7,x8,y8)
                Dist_8 = calc_dist(Pos,x8,y8,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4,Dist_5,Dist_6,Dist_7,Dist_8])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists

            def square_fit(Pos,x1,y1,x2,y2,x3,y3,x4,y4):

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists




            def square_fit_T(Pos,l,L,phi):
                Pos_corners = np.array([[-l/2, -L/2],[l/2,-L/2],[l/2,L/2],[-l/2,L/2]])
                Pos_corners_rota = Pos_corners.dot(Rota(phi))
                x1,y1,x2,y2,x3,y3,x4,y4 = Pos_corners_rota.ravel()

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists


            angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0])+np.pi
            N_angles=64
            d_angle = 2*np.pi/N_angles
            mean_pos_angle=[]
            angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
            for angle in angles:
                selection = (angles >= angle) * (angles < (angle + d_angle))
                mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
            mean_pos_angle = np.array(mean_pos_angle)

            x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
            l,L,phi = x_max-x_min,y_max-y_min,0
            Bounds = np.array([[0,l],[0,L],[0,np.pi*2]]).transpose()
            print(l,L,phi)
            l,L,phi = sp.optimize.curve_fit(square_fit_T,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=(l,L,phi),bounds=Bounds)[0]
            print(l,L,phi)

            Pos_slice = Pos_slice.dot(Rz(-phi))


            if False:
                x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
                d = 3
                dx = (x_max - x_min)/d
                dy = (y_max - y_min)/d
                x1,y1,x2,y2 = x_min,y_min+dy, x_min+dx, y_min
                x3,y3,x4,y4 = x_max-dx,y_min, x_max, y_min+dy
                x5,y5,x6,y6 = x_max,y_max-dy, x_max-dx,y_max
                x7,y7,x8,y8 = x_min+dx,y_max, x_min,y_max-dy
                p0 = (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
                bounds_x_m = (x_min,0)
                bounds_x_M = (0,x_max)

                bounds_y_m = (y_min,0)
                bounds_y_M = (0,y_max)

                Bounds = np.array( [bounds_x_m,bounds_y_m,bounds_x_m,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_M,bounds_x_M,bounds_y_M,bounds_x_m,bounds_y_M,bounds_x_m,bounds_y_M]).transpose()


                angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
                N_angles=64
                d_angle = 2*np.pi/N_angles
                mean_pos_angle=[]
                angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
                for angle in angles:
                    selection = (angles >= angle) * (angles < (angle + d_angle))
                    mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
                mean_pos_angle = np.array(mean_pos_angle)



                x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8 = sp.optimize.curve_fit(points,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=p0,bounds=Bounds)[0]


                # print(np.array([x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8]))

            if False:
                angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
                dist = np.linalg.norm(Pos_slice,axis=1)

                def coss(x,a,phi,b,w):
                    return a*np.cos(w*x+phi) + b
                a, phi, b, w = (np.max(dist)-np.min(dist)) /2, angles[np.argmax(dist)], np.mean(dist), 2
                print(a,phi,b,w)
                a,phi,b,w = sp.optimize.curve_fit(coss,angles,dist,p0=(a,phi,b,w))[0]
                # a,phi,b,w = sp.optimize.curve_fit(coss,angle,dist)[0]
                print(a, phi, b, w)
                if a < 0: a, phi = -a, phi + np.pi

                x=np.linspace(0,2*np.pi,100)
                plt.plot(x,coss(x,a,phi,b,w),"-b")
                plt.plot(angles,dist,"or")

                angle_max_1 = (-phi/w) % (2*np.pi)
                angle_max_2 = ((-phi+2*np.pi)/w)% (2*np.pi)

                angle_max_1, angle_max_2 = max(angle_max_1, angle_max_2), min(angle_max_1, angle_max_2)

                plt.vlines(angle_max_1,b-a,a+b,"g")
                plt.vlines(angle_max_2,b-a,a+b,"b")

                select_int = (angles<angle_max_1) * (angles>angle_max_2)
                int_slice = Pos_slice[select_int]
                ext_slice = Pos_slice[(1-select_int).astype("bool")]

                N_angles = 16
                angles_int = angles[select_int]
                min_angle_int, max_angle_int = np.min(angles_int),np.max(angles_int)


                d_angle = (max_angle_int - min_angle_int)/N_angles
                angles_select_int = np.linspace(min_angle_int,max_angle_int,N_angles,endpoint=False)

                thickness_int = []
                mean_pos_angle_int = []
                for angle in angles_select_int:
                    selection = (angles[select_int] >= angle) * (angles[select_int] < (angle + d_angle))

                    int_slice_angle = int_slice[selection]
                    mean_pos_angle_int.append(np.mean(int_slice_angle,axis=0))
                    print(np.sum(selection))
                    thickness_int.append([])

                plt.plot(angles_select_int,coss(angles_select_int,a,phi,b,w),"oy")
                plt.show()

            if False:
                x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])

                x1,y1,x2,y2,x3,y3,x4,y4 = x_min,y_min, x_max, y_min, x_max,y_max,x_min,y_max

                p0 = (x1,y1,x2,y2,x3,y3,x4,y4)
                bounds_x_m = (x_min,0)
                bounds_x_M = (0,x_max)

                bounds_y_m = (y_min,0)
                bounds_y_M = (0,y_max)

                Bounds = np.array( [bounds_x_m,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_M,bounds_x_m,bounds_y_M]).transpose()
                angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
                N_angles=64
                d_angle = 2*np.pi/N_angles

                mean_pos_angle=[]
                angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
                for angle in angles:
                    selection = (angles >= angle) * (angles < (angle + d_angle))
                    mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
                mean_pos_angle = np.array(mean_pos_angle)
                # x1,y1,x2,y2,x3,y3,x4,y4 = sp.optimize.curve_fit(square_fit,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=p0,bounds=Bounds)[0]

                N_y = 20

                # y_int = np.linspace(y_min,y_max,N_y,endpoint=False)
                y_int = np.linspace(y_min,y_max,N_y,endpoint=False)
                dy = y_int[1] - y_int[0]
                L_T_int = []
                for y in y_int:
                    Pos_extr = Pos_slice[(Pos_slice[:,0] > 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                    L_T_int.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
                T_int.append(np.median(L_T_int))


                # y_ext = np.linspace(y1,y4,N_y,endpoint=False)
                y_ext = np.linspace(y_min,y_max,N_y,endpoint=False)
                dy = y_ext[1] - y_ext[0]
                L_T_ext = []
                for y in y_ext:
                    Pos_extr = Pos_slice[(Pos_slice[:,0] < 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                    L_T_ext.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
                T_ext.append(np.median(L_T_ext))











            if False:

                # print(x1,y1,x2,y2,x3,y3,x4,y4)

                tube = pv.Tube(pointa = [x1,y1,0], pointb = [x2,y2,0])
                plotter.add_mesh(tube,name="ta")
                tube = pv.Tube(pointa = [x2,y2,0], pointb = [x3,y3,0])
                plotter.add_mesh(tube,name="tb")
                tube = pv.Tube(pointa = [x3,y3,0], pointb = [x4,y4,0])
                plotter.add_mesh(tube,name="tc")
                tube = pv.Tube(pointa = [x4,y4,0], pointb = [x5,y5,0])
                plotter.add_mesh(tube,name="td")
                tube = pv.Tube(pointa = [x5,y5,0], pointb = [x6,y6,0])
                plotter.add_mesh(tube,name="te")
                tube = pv.Tube(pointa = [x6,y6,0], pointb = [x7,y7,0])
                plotter.add_mesh(tube,name="tf")
                tube = pv.Tube(pointa = [x7,y7,0], pointb = [x8,y8,0])
                plotter.add_mesh(tube,name="tg")
                tube = pv.Tube(pointa = [x8,y8,0], pointb = [x1,y1,0])
                plotter.add_mesh(tube,name="th")


            if False:
                Pos_corners = np.array([[-l/2, -L/2],[l/2,-L/2],[l/2,L/2],[-l/2,L/2]])
                Pos_corners_rota = Pos_corners.dot(Rota(phi))
                x1,y1,x2,y2,x3,y3,x4,y4 = Pos_corners_rota.ravel()
                # print(x1,y1,x2,y2,x3,y3,x4,y4)

                tube = pv.Tube(pointa = [x1,y1,0], pointb = [x2,y2,0])
                plotter.add_mesh(tube,name="ta")
                tube = pv.Tube(pointa = [x2,y2,0], pointb = [x3,y3,0])
                plotter.add_mesh(tube,name="tb")
                tube = pv.Tube(pointa = [x3,y3,0], pointb = [x4,y4,0])
                plotter.add_mesh(tube,name="tc")
                tube = pv.Tube(pointa = [x4,y4,0], pointb = [x1,y1,0])
                plotter.add_mesh(tube,name="td")





            Types_slice = Types[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]
            Pos_slice = Pos[:]
            Types_slice = Types[:]

            # Pos_slice = Pos_transfo[:]
            # Types_slice = Types[:]

            # Pos_slice = np.append(Pos_slice,[[0]]*len(Pos_slice),axis=1)

            # x,y,z = transfo_inv(Pos_slice,diameter_avg,pitch,thickness)
            # print(diameter_avg,pitch,thickness)
            # print(np.shape(Pos_slice))
            # Pos_slice = np.array([x,y,z]).transpose()
            # print(np.shape(Pos_slice))

            sph = pv.Sphere(radius=0.2)
            if True:
                sph = pv.Sphere(radius=1.0)
                data = pv.PolyData(mean_pos_angle)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="blue",name="aa")
            if False:
                bis = np.append(bisector_points,[[0]]*len(bisector_points),axis=1)
                data = pv.PolyData(bis)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="green",name="bis")

            if False:
                for a,b in zip(perpendicular_dir_coef,perpendicular_affin):
                    x = np.linspace(-90,120,10)
                    y = a*x+b
                    z = np.array([0]*len(x))
                    X,Y,Z = np.meshgrid(x,y,z)
                    data =pv.StructuredGrid(x,y,z)
                    plotter.add_mesh(data,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="red")

            if False:
                sph = pv.Sphere(radius=0.6)
                data = pv.PolyData(int_slice)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="gray",name="si")

                sph = pv.Sphere(radius=0.4)
                data = pv.PolyData(ext_slice)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="red",name="o")

            else:


                # plotter=pv.Plotter()
                # plotter.show_axes()
                sph = pv.Sphere(radius=0.6)
                data = pv.PolyData(Pos_slice[Types_slice==1])
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="gray",name="si")

                sph = pv.Sphere(radius=0.4)
                data = pv.PolyData(Pos_slice[(Types_slice==2) + (Types_slice==3)])
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="red",name="o")


            if True:
                sph = pv.Sphere(radius=1.2)
                centers = pv.PolyData([[0.,0,0]])
                pc = centers.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="black",name="center")


            # plotter.show()

            # N_Si,N_O = np.shape(Bonds)
            # Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))
            #
            # Pos_Si = Pos_cutted[Types_cutted==1]
            # Pos_O = Pos_cutted[((Types_cutted==2) + (Types_cutted==3)).astype("bool")]
            #
            # Indices = Indices[Bonds.ravel()!=0]
            # tubes = pv.MultiBlock()
            # for i_si,i_o in Indices:
            #     t = pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2)
            #     tubes.append(t)
            #
            # mesh = tubes.combine()
            # plotter.add_mesh(mesh,opacity=0.1,pbr=True,roughness=.5,metallic=.2,color="blue",name="Bonds")

    if show:
        plotter = pv.Plotter()
        plotter.show_axes()
        plotter.add_slider_widget(slide, [0,N_z-1],value=N_z/2,title="Pos", fmt="%3.3e")
        plotter.show()

    print("W",np.median(W))
    # print("W_int",np.mean(W_int))
    # print("W_ext",np.mean(W_ext))
    print("T_int",np.median(T_int))
    print("T_ext",np.median(T_ext))
    print("T",np.median(T))
    print("pitch",pitch)
    print("thickness",thickness)
    print("D",2*D+np.median(T))
    print("diameter",diameter)
    return pitch, np.median(W), np.mediant(T), 2*D+np.median(T)

def evaluate_surface_reconstruct(Pos_transfo,Types,slice_thickness=5,threshold_cut=0.01,threshold_lr=2):
    rota = 1
    pitch, width, thickness, D = evaluate_dimenions(Pos_transfo,Types,slice_thickness,threshold_cut,threshold_lr)
    Pos_model_no_rota,Types_model,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(0,D,pitch,width,thickness,int_thick,do_clean=False,circling=True,do_periodic=False,file_duplicate="grid.data")
    Pos_model,Types_model,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,int_thick,do_clean=False,circling=True,do_periodic=False,file_duplicate="grid.data")

    angles = np.arctan2(Pos_model_no_rota[:,1],Pos_model_no_rota[:,0])
    dz = 1
    min_z, max_z = np.min(Pos_model_no_rota[:,2]),np.max(Pos_model_no_rota[:,2])
    Nz = (max_z - min_z)/dz
    Pos = []
    for nz in range(Nz):
        Slice = (Pos_mode_no_rota[:,2] >= (nz * dz)) * (Pos_mode_no_rota[:,2] <= (nz * dz + dz))
        Pos.append(Pos_model_no_rota[Slice])

    Pos = np.array(Pos)

    plotter = pv.Plotter()
    plotter.add_axes()



    Si_c = [240,200,160]
    O_c = [255,13,13]
    O_c2 = [0,255,0]
    H_c = [255,255,255]
    H_c2 = [0,0,255]

    sp = pv.Sphere(radius=0.4)
    Pos_transfo = np.array(Pos)
    Pos_t

    data = pv.PolyData(Pos_transfo[Types==1])
    pc = data.glyph(scale=False,geom=sp,orient=False)
    plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=Si_c,scalars=np.arange(len(Pos_transfo)*len(sp.points)))


    Lims, Atom_types, Atom_pos = read_data("grid.data",do_scale=False,atom_style="atom")


    for z in range(len(Pos_transfo)-Nz-1):

        # P = [Pos_transfo[z],Pos_transfo[z+1],Pos_transfo[z+Nz],Pos_transfo[z+Nz+1]]
        P = [Pos_transfo[z],Pos_transfo[z+Nz]]
        x_coord,y_coord,z_coord = np.array(P).transpose()
        grid = pv.StructuredGrid(x_coord,y_coord,z_coord)
        plotter.add_mesh(grid,opacity=0.9,pbr=True,roughness=.5,metallic=.2)







    # Pos_model =







if __name__=="__main__":
    #
    # file = "quartz_dupl.data"
    # list_BOX,list_ATOMS = read_data(file,do_scale=False)


    # file = "demo/dump_last_oh.lammpstrj"
    # file = "demo/dummp_trimmed.lammpstrj"
    # file = "demo/dummps_snad_last.lammpstrj"
    # file = "demo/dummp_700_last.lammpstrj"
    # file = "dummp_870_last.lammpstrj"
    # file = "demo/dummp_twisted_long.lammpstrj"
    # file = "demo/dummp_144_last.lammpstrj"
    # file = "demo/dummps_snad_last.lammpstrj"
    # file = "demo/dummps_round_2_last.lammpstrj"
    # file = "demo/dummps_round_3_last.lammpstrj"
    file = "dummp_144_last.lammpstrj" #Large 60ps
    # file = "dummp_145_last.lammpstrj" #Large 200ps
    # file = "dummp_156_last.lammpstrj" #New 60ps
    # file = "dummp_157_last.lammpstrj" #New 200ps
    # file = "dummp_258_last.lammpstrj" #no transf
    # file = "demo/dummp_5K.lammpstrj"
    # file = "demo/dummp_3K.lammpstrj"
    # file = "demo/dummps_round_new.lammpstrj"
    # file = "demo/dummps_long_last.lammpstrj"
    # file = "demo/dummp_trimmed_long_0K.lammpstrj"
    # file = "demo/dummp_trimmed_0K_last.lammpstrj"
    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)


    list_TSTEP=[0]
    list_Pos = list_ATOMS[:,:,2:]
    list_Types = list_ATOMS[:,:,1]
    Pos = list_ATOMS[-1][:,2:]
    Types = list_ATOMS[-1][:,1]



    # plot_syst(Pos,Types,Lims=list_BOX[-1])

    # analyze_plot_syst(Pos,Types,periodic=True,Lims=list_BOX[-1])
    # analyze_mult(list_TSTEP,list_Pos,list_Types,periodic=False,Lims=list_BOX[-1],save=False,density=False)
    # curvature_analysis(Pos)
    # save_defects("defects.xyz",Pos,Types,periodic=True,Lims=list_BOX[-1])
    # analyze_defects(Pos,Types,periodic=True,Lims=list_BOX[-1])
    # analyze_density(Pos)
    # evaluate_dimenions(Pos,Types,slice_thickness=8,show=False,threshold_cut=0.001)
    evaluate_dimenions(Pos,Types,slice_thickness=8,show=False,threshold_cut=0.001)
    # evaluate_surface_reconstruct(Pos,Types,slice_thickness=8,threshold_cut=0.001)