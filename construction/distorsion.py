import numpy as np
from script_analysis import *
from read_write import *




def duplicate(Nx_list,Ny_list,Nz,Lims,Atom_type,Atom_pos,Bonds_OH=[],Angles_OH=[]):
    """
    duplicate(Nx_list,Ny_list,Nz,Lims,Atom_type,Atom_pos,clean=False,Bonds_OH=[],Angles_OH=[])

    Duplicates a system multiple times.
    Nx_list and Ny_list can be a list or an int.
    Int are used to duplicate a certain amount of time
    Lists are used to create hollow structures. Typically [0,3,17,20] will duplicate for the positions 0,1,2, 17,18,19

    The data OH are used to also duplicate the bonds and angles as the H are added using harmonic potential

    Parameters
    ----------
        Nx_list : list or int,
            See description
        Ny_list : list or int,
            See description
        Nz : int,
            The amount of times the initial structure is duplicated
        Lims : list,
            The limit coordinates of the system
        Atom_type : list,
            The type of the atoms
        Atom_pos : list,
            the position of the atoms
        Bonds_OH : list, optional
            The indices of the bonds OH
        Anlges_OH : list, optional
            The indices of the angles SiOH

    Returns
    -------
        Pos : The updated positions
        Types : The updated Types
        Lims_tot : The updated limits
        Bonds_OH_tot : The updated bonds
        Angles_OH_tot : The updated angles

    """
    lx,Lx = Lims[0]
    ly,Ly = Lims[1]
    lz,Lz = Lims[2]

    Nx_0,Ny_0 = 0,0

    bonds = False
    if len(Bonds_OH) > 0:
        bonds = True

    Pos = []
    Types = []
    Pos_O = Atom_pos[Atom_type==2]
    num_O = np.arange(len(Atom_pos))[Atom_type==2]
    z_prop_add = 0

    #Check if int isntead of list
    flag_x = 0
    flag_y = 0
    if type(Nx_list) is int:
        Nx = Nx_list
        flag_x = 1

    elif len(Nx_list) == 2:
        Nx_0 = Nx_list[0]
        Nx = Nx_list[1]
        flag_x = 1

    else: Nx = Nx_list[-1]

    if type(Ny_list) is int:
        Ny = Ny_list
        flag_y = 1

    elif len(Ny_list) == 2:
        Ny_0 = Ny_list[0]
        Ny = Ny_list[1]
        flag_y = 1

    else: Ny = Ny_list[-1]



    for x in range(Nx_0,Nx):
        for y in range(Ny_0,Ny):
            #Check if inside or surface
            if flag_x or flag_y or (x >= Nx_list[0] and x< Nx_list[1]) or (x >= Nx_list[2] and x <= Nx_list[3]) or (y >= Ny_list[0] and y < Ny_list[1]) or (y >= Ny_list[2] and y <= Ny_list[3]):
                for z in range(Nz):
                    for at,ty,num in zip(Atom_pos,Atom_type,range(len(Atom_pos))):
                        Pos.append(at + np.array([x*Lx+lx,y*Ly+ly,z*Lz+lz]))
                        Types.append(ty)




    Pos = np.array(Pos).reshape(len(Pos),3)
    min_x,max_x = np.min(Pos[:,0]),np.max(Pos[:,0])
    min_y,max_y = np.min(Pos[:,1]),np.max(Pos[:,1])
    dx, dy  = (max_x), (max_y)
    max_d = max(dx,dy)
    maxx =  max_d * (2**(1/2))
    maxy =  max_d * (2**(1/2))

    Lims_tot = np.array([[-maxx,maxx],[-maxy,maxy],[lz,Lz]])

    num_at = len(Atom_pos)
    Bonds_OH_tot = []
    Angles_OH_tot = []
    if bonds:
        for j in range(Nz):
            for bond in Bonds_OH:
                Bonds_OH_tot.append([bond[0]+num_at*j,bond[1]+num_at*j])
            for angle in Angles_OH:
                Angles_OH_tot.append([angle[0]+num_at*j,angle[1]+num_at*j,angle[2]+num_at*j])
        return Pos,np.array(Types),Lims_tot, Bonds_OH_tot, Angles_OH_tot

    else:
        return Pos,np.array(Types),Lims_tot, [], []




def clean_structure(Pos,Types,Lims,N,periodic=True):
    """
    clean_structure(Pos,Types,Lims,N,periodic=True)

    Cleans the structure by adding H2O in the system.
    This is used inside create_syst on the slab.
    This clean will add bonds and angles to the system for lammps to use

    Please note that the added OH will be done so on the x axis and suppose an orthogonal system

    Parameters
    ----------
        Pos : list,
            Position of the atoms
        Types : list,
            Type of the atoms
        Lims : list,
            the limit of the non duplicated system
        N : list,
            the list of the three number of duplication
        periodic : bool, optional
            Computes the bonds as a periodic system. By default True

    Returns
    -------
        New_Pos : The new positions
        New_Types : The new types
        Bonds_OH_corrected : The Bonds of OH
        Angles_OH : The angles SiOH
    """


    Center = [(Lims[0][1]*N[0][0] - Lims[0][0])/2,(Lims[1][1]*N[1][0] - Lims[1][0])/2, (Lims[2][1]*N[2][0] - Lims[2][0])/2]
    Dist_list, Si_count_O, O_count_Si = compute_hist_neighbors(Pos,Types,Lims=Lims*N,periodic=periodic)[:3]

    Pos_Si = Pos[Types==1]
    Pos_O = Pos[Types==2]

    #The Si that have only 2 bonds are located on the edges
    Lack_Si = Pos_Si[Si_count_O == 2]

    Atoms_add_pos = []
    Atoms_add_types = []
    Bonds_OH = []
    num_at = len(Pos)

    for Si in Lack_Si:
        #Add OH next to a Lack_Si on the x axis at 1.6 A and 2.6 A
        Atom_O_add = np.sign(Si[0]-Center[0]) * np.array([1.6,0,0]) + Si
        Atom_H_add = np.sign(Si[0]-Center[0]) * np.array([2.6,0,0]) + Si

        Atoms_add_pos.append(Atom_O_add)
        Atoms_add_pos.append(Atom_H_add)
        Atoms_add_types.append(3)
        Atoms_add_types.append(4)
        Bonds_OH.append([num_at,num_at+1])


        #Get the closest O symmetric with respect to the plane yz and xz
        Pos_O_trunc = Pos_O[(Pos_O[:,2] > (Si[2]-2)) * (Pos_O[:,2] < (Si[2]+2))]
        #The factor should be 2, but 2.2 is used in order to make sure to not put the OH inside
        Si_symm = np.array([2*Center[0] -Si[0],Si[1], Si[2]])
        #This suppose a orthogonal system
        Dist_O = np.sum((Pos_O_trunc - Si_symm.reshape((1,3)))**2,axis=1)
        Pos_O_add_H = Pos_O_trunc[np.argmin(Dist_O)]
        #Add the H atom to the symmetric O
        Atom_H_add = np.sign(Pos_O_add_H-Center[0]) * np.array([1.0,0,0.0]) + Pos_O_add_H
        Atoms_add_pos.append(Atom_H_add)
        Atoms_add_types.append(4)

        #Transform the O to the proper type
        Index_O_trunc = (Types==2) * (Pos[:,2] > (Si[2]-2)) * (Pos[:,2] < (Si[2]+2))
        Index_O_bond = np.argmax((np.cumsum(Index_O_trunc)-1)==np.argmin(Dist_O))
        Types[Index_O_bond] = 3

        Bonds_OH.append([Index_O_bond,num_at+2])

        num_at += 3


    #Appends everyting together
    New_Pos, New_Types = np.append(Pos,Atoms_add_pos,axis=0), np.append(Types,Atoms_add_types,axis=0)
    Middle = (New_Pos[:,2] >= (Lims[2,1]+Lims[2,0])) * (New_Pos[:,2] < (Lims[2,1]*2+Lims[2,0]))
    New_Pos = New_Pos[Middle]
    New_Pos = New_Pos - np.array([0,0,np.min(New_Pos[:,2])])
    New_Types = New_Types[Middle]

    Bonds_OH_corrected = []
    New_index = (np.cumsum(Middle)-1) * Middle


    for bond in Bonds_OH:
        if New_index[bond[0]] != 0 and New_index[bond[1]] != 0:
            Bonds_OH_corrected.append([New_index[bond[0]]+1,New_index[bond[1]]+1])


    #Bonds : O then H
    Angles_OH = []
    Pos_Si = New_Pos[New_Types==1]
    #Compute the angles using the closest Si to the OH
    for bond in Bonds_OH_corrected:
        O = New_Pos[bond[0]-1]
        Pos_Si_trunc =  Pos_Si[(Pos_Si[:,2] > (O[2]-2)) * (Pos_Si[:,2] < (O[2]+2))]
        Dist_Si = np.sum((Pos_Si_trunc - O.reshape((1,3)))**2,axis=1)

        Index_Si_trunc = (New_Types==1) * (New_Pos[:,2] > (O[2]-2)) * (New_Pos[:,2] < (O[2]+2))
        Index_Si_bond = np.argmax((np.cumsum(Index_Si_trunc)-1) == (np.argmin(Dist_Si)))
        Angles_OH.append([Index_Si_bond+1,bond[0],bond[1]])



    return New_Pos, New_Types, Bonds_OH_corrected, Angles_OH





def transfo(Pos,slide_z=0,D=0,rota=0,enlarge=10,enlarge_z=0.700758,do_periodic=True,circling=True,do_rota_transf=False,params_helix=[]):
    """
    transfo(Pos,slide_z=0,D=0,rota=0,enlarge=10,enlarge_z=0.700758,do_periodic=True,circling=True,do_rota_transf=False)

    Do the coordinate transformations of the initial cuboid.
    The formula used for the circling is the elliptical grid mapping


    Parameters
    ----------
        Pos : List,
            The transformed position of the atoms
        slide_z : float, optional
            The amount of shifting in the z coordinate. If 0, will automatically put the lowest atom to 0
            This is used to slide the transformed cast inside the helix
            By default : 0
        rota : float, optional
            The amount of turn to do. By default 0
        enlarge : float, optional
            The space in Angstrom that is added in x and y. This is used to create empty space around the helix. By default 10
        enlarge_z : float, optional
            The space in A that is added in z. The value represent the distance between the highest atom and the border.
            By default 0.700758
        do_periodic : bool, optional
            do the creation of a periodic system after the transformation
        circling : bool, optional
            Circle the basis of the cuboid. This should be used to create helices as is gives much better curvatures. By default True
        do_rota_transf : bool, optional,
            Use the rota transformation instead of the helix one. This should not be used as it gives much worse structure. By default False

    Returns
    -------
        Pos_transfo : The transformed positions
        Lims : the limits of the new system
        slide_z : the amount of shift in the z coordinate the system has taken through the transformation
    """

    mean = np.mean(Pos,axis=0)
    mean[2] = np.min(Pos[:,2])
    Pos = Pos - mean

    Lz = np.max(Pos[:,2]) / 2 / np.pi
    x,y,z = Pos.transpose()
    z = z/Lz

    Lx = np.max(x)
    lx = np.min(x)
    Ly = np.max(y)
    ly = np.min(y)

    if len(params_helix):
        LX = params_helix[1]
        LY = params_helix[2]
    else:
        LX = Lx-lx
        LY = Ly-ly


    if circling:
        x = (x-lx - LX/2) / LX * 2
        y = (y-ly - LY/2) / LY * 2

        # #FG Squircular Mapping
        # x_coord = x * (x**2 + y**2 - x**2*y**2)**(1/2) / (x**2+y**2)**(1/2)
        # y_coord = y * (x**2 + y**2 - x**2*y**2)**(1/2) / (x**2+y**2)**(1/2)

        # #Simple Strecthing
        # x_coord = np.sign(x)*(x**2 / (x**2+y**2)**(1/2)) * (x**2>=y**2) + np.sign(y)*(x*y)/(x**2+y**2)**(1/2) * (x**2 < y**2)
        # y_coord = np.sign(x)*(x*y / (x**2+y**2)**(1/2)) * (x**2>=y**2) + np.sign(y)*(y*y)/(x**2+y**2)**(1/2) * (x**2 < y**2)

        # #Elliptical Mapping
        x_coord = x * (1-1/2*y**2)**(1/2)
        y_coord = y * (1-1/2*x**2)**(1/2)

        # #Schwarz Christoffel Mapping
        #This one has not been tested too extensively as it did not provide convincing results
        #I am not sure it even works
        # import scipy.special as sps
        # import mpmath
        # # print(sps.ellipj(x+1j*y,(1/2)**(1/2)))
        # x_coord = []
        # y_coord = []
        # for xa,ya in zip(x,y):
        #     c = (1-1j)/(2)**(1/2) * mpmath.ellipfun("cn",1.854 * (1+1j)/2 * (xa+1j*ya)-1.854,(1/2))
        #     x_coord.append(float(c.real))
        #     y_coord.append(float(c.imag))
        #
        #
        # x_coord = np.array(x_coord)
        # y_coord = np.array(y_coord)


        x = x_coord * Lx
        y = y_coord * Ly




    if do_rota_transf:
        #The worse version of the transformation.
        #This is used for illustration purposes to show the progress of the method
        Lz = 2*np.pi*Lz
        z_coord = (Lz**2-rota**2*D**2)**(1/2) *(z)/2/np.pi

        x_coord = (x) * np.cos(rota*(z)) - (y+D) * np.sin(rota*(z))
        y_coord = (x) * np.sin(rota*(z)) + (y+D) * np.cos(rota*(z))-D

    else:
        #The helix transformation
        #The equations were determined using a 2d roation matrix coupled with a rotation of the basis
        #Another way to see the equations is to compute the tangent and two orthogonal normals to the direction of the 1d helix
        #And to transform the cuboid along these directions : one normal corresponds to y, the other to x and the tangent to x

        R = D * rota
        Norm = (Lz**2 + R**2)**(1/2)

        z_coord = Lz * z + R * x / Norm
        # y_coord = R * np.cos(z*rota) - np.cos(z*rota) * y + Lz * np.sin(z*rota) / Norm * x
        y_coord = R * np.cos(z*rota) - np.cos(z*rota) * y + Lz * np.sin(z*rota) / Norm * x
        x_coord = R * np.sin(z*rota) - np.sin(z*rota) * y - Lz * np.cos(z*rota) / Norm * x
        # x_coord = R * np.sin(z*rota) - np.sin(z*rota) * y - Lz * np.cos(z*rota) / Norm * x





    #slide automatic
    if slide_z == 0:
        slide_z = np.min(z_coord)
    z_coord = z_coord - slide_z

    if do_periodic:
        #Cut the system in half, and put the top part on the bottom of the other one
        z_coord = (z_coord <= Lz*2*np.pi) * z_coord +  (z_coord > Lz*2*np.pi) * (z_coord - Lz * 2 * np.pi)

    #Slide to the initial mean
    Pos_transfo = np.array([x_coord + mean[0],y_coord + mean[1],z_coord]).transpose()

    Lims = np.array([[np.min(Pos_transfo[:,0]-enlarge),np.max(Pos_transfo[:,0])+enlarge],[np.min(Pos_transfo[:,1]-enlarge),np.max(Pos_transfo[:,1])+enlarge],[0,np.max(Pos_transfo[:,2])+enlarge_z]])

    return Pos_transfo, Lims, slide_z





def create_syst(rota,D_exp,pitch,width,thickness,int_thick,do_clean=True,do_periodic=True,circling=True,do_rota_transf=False,file_duplicate="beta_quartz.data",do_angles=False):
    """
    create_syst(rota,D,pitch,width,thickness,int_thick,do_clean=True,do_periodic=True,circling=True,do_rota_transf=False,file_duplicate="beta_quartz.data",do_angles=False)

    This functions creates the whole system using the dimensions stated.
    It creates two files : quartz_dupl.data and quartz_int.data which contains the data of the surface and the data of the cast respectively.

    Parameters
    ----------
        rota : float
            the amount of turns made by the helix
        D_exp : float
            the diameter of the circle containing the helix.
        pitch : float
            the pitch, or the period of the helix
        width : float
            the width of the helix
        thickness : float
            the thickness of the helix
        do_clean : bool, optional
            To clean or not the structure by adding H2O. By default True
        do_periodic : bool, optional
            do the creation of a periodic system after the transformation
        circling : bool, optional
            Circle the basis of the cuboid. This should be used to create helices as is gives much better curvatures. By default True
        do_rota_transf : bool, optional,
            Use the rota transformation instead of the helix one. This should not be used as it gives much worse structure. By default False
        file_duplicate : float, optional,
            the file containing the lattice that will be duplicated
        do_angles : bool, optional
            write the angles SiOH

    Returns
    -------
        Pos_transfo : list
            The position of the atoms of the surface
        Types : list
            The type of the atoms of the surface
        Lims_tot : list
            The lims of the surface
        Angles_OH : list
            The angles SiOH
        Pos_transfo_int : list
            The position of the atoms of the interior
        Types_int : list
            The type of the atoms of the interior
        Lims_tot_int : list
            The lims of the interior

        Two files are also created : quartz_dupl.data and quartz_int.data
    """

    Lims, Atom_types, Atom_pos = read_data(file_duplicate,do_scale=False,atom_style="atom")

    #Get the number of duplication needed to get the proper dimensions
    lx = Lims[0][1] - Lims[0][0]
    ly = Lims[1][1] - Lims[1][0]
    lz = Lims[2][1] - Lims[2][0]
    # lx = np.max(Atom_pos[:,0]) - np.min(Atom_pos[:,0])
    # ly = np.max(Atom_pos[:,1]) - np.min(Atom_pos[:,1])
    # lz = np.max(Atom_pos[:,2]) - np.min(Atom_pos[:,2])

    Nx = int(width // lx +1)

    Ny = int(thickness // ly +1)
    Nz = int(pitch // lz +1)

    Nx_int = int(int_thick // lx +1)
    Ny_int = int(int_thick // ly +1)

    Nx_list = [0,Nx_int,Nx-Nx_int,Nx]
    Ny_list = [0,Ny_int,Ny-Ny_int,Ny]

    N_list = np.array([[Nx],[Ny],[Nz]])

    if int_thick == 0:
        Nx_list = Nx
        Ny_list = Ny

    print(Nx_list,Ny_list,Nz)

    #due to the transformation, the D is not the same as the one experimentally
    #It is however possible to find the D corresponding to the transformation
    if circling:
        #The maximum distance is taken on the ellipse
        W = width
        P = pitch/2/np.pi
        T = thickness


        D_est = (-T+D_exp) /2
        N = P / (P*P + D_est*D_est)**(1/2)
        b = 2*T*D_est / (W*W*N*N - T*T)
        # print(b)

        if b > 1 :
            #The extremum point is the point in the external layer
            D_transfo = D_est

        else:
            #The extremum is in-between

            d0 = (W*W/4 - D_exp**2/4) * P**4 * (W*W-T*T)
            d2 = (P*P*D_exp**2/4 * (T*T-W*W) + P*P*T*T * (D_exp**2/4-W*W/4)) + P**4*W*W
            d4 = (D_exp**2/4 *T*T + W*W*P*P)

            delta = (d2**2 - 4*d4*d0)
            # print(delta)

            if delta > d2**2:
                D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)
            else:
                print("Two possiblities for the D, the higher one has been taken")
                D_transfo = ((-d2 - delta**(1/2))/2/d4)**(1/2)
                D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)

    else:
        #This formula is not exact as it does not take into account the norm
        D_transfo = ((D**2 - width**2)**(1/2) - thickness)/2
    if D_exp < width:
        D_transfo = 0

    print("Old D", D_transfo,D_exp)
    # D_transfo = 22
    # N = P / (P*P + D_transfo*D_transfo)**(1/2)
    # b = 2*T*D_transfo / (W*W*N*N - T*T)
    # print(b)
    # D_transfo = 13.31484968079192
    # D_transfo = 40

    #
    # print(2*(width**2/4 + D_transfo**2 * (1+(thickness**2/(width**2-thickness**2))) ) ** (1/2))


    ##Surface
    if do_clean:
        #create one slab that is corrected then duplicated Nz times
        #The slab contains 3 duplicates as to only take the inside when doing the cleaning
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx_list,Ny_list,3,Lims,Atom_types,Atom_pos)
        Pos, Types, Bonds_OH, Angles_OH = clean_structure(Pos,Types,Lims,N_list,periodic=True)
        Pos, Types, Lims_tot, Bonds_OH, Angles_OH = duplicate(1,1,Nz,Lims,Types,Pos,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)
        if not do_angles:
            Angles_OH = []
    else:
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx_list,Ny_list,Nz,Lims,Atom_types,Atom_pos)
        Bonds_OH, Angles_OH = [], []

    Pos_transfo,Lims_tot,slide_z = transfo(Pos,D=D_transfo,rota=rota,do_periodic=do_periodic,circling=circling,do_rota_transf=do_rota_transf,params_helix=[pitch,width,thickness])
    write_data("quartz_dupl.data",Pos_transfo,Types,Lims_tot,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)

    ##Inside
    if not type(Nx_list) is int:
        Nx_int = Nx_list[1:3]
        Ny_int = Ny_list[1:3]
        Pos_int, Types_int, Lims_tot_int, _a, _b = duplicate(Nx_int,Ny_int,Nz,Lims,Atom_types,Atom_pos)
        Pos_transfo_int ,Lims_tot_int,slide_z = transfo(Pos_int,D=D_transfo,rota=rota,slide_z=slide_z,do_periodic=do_periodic,circling=circling)
        write_data("quartz_int.data",Pos_transfo_int,Types_int,Lims_tot_int,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)
    else:
        Pos_transfo_int = None
        Types_int = None
        Lims_tot_int = None

    return Pos_transfo, Types, Lims_tot, Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int




if __name__ == "__main__":
    rota = 1.0
    # rota = 0.0

    #Real system
    # D = 271
    # pitch = 453
    # width = 226
    # thickness = 112
    # int_thick = 35

    D = 244
    # D =
    pitch = 453
    # pitch = 200
    width = 226
    thickness = 112
    int_thick = 35

    # a = 400/453
    #
    # D = 215.45
    # pitch = 400
    # width = 199.56
    # thickness = 98.89
    # int_thick = 15

    D = D * 1/3
    pitch = pitch * 1/3
    width = width* 1/3
    thickness = thickness * 1/3
    int_thick = 35/3

    #System
    # D = 180
    # pitch = 300
    # width = 150
    # thickness = 75
    # int_thick = 23

    ##Faire rapport pour 100 checker courbure


    #Smaller system
    # D = 100
    # pitch = 150
    # width = 40
    # thickness = 30
    # int_thick = 10

    #Miniature system used for tests
    # D = 0
    # pitch = 15
    # width = 10
    # thickness = 10
    # int_thick = 0
    # D = 150
    # pitch = 200
    # width = 60
    # thickness = 30
    # int_thick=0


    Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,int_thick,do_clean=False,circling=True,do_rota_transf=False)
    # Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,int_thick,circling=True)

    print("Number of Si : ",np.sum(Types==1))
    # plt.plot(np.sort(Pos_transfo[:,0]),"o-r")
    # plt.show()
    print(D)
    print("Diameter of the system : ",(np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0])))
    print("Diameter of the system : ",(np.max(Pos_transfo[:,1])-np.min(Pos_transfo[:,1])))

    #Do the analysis
    if 1:
        import matplotlib.pyplot as plt
        analyze_mult([0],[Pos_transfo],[Types],periodic=True,Lims=Lims_tot)
        # import pyvista as pv
        # analyze_plot_syst(Pos_transfo,Types,periodic=True,Lims=Lims_tot)

    #Plot the whole system
    #Not recommended for huge systems
    if 0:
        import pyvista as pv

        # Bonds = compute_bonds(Pos_transfo,Types)[0]

        plotter = pv.Plotter()
        plotter.add_axes()

        # purple = np.array([96,25,255])/255
        # dark_purple = np.array([56,20,180])/255

        Si_c = [240,200,160]
        O_c = [255,13,13]
        O_c2 = [0,255,0]
        H_c = [255,255,255]
        H_c2 = [0,0,255]

        sp = pv.Sphere(radius=0.4)


        data = pv.PolyData(Pos_transfo[Types==1])
        pc = data.glyph(scale=False,geom=sp,orient=False)
        plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=Si_c)

        data = pv.PolyData(Pos_transfo[Types==2])
        pc = data.glyph(scale=False,geom=sp,orient=False)
        plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=O_c)

        #If there is
        if np.sum(Types==3):
            data = pv.PolyData(Pos_transfo[Types==3])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=O_c2)

            data = pv.PolyData(Pos_transfo[Types==4])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=H_c2)

        if type(Types_int) is type(np.array([])):
            data = pv.PolyData(Pos_transfo_int[Types_int==1])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="black")

            data = pv.PolyData(Pos_transfo_int[Types_int==2])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="green")

        #Plot the bonds / angles
        if 0:
            # N_Si,N_O = np.shape(Bonds)
            # Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))
            # Pos_Si = Pos_transfo[Types==1]
            # Pos_O = Pos_transfo[(Types==2)+(Types==3)]
            #
            # Indices = Indices[Bonds.ravel()!=0]
            #
            # tubes = [pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2) for i_si,i_o in Indices]
            # mesh = tubes[0].merge(tubes[1:])
            #
            # plotter.add_mesh(mesh,opacity=0.3,pbr=True,roughness=.5,metallic=.2,color="white")
            #
            tubes = []
            for k in Angles_OH:
                tubes.append(pv.Tube(Pos_transfo[k[0]-1],Pos_transfo[k[1]-1],n_sides=5,radius=0.2))
                tubes.append(pv.Tube(Pos_transfo[k[1]-1],Pos_transfo[k[2]-1],n_sides=5,radius=0.2))
            mesh = tubes[0].merge(tubes[1:])

            plotter.add_mesh(mesh,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="blue")

        plotter.show()