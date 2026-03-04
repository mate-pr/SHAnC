import numpy as np
# import matplotlib.pyplot as plt


def read_dump(dump_file,unscale=False):
    """
    read dump file
    Suppose ATOM dump (i think)
    """
    flag_step = 0
    flag_num_at = 0
    flag_box_bound = 0
    flag_atoms = 0
    list_TSTEP = []
    list_NUM_AT = []
    list_BOX = []
    list_ATOMS = []
    for line in open(dump_file,"r"):
        if flag_step:
            list_TSTEP.append(int(line))
            flag_step = 0

        elif flag_num_at:
            list_NUM_AT.append(int(line))
            flag_num_at = 0

        elif flag_box_bound:
            lsplit = line.split()
            BOX.append([float(lsplit[0]),float(lsplit[1])])
            flag_box_bound -= 1
            if not flag_box_bound:
                list_BOX.append(BOX)

        elif flag_atoms:
            if "ITEM: TIMESTEP" in line:
                flag_step = 1
                flag_atoms = 0
                list_ATOMS.append(list_at_t)
            else:
                lsplit = line.split()
                if len(lsplit) != 5:
                    raise TypeError("This function suppose that the dump has used the atom type")
                if unscale:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    list_at_t.append([int(lsplit[0]),int(lsplit[1]),float(lsplit[2])*Lx+BOX[0][0],float(lsplit[3])*Ly+BOX[1][0],float(lsplit[4])*Lz+BOX[2][0]])
                else: list_at_t.append([int(lsplit[0]),int(lsplit[1]),float(lsplit[2]),float(lsplit[3]),float(lsplit[4])])

        elif "ITEM: TIMESTEP" in line:
            flag_step = 1
        elif "ITEM: NUMBER OF ATOMS" in line:
            flag_num_at = 1
        elif "ITEM: BOX BOUNDS" in line:
            flag_box_bound = 3
            BOX = []
        elif "ITEM: ATOMS" in line:
            flag_atoms = 1
            list_at_t = []
    list_ATOMS.append(list_at_t)
    try:
        list_ATOMS = np.array(list_ATOMS)
    except:
        print("Atoms lost or removed during the dynamic. Will only extract the last timestep")
        list_TSTEP = [list_TSTEP[-1]]
        list_NUM_AT = [list_NUM_AT[-1]]
        list_BOX = [list_BOX[-1]]
        list_ATOMS = np.array([list_ATOMS[-1]])
    if unscale:
        C_min = np.mean(list_ATOMS[:,:,2:],axis=1) + np.array([Lx/2,Ly/2,Lz/2])
        C_min[:,2] = 0

        # print(np.shape(C_min))
        C_min = C_min.reshape((len(C_min),1,3))
        list_ATOMS[:,:,2:] = list_ATOMS[:,:,2:] - C_min
        list_ATOMS[:,:,2] = list_ATOMS[:,:,2] % Lx
        list_ATOMS[:,:,3] = list_ATOMS[:,:,3] % Ly
        list_ATOMS[:,:,4] = (list_ATOMS[:,:,4]-BOX[2][0]) % Lz + BOX[2][0]
        # Pos_Z = list_ATOMS[:,:,4]
        # list_ATOMS[:,:,4] = (Pos_Z-Lz) * (Pos_Z > Lz) + (Pos_Z-Lz) * (Pos_Z > Lz)

    return list_TSTEP, list_NUM_AT, list_BOX, np.array(list_ATOMS)


def write_dump(file_name,list_TSTEP,list_NUM_AT,list_BOX,list_ATOMS):
    """write dump file"""
    with open(file_name,"w") as file:
        for tstep, num_at, box, atoms in zip(list_TSTEP,list_NUM_AT,list_BOX,list_ATOMS):

            Lx = box[0][1] - box[0][0]
            Ly = box[1][1] - box[1][0]
            Lz = box[2][1] - box[2][0]

            file.write("ITEM: TIMESTEP")
            file.write("\n")
            file.write(str(tstep))
            file.write("\n")
            file.write("ITEM: NUMBER OF ATOMS")
            file.write("\n")
            file.write(str(num_at))
            file.write("\n")
            file.write("ITEM: BOX BOUNDS pp pp pp")
            file.write("\n")
            for box_c in box:
                file.write("{:3.6f} {:3.6f}".format(box_c[0],box_c[1]))
                file.write("\n")
            file.write("ITEM: ATOMS id type xs ys zs")
            file.write("\n")
            for at in atoms:
                file.write("{} {} {:3.6f} {:3.6f} {:3.6f}\n".format(int(at[0]),int(at[1]),at[2]/Lx,at[3]/Ly,at[4]/Lz))





def read_data(file,do_scale=True,atom_style="full"):
    """
    read a data file

    Two styles are implemented : full and atom
    Be careful that they do not have the same return



    """

    if atom_style == "full":
        BOX = []
        list_at_t = []
        for line in open(file,"r"):
            lsplit = line.split()
            if "xlo" in line or "ylo" in line or "zlo" in line:
                BOX.append([float(lsplit[0]),float(lsplit[1])])

            if len(lsplit) == 7:
                Lx = BOX[0][1] - BOX[0][0]
                Ly = BOX[1][1] - BOX[1][0]
                Lz = BOX[2][1] - BOX[2][0]
                if do_scale:
                    list_at_t.append([int(lsplit[0]),int(lsplit[2]),float(lsplit[4])/Lx,float(lsplit[5])/Ly,float(lsplit[6])/Lz])
                else: list_at_t.append([int(lsplit[0]),int(lsplit[2]),float(lsplit[4]),float(lsplit[5]),float(lsplit[6])])
        list_ATOMS = [list_at_t]
        list_BOX = [BOX]
        return list_BOX, np.array(list_ATOMS)

    elif atom_style == "atom":
        Lims = []
        Atom_types = []
        Atom_pos = []
        for line in open(file):
            lsplit = line.split()
            if len(lsplit) == 4:
                Lims.append([float(lsplit[0]),float(lsplit[1])])

            elif len(lsplit) == 6:
                Atom_types.append(int(lsplit[1]))
                Atom_pos.append([float(lsplit[3]),float(lsplit[4]),float(lsplit[5])])
        Atom_pos = np.array(Atom_pos)
        Atom_pos = Atom_pos - np.array([0,0,np.min(Atom_pos[:,2])])
        Lims[2][1] = Lims[2][1] - Lims[2][0]
        Lims[2][0] = 0
        return np.array(Lims),np.array(Atom_types),Atom_pos



def convert_dump_to_xyz(file):
    """
    Convert a dump file to a xyz file

    This should only be used with dump files created with SHAnC as the types are hardcoded (for now)

    """

    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file)



    tx,Tx = list_BOX[0][0]
    ty,Ty = list_BOX[0][1]
    tz,Tz = list_BOX[0][2]

    Lx = Tx-tx
    Ly = Ty-ty
    Lz = Tz-tz

    last_ATOMS = list_ATOMS[-1]
    list_Type = (last_ATOMS[:,1] != 5) * (last_ATOMS[:,1] != 6)

    list_ATOMS_sel = last_ATOMS[list_Type]

    with open(file.split(".")[0] + ".xyz","w") as f:
        f.write(str(len(list_ATOMS_sel)))
        f.write("\n")
        f.write("\n")
        for num,type_at,x,y,z in list_ATOMS_sel:
            if type_at == 1:
                type_write = "Si "
            elif type_at == 4:
                type_write = "H "
            else: type_write = "O "
            xs = x*Lx + tx
            ys = y*Ly + ty
            zs = z*Lz + tz

            f.write(type_write + " {:3.6f} {:3.6f} {:3.6f}\n".format(xs,ys,zs))
    return

def convert_data_to_xyz(file):
    list_BOX,list_ATOMS = read_data(file,do_scale=False)
    # print(np.shape(list_ATOMS))
    list_ATOMS = list_ATOMS[0]

    with open(file.split(".")[0] + ".xyz","w") as f:
        f.write(str(len(list_ATOMS)))
        f.write("\n")
        f.write("\n")
        for num,type_at,x,y,z in list_ATOMS:
            if type_at == 1:
                type_write = "Si "
            elif type_at == 4:
                type_write = "H "
            else: type_write = "O "
            # xs = x*Lx + tx
            # ys = y*Ly + ty
            # zs = z*Lz + tz

            f.write(type_write + " {:3.6f} {:3.6f} {:3.6f}\n".format(x,y,z))

def read_xyz(file_name, ):
    type_map = {"Si": 5, "O": 6, "H" : 7}
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0])
    atoms = []
    coords = []

    for line in lines[2:2+n_atoms]:
        parts = line.split()
        symbol = parts[0]

        atoms.append(type_map[symbol])
        coords.append([float(x) for x in parts[1:4]])
    
    return atoms, np.array(coords)


def write_data(file_name,Pos,Types,Lims,D=0,test_particle = False,Bonds_OH=[],Angles_OH=[]):
    H_present = False
    if (3 in Types) or (7 in Types):
        H_present = True

    bonds = False
    angles = False
    if len(Bonds_OH) > 0:
        bonds = True
    if len(Angles_OH) > 0:
        angles = True

    lines = ["1 28.0855", "2 15.9994", "3 15.9994", "4 1.0080", 
             "5 28.0855", "6 15.9994", "7 1.0080", "8 14.0067"] 

    with open(file_name,"w") as file:
        file.write("\n")
        file.write("{} atoms".format(len(Types)))
        file.write("\n")

        if bonds and H_present:
            file.write("{} bonds".format(len(Bonds_OH)))
            file.write("\n")
        if angles and H_present:
            file.write("{} angles".format(len(Angles_OH)))
            file.write("\n")
        file.write("\n")

        if test_particle == True:
            file.write("{} atom types".format((np.max(Types)-np.min(Types) + 1)))
        else:
            file.write("{} atom types".format(np.max(Types)))
        file.write("\n")

        if bonds and H_present:
            file.write("1 bond types")
            file.write("\n")
        if angles and H_present:
            file.write("1 angle types")
            file.write("\n")

        file.write("\n")
        file.write("{:3.6f} {:3.6f} xlo xhi".format(Lims[0][0],Lims[0][1]))
        file.write("\n")
        file.write("{:3.6f} {:3.6f} ylo yhi".format(Lims[1][0],Lims[1][1]))
        file.write("\n")
        file.write("{:3.6f} {:3.6f} zlo zhi".format(Lims[2][0],Lims[2][1]))
        file.write("\n")
        file.write("\n")

        if test_particle == True: 
            file.write("Masses")
            file.write("\n")
            file.write("\n")
            for line in lines:
                tokens = line.split()
                type_line = int(tokens[0])
                unique_types = np.unique(Types)
                if type_line in unique_types:
                    #print(line)
                    file.write(line)
                    file.write("\n")
        else:  
            file.write("Masses")
            file.write("\n")
            file.write("\n")
            file.write("1 28.0855")
            file.write("\n")
            file.write("2 15.9994")
            file.write("\n")

            if H_present:
                file.write("3 15.9994")
                file.write("\n")
                file.write("4 1.0080")
                file.write("\n")

        file.write("\n")
        file.write("Atoms")
        file.write("\n")
        file.write("\n")

        for num,pos,typ in zip(range(len(Pos)),Pos,Types):
            num = str(num+1)
            file.write(num + " "*(8-len(num)) + "1 {} 0.0 {:3.6f} {:3.6f} {:3.6f}\n".format(typ,pos[0],pos[1],pos[2]))

        if bonds and H_present:
            file.write("\n")
            file.write("Bonds")
            file.write("\n")
            file.write("\n")
            for num,bond in zip(range(len(Bonds_OH)),Bonds_OH):
                file.write("{} 1 {} {}\n".format(num+1,bond[0],bond[1]))

        if angles and H_present:
            file.write("\n")
            file.write("Angles")
            file.write("\n")
            file.write("\n")
            for num,bond in zip(range(len(Angles_OH)),Angles_OH):
                file.write("{} 1 {} {} {}\n".format(num+1,bond[0],bond[1],bond[2]))

if __name__ == "__main__":
    # list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump("dumps_compiled.lammpstrj")
    # write_dump("dumps_trimmed.lammpstrj",list_TSTEP[::10], list_NUM_AT[::10], list_BOX[::10], list_ATOMS[::10])
    # list_BOX,list_ATOMS = read_data("quartz_dupl.data",do_scale=False)
    # write_dump("quartz_dupl.lammpstrj",[0], [len(list_ATOMS[0])], list_BOX, list_ATOMS)
    convert_dump_to_xyz("demo/dummps_round_3_last.lammpstrj")



