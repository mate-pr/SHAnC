import distorsion
import matplotlib.pyplot as plt
import pyvista as pv 
import script_analysis
import read_write
from read_write import *
from script_analysis import *
from distorsion import *


### Test of creation and analysis of a system ###

# #Parameters definition 
rota = 1.0
# D = 310 
# pitch = 580
# width = 260
# thickness = 114
# int_thick = 20


# #Create the system 
# Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,
#                                                                                              int_thick,do_clean=True,
#                                                                                              circling=True,do_rota_transf=False)
# print("Number of Si : ",np.sum(Types==1))
# print("Diameter of the system : ",(np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0])))          

# #Write data file 
# list_BOX, list_ATOMS = read_data("quartz_dupl.data", do_scale=False) #Why false ? 
# write_dump("Test_system.lammpstrj", [0], [len(list_ATOMS[0])], list_BOX, list_ATOMS)

# #Read data file
# file = "Test_system.lammpstrj"
# list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)

# #Analyze data file 
# list_TSTEP=[0]
# list_Pos = list_ATOMS[:,:,2:]
# list_Types = list_ATOMS[:,:,1]
# Pos = list_ATOMS[-1][:,2:]
# Types = list_ATOMS[-1][:,1]
# Lims = list_BOX[-1]

# #analyze_defects(Pos, Types, True, Lims) #Bus error for big systems 
# curvature_analysis(Pos)
# print("Analysis")
# #analyze_mult(list_TSTEP,list_Pos,list_Types,periodic=False,Lims=list_BOX[-1],save=False)


### Linear regression ###

def linear_regression(x,y):
    if len(x) != len(y):
        raise ValueError("x and y need same size")
    n= len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi*xi for xi in x)
    sxy = sum(xi*yi for xi,yi in zip(x,y))

    denom = n*sxx - sx*sx
    if denom == 0:
        raise ValueError("No regression x constant")
    a = (n*sxy - sx*sy)/denom
    b = (sy - a*sx)/n

    return a,b

Pitch_list = [i for i in range (300, 1000, 50)]
Width_list = [int(((0.293*p)+90)) for p in Pitch_list] 
Thickness_list = [int(0.5*Width_list[i]-20) for i in range(len(Pitch_list))]
Dexp_list = []
Dcal_list = []
for i in range(len(Pitch_list)):
    print(Pitch_list[i])
    D_exp = int(((0.255*Pitch_list[i]+ 123))) #int take the infior value 
    print("Dexp=",D_exp)
    Dexp_list.append(D_exp)

    Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D_exp, Pitch_list[i],Width_list[i],Thickness_list[i],
                                                                                                    int_thick = 20, do_clean=True,
                                                                                                circling=True,do_rota_transf=False)
    print(f"{type} Number of Si : ",np.sum(Types==1))
    D_calculated = (np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0]))
    print("Diameter of the system : ",D_calculated)          
    Dcal_list.append(D_calculated)

print("Linear regression")
a,b = linear_regression(Dexp_list,Dcal_list)
print("a = ", a,"b =", b)


y_fit = [a*d_exp + b for d_exp in Dexp_list]
plt.figure()
plt.scatter(Dexp_list,Dcal_list)
plt.plot(Dexp_list,y_fit)
plt.xlabel("D_exp")
plt.ylabel("D_calculated")
plt.legend()
plt.show()
plt.savefig()
plt.close()



