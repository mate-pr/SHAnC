import distorsion
import matplotlib.pyplot as plt
import pyvista as pv 
import script_analysis
import read_write
import os 
from read_write import *
from script_analysis import *
from distorsion import *
from pathlib import Path

### Definition of the parameters and creation of the files ###
rota = 1.0
Pitch_list = [100, 200, 400] #angstrom maybe exp data bc otherwise relations doon't work no more
print("P=",Pitch_list)

Width_list = [int(((0.29*p+89))/2) if p < 300 else int(((0.29*p)+89))  for p in Pitch_list] 
print("W=",Width_list)

Thickness_list = [int(0.5*Width_list[i]+10) if Pitch_list[i] < 300 else int(0.5*Width_list[i]-20) for i in range(len(Pitch_list))]
print("T=",Thickness_list)

#Thickness_list = []
D_list=[]
for i in range(len(Pitch_list)):
    if Pitch_list[i] < 300:
        D_exp = int(((0.26*Pitch_list[i]+123))/2) #int take the infior value 
    else:
        D_exp = int(((0.26*Pitch_list[i]+ 123)))
    print("Dexp=",D_exp) 
    D_list.append(D_exp)

### Real values ###
rota = 1.0
D = 244
P = 453
T = 112
W = 226


### Proportional lists ###
Pitch_list = [100, 120, 150, 200, 400]
print("P=", Pitch_list)
D_list = [(p/P)*D for p in Pitch_list]
print("D=", D_list)
Thickness_list = [(p/P)*T + 10 if p< 200 else (p/P)*T for p in Pitch_list]
print("T=", Thickness_list)
Width_list = [(p/P)*W for p in Pitch_list]
print("W=", Width_list)
Int_thick_list = [10, 10, 10, 15, 15]


a = 3
Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int, slide_z, mean = create_syst(rota,D_list[a],Pitch_list[a],Width_list[a],Thickness_list[a],Int_thick_list[a], 
                                                                                                            do_clean=True,circling=True,do_rota_transf=False)


print(f"{type} Number of Si : ",np.sum(Types==1))
D_calculated = (np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0]))
print("Diameter of the system : ",(np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0])))          
curvature_analysis(Pos_transfo)



Pos = np.append(Pos_transfo,Pos_transfo_int,axis=0)
Types_int = Types_int + 4
print(np.min(Types_int),np.max(Types_int))
Types = np.append(Types,Types_int)
plot_syst(Pos,Types)





