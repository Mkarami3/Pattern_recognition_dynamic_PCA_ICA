
from Read   import *
from DPCA   import *
from DICA   import *
from visual import *

plt.close("all")

'''
Read the file 
Creating grid for data visualization
Interpolating the missing data 
'''
Cp = Read_File(C=1) 
x,y = Grid_Gen(isPlot=False) 

#Cp_interp = Interp_missing(write=True)
Cp_interp = np.loadtxt('Cp_interp.txt')

'''
Execute PCA
'''
Number_modes = 5
pca = PCA(Cp_interp,Number_modes)
pca_modes,pca_eigen = pca.execute()
pca.visualize()

'''
Execute DynamicPCA
'''

Number_modes = 5
f1 = 1
f2 = 11
sampling_fequency = 700
dynamic_pca = Dynamic_PCA(Cp_interp,Number_modes,f1,f2,sampling_fequency)
modes,eigen = dynamic_pca.execute()

mode_no = 1 # Select mode number for visualization
dynamic_pca.Visualize(mode_no)

'''
Execute ICA technique
'''
'''
Fun for computing non-Gaussianity using negentropy:
Fun = 1:  g_y = Sk .* exp(-0.5 * Sk.^2)
Fun = 2:  g_y = Sk .^ 3
Fun = 3:  g_y = tanh(Sk)
'''

Fun = 1
Number_modes = 5
ica = ICA(Cp_interp,Number_modes,Fun)
ica.execute()
ica.visualize()

'''
Execute Dynamic ICA
'''

Number_modes = 3
f1 = 1
f2 = 11
sampling_fequency = 700
Fun = 1

dynamic_ica = Dynamic_ICA(Cp_interp,Number_modes,Fun,f1,f2,sampling_fequency)
modes = dynamic_ica.execute()

mode_no = 1 # Select mode number for visualization
dynamic_ica.Visualize(mode_no)

