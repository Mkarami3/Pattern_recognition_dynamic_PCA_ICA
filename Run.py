
from Read   import *
from DPCA   import *
from visual import *
plt.close("all")

## Read the File
Cp = Read_File(C=1) # Raw data including missing data

## Creating grid 
x,y = Grid_Gen(isPlot=False) # Position 

## Interpolating missing data
# You can either run the interpolation function or load the interploated results

Cp_interp = Interp_missing(write=True)
#Cp_interp = np.loadtxt('Cp_interp.txt')

# PCA modes
Number_modes = 5
pca = PCA(Cp_interp,Number_modes)
pca_modes,pca_eigen = pca.execute()
pca.visualize()

# Dynamic_PCA
Number_modes = 5
f1 = 1
f2 = 11
sampling_fequency = 700
dynamic_pca = Dynamic_PCA(Cp_interp,Number_modes,f1,f2,sampling_fequency)
modes,eigen = dynamic_pca.execute()

mode_no = 1 # Select mode number for visualization
dynamic_pca.Visualize(mode_no)

              