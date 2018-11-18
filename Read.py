
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata

def Read_File(C=1):
    
    '''
    Importing experimental data
    C is Swirl ratio case
    C = 1 : Swirl ratio = 0.22
    C = 2 : Swirl ratio = 0.57
    C = 3 : Swirl ratio = 0.96
    Note that Swirl ratio 0.57 & 0.96 are not uploaded
    '''
    
    Case={}
    Case[1] = 'Cp_tap.mat'
    Case[2] = 'Cp_tap_0.57.mat'
    Case[3] = 'Cp_tap_0.96.mat'
    Cp = loadmat(Case[C])
    
    return Cp['Cp_tap']

def Interp_missing(write=True):
    '''
    Interpolating the missing pressure coefficient values
    '''
    Cp = Read_File(C=1)
    x,y = Grid(isPlot=False)
    x_i,y_i = Grid_Gen(isPlot=False)  
    
    
    Num_taps = [16, 11, 14, 11, 15, 11, 14, 11]*4
    
    # 16 taps in radial direction and 32 taps in azimuthal direction
    Boolean_taps = np.zeros((32, 16), dtype=bool)
    
    for i in range(32):
        Boolean_taps[i,0:Num_taps[i]] = True
    
    Boolean_taps = Boolean_taps.reshape(-1)
    
    Cp_taps = (np.concatenate((Cp[:,Boolean_taps],Cp[:,411].
                               reshape(-1,1)),axis=1))
    
    Cp_interp = np.zeros((42000,513))
    
    for i in range(Cp_taps.shape[0]):
        
        Cp_instan = Cp_taps[i,:] 
        Cp_interp[i,:] = griddata((x,y), Cp_instan, (x_i, y_i), method='cubic')
    
    if write:
        np.savetxt('Cp_interp.txt', Cp_interp)
        
    return Cp_interp


def Grid(isPlot=True):
    '''
    Creating Polar Grid for Pressure Taps Positions
    '''
    r = (np.array([1.4, 1.2, 1.0,0.8, 0.7, 0.6, 0.5, 0.4, 0.3,    
            0.25, 0.2, 0.15, 0.10, 0.075, 0.05, 0.025, 0]))  
    
    theta = np.linspace(start = 0, stop = 2*np.pi, num=32, endpoint=False)
    
    # Number of pressure taps in each radial direction
    # e.g. 16 taps are working in the first radial direction
    Num_taps = [16, 11, 14, 11, 15, 11, 14, 11] *4    
    
    r_list = [None] * theta.shape[0]
    for i in range(theta.shape[0]):
        r_list[i] = r[0:Num_taps[i]]
    
    x = []
    y = []
    for i in range(32):
        x.extend(np.dot(r_list[i],np.cos(theta[i])))
        y.extend(np.dot(r_list[i],np.sin(theta[i])))  
        
    x.append(0)
    y.append(0)
    
    if isPlot:
        plt.ion()
        for i in range(theta.shape[0]):
            (plt.scatter(np.dot(r_list[i],np.cos(theta[i])),
                        np.dot(r_list[i],np.sin(theta[i]))))    
    
    return x,y

    

def Grid_Gen(isPlot=True):
    '''
    Creating New Grid for Interpolating the missing Data.
    Not all pressure taps are used. Only 413 out of 512 taps are used.
    '''
    
    Num_taps_interp = [16]*32

    r = (np.array([1.4, 1.2, 1.0,0.8, 0.7, 0.6, 0.5, 0.4, 0.3,    
            0.25, 0.2, 0.15, 0.10, 0.075, 0.05, 0.025, 0]))  
    
    theta = np.linspace(start = 0, stop = 2*np.pi, num=32, endpoint=False)
    
    r_list = [None] * theta.shape[0]
    for i in range(theta.shape[0]):
        r_list[i] = r[0:Num_taps_interp[i]]
    
    x_interp = []
    y_interp = []
    for i in range(32):
        x_interp.extend(np.dot(r_list[i],np.cos(theta[i])))
        y_interp.extend(np.dot(r_list[i],np.sin(theta[i])))  
        
    x_interp.append(0)
    y_interp.append(0)
    
    if isPlot:
        plt.ion()
        for i in range(theta.shape[0]):
            (plt.scatter(np.dot(r_list[i],np.cos(theta[i])),
                        np.dot(r_list[i],np.sin(theta[i]))))    
    
    return x_interp,y_interp