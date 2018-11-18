import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from matplotlib.collections import PatchCollection
import matplotlib
import operator
from functools import reduce 
import matplotlib.colors as mpc
import matplotlib.cm as cm

def Polygon_collec():
    '''
    Creating Polygon for meshing the grid
    and visualization of Cp_interp data
    '''
    
    r  = (np.array([1.4, 1.2, 1.0,0.8, 0.7, 0.6, 0.5, 0.4, 0.3,    
                   0.25, 0.2, 0.15, 0.10, 0.075, 0.05, 0.025, 0]))
    
    rr = np.dot((np.add(r[0:16],r[1:17])),0.5)
    rr = np.insert(rr,0,1.5)
    
    theta = np.dot(np.arange(33),2*np.pi/32)
    
    patches = []
    x_cor = []
    y_cor = []
    
    for i in range(32):
        
        Poly_th = theta[i:i+2].reshape(1,-1)
        
        for j in range(16):
            x_cor.append(np.dot(rr[j],np.cos(theta[i])))
            y_cor.append(np.dot(rr[j],np.sin(theta[i])))
            
            Poly_r = rr[j:j+2].reshape(-1,1)
            Poly_x = np.multiply(Poly_r,np.cos(Poly_th))
            Poly_y = np.multiply(Poly_r,np.sin(Poly_th))
            Poly_x = reduce(operator.add,Poly_x.tolist()) 
            Poly_y = reduce(operator.add,Poly_y.tolist())
            
            Verts= list(zip(Poly_x,Poly_y)) 
            Verts[2], Verts[3] = Verts[3], Verts[2] 
            
            polygon = Polygon(Verts,closed=True)
            patches.append(polygon)
    
    x_cor.append(0)  
    y_cor.append(0)
    
    circle = Circle((0, 0), rr.min())
    patches.append(circle)        
    collection = PatchCollection(patches)
    
    return collection

def Plot_press(p, color_list=['blue','white','red'],txt=False,num=1
               ,save_fig = False,update=True,cbar = False):
    '''
    p is 1D array. 
    Here it is (513,)
    '''
    
    collection = Polygon_collec() # Importing Polygons
    
    cc = mpc.LinearSegmentedColormap.from_list("",color_list,N=125)
    
    m    = 0.8* np.max(np.abs(p)) # Set colorbar range
#    minima =  p.min()
#    maxima =  p.max()
    
    norm = matplotlib.colors.Normalize(vmin=-m, 
                                       vmax= m, clip=False)
    
    mapper = cm.ScalarMappable(norm=norm, cmap=cc)
    
    colors = mapper.to_rgba(p) # Mapping values of p data to color
    
       
    fig = plt.figure(num)
    ax = plt.gca()
    ax.add_collection(collection) # Adding Polygongs
    
    collection.set_facecolor(colors)
    collection.set_edgecolor('black')
    collection.set_linewidth(0.25)
    
    plt.axis('scaled')
    ax.set_xlim(-0.9,0.9)
    ax.set_ylim(-0.9,0.9)   
    plt.xticks([], [])
    plt.yticks([], [])
    
    if txt:
        ax.text(0.74,-0.82, r'$\phi_%i$' %num, fontsize=15,
                     bbox=dict(facecolor='white'))
    if cbar:
       mapper.set_array([]) 
       fig.colorbar(mapper, ax = ax)
    
    if save_fig:
       fig.savefig('PCA%i.png' %num) 
    
    plt.show()  


