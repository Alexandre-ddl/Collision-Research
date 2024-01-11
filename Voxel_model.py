import pyvista as pv
import numpy as np
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Voxels():
    
    def __init__(self, scale_print, scale_voxel):
        """
    
        Args :
    
            scale_print (float) : the size of the printing area
            scale_voxel (float) : the dimension of a voxel
    
        """
        
        self.scale_print = scale_print 
        self.scale_voxel = scale_voxel

        self.nb_voxels = torch.ceil(torch.tensor(scale_print, device=device) / torch.tensor(scale_voxel, device=device)).int()
        
        self.voxels = np.zeros((self.nb_voxels,self.nb_voxels,self.nb_voxels)) #creation of the voxel model initialized with a density value equal to 0 

        self.voxels  = torch.tensor( self.voxels , device=device)
        self.midle = (torch.tensor([1., 1., 0.], device=device) * self.nb_voxels / 2).int()

        
        
    def add_density(self, coord):
        """
    
        Args :
    
            coord (numpy array) : coordinate where density must 
                                  be added (there is one voxel)
            
    
        """

        self.voxels[coord[:,0],coord[:,1],coord[:,2]]= 1

    def substracted_density(self,coord):
        """
    
        Args :
    
            coord (numpy array) : coordinates where the 
                                  density must be removed (there is no voxel)
            
    
        """
        
        self.voxels[coord[:,0],coord[:,1],coord[:,2]]= 0

    def density(self, coord):
        """
    
        Args :
    
            coord (numpy array) : test to see if any of the 
                                  coordinates contain density 
            
        Return :
    
            (bool) : True -> there is density (equivalent to collision) 
            
            
        """
        
        density_bool = self.voxels[coord[:,0],coord[:,1],coord[:,2]] == 1
        return np.any(density_bool)
 
    

        
       

