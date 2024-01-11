import numpy as np
import networkx as nx
import trimesh
import time
import torch

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from Voxel_model import Voxels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def sens_angulaire(ang_1,ang_2):
    
    diff = ang_2 - ang_1
    diff_abs = abs(diff)
    diff_modulo = diff%(2*np.pi)
    inf = min(diff_abs , diff_modulo)
    
    if inf == diff_abs :
        return diff
    else : 
        return diff_modulo



def Thread_iteration(args):
    type_object , tool_path, position_head_layer_circle, layer_vertex, vox, i, max_xyz, bead_voxel_coord , add_distance= args
    
    position = position_head_layer_circle
        
    #mask = ((position[:,:,2]<= max_xyz[2])&
    #        (position[:,:,1]<= max_xyz[1])&
    #        (position[:,:,0]<= max_xyz[0]))
    
    
    #H = np.max(np.sum(mask, axis=1))

    #padding = vox.nb_voxels-1
    #padded_tabs = np.array([np.pad(position[n, mask[n], :], ((0, H - np.sum(mask[n])), (0, 0)), 'constant', constant_values=  padding ) for n in range(len(position))])

    
    indices = torch.where(~torch.any(
            vox.voxels[torch.floor(position[:,:,0] + tool_path[0]  + tool_path[3] * (6+add_distance)/vox.scale_voxel ).int(),
                       torch.floor(position[:,:,1]  + tool_path[1] + tool_path[4] * (6+add_distance)/vox.scale_voxel  ).int(),
                       torch.floor(position[:,:,2]  + tool_path[2] + tool_path[5] * (6+add_distance)/vox.scale_voxel ).int()]==1                
                        , axis =1))
    
    #if not vox.density(np.floor(position[mask]+intervale).astype(int)):  
        
    #current_vertex = (i, j, ind)          

    layer_vertex[i].append(type_object[indices[0]]) 
    add_voxel = torch.floor(bead_voxel_coord + tool_path[:3]).int()
    vox.add_density(add_voxel)
    
    
 

def Paralel_graph(arg): 
        G , i , layer_vertex , theta,source , target ,num_layers , Circle =arg

        if i > 0:
            prev_layer = i - 1
            prev_layer_vertices = layer_vertex[prev_layer]
        else :
                prev_layer_vertices = source

        for ele in layer_vertex[i]:
            current_vertex = ele
            G.add_node(current_vertex, angle_z=ele[2] * theta)
            if i > 0:
                for prev_vertex in prev_layer_vertices: 
                    angle = abs(G.nodes[prev_vertex]['angle_z'] - G.nodes[current_vertex]['angle_z'])
                    angle_Z = abs(Circle[current_vertex[1]][current_vertex[2]] - Circle[prev_vertex[1]][prev_vertex[2]])
                    if angle <= theta : #& angle_Z < 2*np.pi: 
                            G.add_edge(prev_vertex, current_vertex, weight=abs(sens_angulaire(Circle[prev_vertex[1]][prev_vertex[2]],Circle[current_vertex[1]][current_vertex[2]]))) #2*np.pi/ (2*np.pi-  angle_Z)**10)
                    else : 
                        G.add_edge(prev_vertex, current_vertex, weight=abs(sens_angulaire(Circle[prev_vertex[1]][prev_vertex[2]],Circle[current_vertex[1]][current_vertex[2]]))) #2*np.pi/ (2*np.pi-  angle_Z)**10)
            if i == 0:
                G.add_edge(source, current_vertex, weight=0)
            if i == num_layers - 1:
                G.add_edge(current_vertex, target, weight=0)

def Free_path_parallelize_ThreadPool(vox, 
                                     tool_path, 
                                     theta, 
                                     position_head_layer_circle, 
                                     G, 
                                     layer_vertex , 
                                     max_xyz,
                                     bead_voxel_coord,
                                     type_object):
    
    num_layers = len(tool_path)
    source = 's'
    target = 't'

    G.add_node(source)
    G.add_node(target)
    Error = []
            
    #with ThreadPoolExecutor(8) as executor:
                    
    iterations_args = [(type_object,tool_path[i], position_head_layer_circle, layer_vertex, vox,i, max_xyz, bead_voxel_coord ,0) for i in range(num_layers) ]
    for ele in  iterations_args : 
            Thread_iteration(ele)
            # Parallel calls to the Thread_iteration function
            #executor.map(Thread_iteration, iterations_args)
              
            #add_voxel = np.floor(bead_voxel_coord + tool_path[i][:3]).astype(int)
            #vox.add_density(add_voxel)
    
    
    for ele in layer_vertex : 
        if len(layer_vertex[ele]) == 0 : 
                Error.append(ele)
    
    
    #return G , layer_vertex , theta,source , target ,num_layers , Circle , Error

    if not Error : 
        pass
        #with ProcessPoolExecutor(8) as executor:
        #    iterations_args = [(G , i , layer_vertex , theta,source , target ,num_layers , Circle) for i in range(num_layers)]
        #    executor.map(Paralel_graph, iterations_args) 
            #for arg in iterations_args : 
            #    Paralel_graph(arg)
            
        #    shortest_path = nx.dijkstra_path(G, source, target)
    
    else : 
        pass
        
        #iteration_layer = 0 
        #   while (len(layer_vertex[i]) == 0) & iteration_layer < 4 : 
        #        iteration_layer += 1
        #        add_distance = 0.5 * iteration_layer
        #       iterations_args = [(Circle,delta, ind, j,G,theta,tool_path, position_head_layer_circle, layer_vertex, prev_layer_vertices, vox ,i, target, source , max_xyz, min_xyz,bead_voxel_coord, test,add_distance) for j in Circle for ind, delta in enumerate(Circle[j])]
        #        executor.map(Thread_iteration, iterations_args) 
        
        #shortest_path = False
    return Error #shortest_path 


if __name__ == "__main__":
     
    G = nx.DiGraph()
    type_object = np.load('data/next_inclinaison.npy') # en radian
    bead_voxel_coord = np.load('data/bead_voxel_coord.npy')
    scale_print,scale_voxel,theta = np.load('data/params.npy')
    tool_path = np.load('data/tool_path.npy')
    tool_path = tool_path[:2000]
    colli_voxel_coord = np.load('data/colli_voxel_coord.npy')
    position_head_layer_circle = np.load('data/position_head_layer_circle.npy')
    layer_vertex = {i: [] for i in range(len(tool_path))} 
    max_xyz = np.max(colli_voxel_coord, axis = 0)
    vox = Voxels(scale_print,scale_voxel)

    trimesh_collision = trimesh.load('data/Mold coating.stl')
    min_bound , max_bound = trimesh_collision.bounds.astype(int)
    trimesh_collision.vertices-= min_bound 
    trimesh_collision.vertices /= scale_voxel

    translation = (vox.midle - torch.tensor(trimesh_collision.center_mass, device=device)).int()
    translation[2]=0
    trimesh_collision.vertices+=translation.to('cpu').numpy()
    colli_voxel_coord+=translation.to('cpu').numpy()
    tool_path[:,:3] +=translation.to('cpu').numpy()

    print(tool_path.shape)
    vox.add_density(colli_voxel_coord)
    print(vox.nb_voxels)

    #vox.density(colli_voxel_coord)

    #scene = trimesh.Scene()
    #points = trimesh.points.PointCloud(tool_path[:,:3], colors=[255, 0, 0, 255]) 
    #points2 = trimesh.points.PointCloud(colli_voxel_coord, colors=[0, 0, 0, 255])   
    #scene.add_geometry([trimesh_collision, points,points2])

    #scene.show()
    debut = time.perf_counter()
    Error = Free_path_parallelize_ThreadPool(vox, 
                                     torch.tensor(tool_path,device=device), 
                                     torch.tensor(theta,device=device), 
                                     torch.tensor(position_head_layer_circle,device=device), 
                                     G, 
                                     layer_vertex , 
                                     max_xyz,
                                     torch.tensor(bead_voxel_coord,device=device),
                                     torch.tensor(type_object,device=device))
    fin = time.perf_counter()
    print(fin-debut ,(fin-debut) /len(tool_path) )

    print(layer_vertex)


