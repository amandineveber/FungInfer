"""
This file contains the definitions of the classes Reseau, Brindille, and Branche.
These three classes were developed to analyse the fungal networks of *Podospora anserina* by uniquely defining a dynamic graph for each experiment.
For more details, refer to the article:
"Full identification of a growing and branching network's spatio-temporal 
structures", T. Chassereau, F. Chapeland-Leclerc, and E. Herbert, 2024-25

This is a reduced version that does not allow reconstruction from images but 
allows working with the Reseau objects for the intended analysis.
"""

import numpy as np #For everything
import matplotlib.pyplot as plt #For visualisation
import networkx as nx #For graph object
from tqdm import tqdm #For nice loading bar
import os  #For creating folder
import pickle #For saving object

class Reseau():
    """
    Definition of the Reseau class.    

    Each instance of this class contains all the information necessary for analysing an experiment, namely all the grayscale images
    over time, the start and end times of the analysis, the two corresponding binarised images, and the spatial graph associated with the last     image.

    For more information on generating the spatial graph, refer to the files 'TotalReconstruction.py' and 'Vectorisation.py'.
    """
    #float|int : Length treshold between 2 nodes over which it is considered false
    SEUIL_LONG:float = 10 # in hyphal diameter unit
    #int : "Latency" threshold before classification as a lateral branch
    SEUIL_LAT:int = 5 #frames
    #float|int : Length threshold for the start of a branch
    SEUIL_DEPART:float = 2 # in hyphal diameter unit
    #int : Threshold for the number of loops during the calculation of the dynamics before considering a case as suspicious.
    SEUIL_BOUCLE_DYNAMIQUE:int = 40

    """
    ===========================================================================
    Declaration and representation
    """
    def __init__(self,
                 name:str, #str, name of the experiment
                 g:nx.Graph, #Networkx graph of the experiment
                 imgs:dict[int,str], #{frame: img_path} Sequence of images
                 manip_dir:str, #str, Experiment's folder
                 output_dir:str, #str, Output folder
                 first, #First image Binarized
                 last, #Last image Binarized
                 start:int,end:int, #ints, idx of start, and end of experiment
                 brindilles:list = None,#Brindilles
                 branches:list = None,#Branches
                 diameter:float = 7#float, Hyphal diamater in pixels
                 ):
        """
        Defines the initialisation of an instance of Reseau.
        """
        self.name = name
        self.g = g
        self.imgs = imgs
        self.first = first
        self.last = last
        self.manip_dir = manip_dir
        self.output_dir = output_dir
        self.start = start
        self.end = end
        #Initialisation of some characteristic of the Reseau
        self.brindilles = brindilles if brindilles is not None else [] #List of twigs
        self.branches = branches if branches is not None else []#List of Branches
        self.source = None
        self.diameter = diameter
        self.n2x = nx.get_node_attributes(g,"x")
        self.n2y = nx.get_node_attributes(g,"y")
        self.n2t = {}
        #Initialisation of the output files
        directories = ["","GraphesVisu",
                       "PotentialErrors","Binarization"]
        for directory in directories:
            if not os.path.exists(output_dir+directory):
                os.mkdir(output_dir+directory)

    def __repr__(self) -> str:
        """
        Defines what is displayed when using 'print' with the network as an argument.
        """
        repr = "\n"
        repr += "-"*80
        repr += f"\nReseau {self.name}\n"
        repr += f"\tStart frame {self.start}\n"
        repr += f"\tEnd frame {self.end}\n"
        repr += "-"*80
        repr += "\n"
        return repr

    """
    ===========================================================================
    Utilities
    """
    def save(self,suffix:str):
        """
        Saves the network in gpickle format
        """
        file = self.output_dir+self.name+"_"+suffix+".gpickle"
        with open(file,"wb") as f:
            pickle.dump(self,f)
        return self
    
    def network_at(self, f):
        """
        Returns the sub-network g_frame extracted from the entire network self.g containing only the points satisfying t <= f
        """
        g_frame = self.g.copy()
        toKeep = [n for n in g_frame.nodes if self.n2t[n]<=f]
        g_frame = g_frame.subgraph(max(nx.connected_components(g_frame.subgraph(toKeep)),
                                          key = len)).copy()
        return g_frame

    def image_at(self, f:int):
        """
        Returns the image corresponding to frame f open as numpy array
        """
        image = np.asarray(Image.open(self.imgs.format(f)))
        return image
    

    def classification_nature_branches(self,threshold:float):
        """
        Classifies each branch according to the latency before branching
        """
        for b in self.branches:
            b.nature = "Apical" if b.get_tstart()-b.t[0] <= threshold else "Lateral"
        return self

    def convert2txt(self,path2file:str)->nx.Graph: 
        """
        Converts the network to a txt file of the form:
        U,V,XU,YU,TU,XV,YV,TV,B
        With (U,V) the edges X_,Y_,T_ the coordinates of the point and B the 
        index of the corresponding branch.
        In the header, informations can be found about the source and its position.
        """
        g_prune = self.prune_d2()
        data = []
        for b in self.branches:
            noeuds= [n for n in b.noeuds if n in g_prune]
            for u,v in zip(noeuds[:-1],noeuds[1:]):
                data.append([u,v,self.n2x[u],self.n2y[u],self.n2t[u],self.n2x[v],self.n2y[v],self.n2t[v],b.index])
        data = np.array(data)
        np.savetxt(path2file,data,fmt="%d",
                   header=f"#Spore = {self.source} XS = {int(self.n2x[self.source])}, YS = {int(self.n2y[self.source])}\n"+
                           "#U,V,XU,YU,TU,XV,YV,TV,B",
                   delimiter=",")
        return self
    """
    ===========================================================================
    Visualisation
    """
    def show_at(self,t:float, ax)->None:
        """
        Draws the network at the instant 't' on the axe 'ax'
        """
        degree2size = {1:10,2:2,3:14}
        degree2color = {1:"cyan",2:"slategrey",3:"orangered"}
        subg = self.g.subgraph((n for n in self.g.nodes if self.n2t[n]<= t))
        nx.draw(subg,pos={n:(self.n2x[n],self.n2y[n])
                          for n in subg.nodes},
                ax=ax,
                node_size=[degree2size.get(d,20) 
                           for n,d in subg.degree],
                node_color=[degree2color.get(d,"red") 
                            for n,d in subg.degree])
        return None

    @property
    def times(self)->np.ndarray:
        """
        Returns the instant from self.start to self.end
        """
        times = np.arange(self.start,self.end+1)
        return times
    
    @property
    def Nbranches(self)->np.ndarray:
        """
        Returns Nbranches the total number of branches at time t 
        for t ranging from self.start to self.end
        """
        N_t0 = np.array([b.get_tstart() for b in self.branches])
        Nbranches = np.array([np.sum(N_t0<=t) for t in self.times])
        return Nbranches
    
    @property
    def total_length(self)->np.ndarray:
        """
        Returns total_length the length of the reseau
        for t ranging from self.start to self.end
        """
        L_edges = np.sqrt([(self.n2x[u]-self.n2x[v])**2+
                        (self.n2y[u]-self.n2y[v])**2
                        for u,v in self.g.edges])
        t_edges = np.array([max(self.n2t[u],self.n2t[v]) 
                            for u,v in self.g.edges])
        total_length = np.array([np.sum(L_edges[np.where(t_edges<=t)])
                                 for t in self.times])
        return total_length


"""
===========================================================================
Brindilles
"""
class Brindille():
    """
    Definition of the Brindilles class.
    """
    def __init__(self,
                index:int,
                noeuds:list[int],
                n2x:dict[int,float],
                n2y:dict[int,float],
                n2t:dict[int,float],
                inBranches:list = [],
                confiance:float = 0):
        self.index = index #index of the twig
        self.noeuds = noeuds 
        self.n2x = n2x
        self.n2y = n2y 
        self.n2t = n2t 
        self.inBranches = inBranches #list of the indices of all the branches containing the twig
        self.confiance = confiance
    
    def __repr__(self) -> str:
        repr = f"Brindille {self.index} - {len(self.noeuds)} noeuds"
        return repr
    
    def abscisse_curviligne(self)->np.ndarray:
        """
        Returns the list of curvilinear abscissas of the branch nodes
        """
        pos = np.array([[self.n2x[n],self.n2y[n]] for n in self.noeuds])
        abscisse = np.sqrt(np.sum((pos[1:,:]-pos[:-1,:])**2,axis=-1))
        abscisse = np.cumsum(abscisse)
        abscisse = np.insert(abscisse,0,0)
        return abscisse
    
    def get_tstart(self)->float:                                        
        """
        Returns the coordinate t corresponding to the start of the twig.
        """
        tstart = self.n2t[self.noeuds[1]]
        return tstart

    def get_tend(self)->float:
        """
        Returns the coordinate t corresponding to the end of the twig.
        """
        tt = [self.n2t[n] for n in self.noeuds]
        tend = np.max(tt)
        return tend

    def is_growing_at(self, t:float)->bool:
        """
        Returns whether or not the branch is growing at the time t given as an argument.
        """
        return self.get_tstart()<=t<=self.get_tend()
    
    def detection_latence(self, seuil:int = 4)->bool:
        """
        Start with latency -> True 
        Start without latency -> False
        """
        bLatence = bool(self.get_tstart()-self.n2t[self.noeuds[0]] < seuil)
        return bLatence
    
    def unit_vector(self, end, r = np.inf):
        """
        Computes the unit vector at the specified endpoint measured with a radius r.
        """
        abscisse = self.abscisse_curviligne()
        if end not in [self.noeuds[0],self.noeuds[-1]]:
            raise ValueError(f"{end} is not an end of twig {self.index}")
        if self.noeuds[0] == end:
            kstop = next((k for k,s in enumerate(abscisse) if s > r),-1)
            u1 = end
            u2 = self.noeuds[kstop]
        else:
            abscisse = reversed(abscisse - abscisse[-1])
            kstop = next((len(self.noeuds)-k-1 for k,s in enumerate(abscisse) 
                        if s < -r),0)
            u1 = self.noeuds[kstop]
            u2 = end
        vect = np.array([self.n2x[u2]-self.n2x[u1],
                            self.n2y[u2]-self.n2y[u1]])
        unit_vect = vect/np.linalg.norm(vect)
        return unit_vect
    
    def reverse(self):
        """
        Reverses the twig.
        """
        self.noeuds.reverse()
        self.confiance = - self.confiance
        return self
    
    def calcul_confiance(self, seuil:float)->float:
        """
        Computes the confidence in the orientation of the twig
        """
        tt = np.array([self.n2t[n] for n in self.noeuds])
        dtt = tt[1:]-tt[:-1]
        #Incrément
        dtt = np.where(dtt>0, 1, dtt)
        #Décrément
        dtt = np.where(dtt<0,-1, dtt)
        #dtt = np.where(np.abs(dtt) != 1, 0, dtt)
        nP = np.sum(np.abs(dtt)+dtt)//2
        nM = np.sum(np.abs(dtt)-dtt)//2
        self.confiance = (nP-nM)/(nP+nM)*np.tanh((nP+nM)/seuil) if nP+nM else 0.
        return self.confiance
    
    def get_apex_at(self, t:float, index:bool = False)->int:
        """
        Returns the current 'apex' on the twig|branch at time t
        if t>tEnd then the apex is the last node of the twig|branch
        if index == True then return also the index of apex in b.noeuds 
        """
        tt = [self.n2t[n] for n in self.noeuds]
        if self.get_tstart()>t:
            raise ValueError("The branch has not yet appeared at this time.")
        iapex = next((i for i,ti in enumerate(tt[1:])
                    if ti > t),
                    len(tt)-1)
        apex = self.noeuds[iapex]
        if index:
            return (iapex,apex)
        return apex

    def get_all_apex(self)->tuple[list[int],list[int]]:
        """
        Returns all the successive apexes at the different growth time
        """
        tstart = int(self.get_tstart())
        tend = int(self.get_tend())
        time = list(range(tstart,tend+1))
        tt = self.t
        nnoeuds = len(self.noeuds)
        ntime = len(time)
        apex = []
        i=0
        it = 0
        while i < nnoeuds-1:
            if tt[i+1]>time[it]:
                apex.append(self.noeuds[i])
                it += 1
            else:
                i += 1
        time = list(range(tstart,tend+1))
        apex.append(self.noeuds[-1])
        while len(apex)<ntime:#Strange, should only occur with branches/twigs made of 2 nodes
            apex.append(self.noeuds[-1])#It's a fix but not a good one...
        return time, apex
    
    @property
    def x(self)->list[float]:
        """
        Returns the list of x coordinates of each node in twig.nodes
        """
        return [self.n2x[n] for n in self.noeuds]
    
    @property
    def y(self)->list[float]:
        """
        Returns the list of y coordinates of each node in twig.nodes
        """
        return [self.n2y[n] for n in self.noeuds]
    
    @property
    def t(self)->list[float]:
        """
        Returns the list of t coordinates of each node in twig.nodes
        """
        return [self.n2t[n] for n in self.noeuds]
    
    @property
    def s(self)->np.ndarray:
        """
        Returns the list of s, arc length of each node in twig.nodes
        """
        return self.abscisse_curviligne()

    @property
    def theta(self,radius:float = 7)->np.ndarray:
        """
        Returns the list of theta, the orientation of the twig.
        Orientation is absolute.
        """
        x = np.array(self.x)
        y = np.array(self.y)
        thetas = np.zeros_like(x)
        #Cas général
        radius_squared = radius*radius
        for i,n in enumerate(self.noeuds):
            x0,y0 = x[i],y[i]
            r2 = (x-x0)*(x-x0)+(y-y0)*(y-y0)
            im = next((i-j for j,rsq in enumerate(r2[i::-1])if rsq>radius_squared ),
                    0)
            ip = next((j for j,rsq in enumerate(r2[i:],start=i) if rsq>radius_squared),
                      -1)
            im = next((i-j for j,rsq in enumerate(r2[i::-1])if rsq>radius_squared ),
                      0)
            thetas[i] = np.arctan2(y[ip]-y[im],x[ip]-x[im])
        return np.unwrap(thetas)
    
    def tangent(self,noeud)->np.ndarray:
        """
        Returns the unit tangent vector of the twig at node 'noeud'.
        """
        if noeud not in self.noeuds:
            raise ValueError(f"{noeud} is not in {self}.\nThe argument 'noeud' must be inside twig/branch.noeuds")
        n2i = {n:i for i,n in enumerate(self.noeuds)}
        i = n2i[noeud]
        ip = min(i+1,len(self.noeuds)-1)
        im = max(i-1,0)
        dx = self.x[ip]-self.x[im]
        dy = self.y[ip]-self.y[im]
        direction = np.array([dx,dy])/np.sqrt(dx*dx+dy*dy)
        return direction

"""
===========================================================================
Branches
"""
class Branche(Brindille):
    """
    Definition of the secondary class of branches.
    Inherits the secondary class of brindilles.
    """
    def __init__(self,
                    index:int, #int, index of the branch
                    noeuds:list[int], #List of the branch nodes
                    n2x:dict[int,float],
                    n2y:dict[int,float],
                    n2t:dict[int,float],
                    brindilles:list[int], #List of the brindilles indices
                    nature:str, #nature de la branche : Apical,Lateral,Initial,...
                    ending:str, #reason the growth arrest : d1, Fusion ?
                    list_overlap:list = None #List of overlaps
                    ):
        self.index = index
        self.noeuds = noeuds
        self.n2x = n2x
        self.n2y = n2y
        self.n2t = n2t
        self.brindilles = brindilles
        self.nature = nature
        self.ending = ending
        self.list_overlap = list_overlap if list_overlap else []
    
    def __repr__(self) -> str:
        repr = f"Branche {self.index} - {len(self.noeuds)} noeuds"
        return repr

    def grow(self,brindille):
        """
        Extension of the branch by the twig.
        """
        self.noeuds = [*self.noeuds,*brindille.noeuds[1:]]
        self.brindilles.append(brindille.index)
        return self

    def normes_vitesses(self, seuil_lat = 4):
        """
        Computes and returns the norms of the branch velocities and the list of 
        corresponding time points.
        seuil_lat allows excluding slow velocities due to latency.

        Returns ([], []) if it is not possible to define a velocity on the branch
        """
        times,vitesses = self.vecteurs_vitesses(seuil_lat = seuil_lat)
        vs = []
        if len(vitesses):
            vs = np.sqrt(np.sum(vitesses*vitesses,axis=1))
        return times,vs

    def vecteurs_vitesses(self, seuil_lat = 4):
        """
        Computes the branch growth velocity vectors.
        Returns a tuple with the list of corresponding time points t
        and the list of velocity vectors.

        Returns [] if it is not possible to define a velocity on the
        branch.
        """
        vecteursV = []
        times = []
        pos = np.array([[self.n2x[n],self.n2y[n]] for n in self.noeuds])
        abscisse = self.s
        tt = np.array(self.t)
        index = [i for i,t in enumerate(tt[:-1]) if t != tt[i+1]]
        #index : list of indices at which there is a variation of t
        if index:
            index.append(len(tt)-1)
            for i0,i in zip(index[:-1],index[1:]):
                deltaT = tt[i]-tt[i0]
                deltaS = abscisse[i]-abscisse[i0]
                deltaPos = pos[i,:]-pos[i0,:]
                if 0<deltaT<seuil_lat:
                    normedPos = np.linalg.norm(deltaPos)
                    direction = deltaPos/normedPos if normedPos else deltaPos
                    vecteursV.append(deltaS/deltaT*direction)
                    times.append((tt[i0]+tt[i])*.5)
        times = np.array(times)
        vecteursV = np.array(vecteursV)
        return times,vecteursV

    def positions_vitesses(self):     
        """
        Returns the list of velocity vectors and corresponding positions of the branch.
        """
        dX,dY,dT = self.n2x,self.n2y,self.n2t
        pos = np.array([[dX[n],dY[n]] for n in self.noeuds])
        tstart = int(self.get_tstart())
        tend = int(self.get_tend())
        _,apex = self.get_all_apex()
        apex.insert(0,self.noeuds[0])
        temps = np.array([t-0.5 for t in range(tstart,tend+1)])
        if len(apex)<2:
            return np.empty(shape=(1,3)),np.empty(shape=(1,2))
        positions = np.zeros(shape=(len(apex)-1,3))
        vecteursV = np.zeros(shape=(len(apex)-1,2))
        positions[:,2] = np.arange(tstart,tend+1)-.5
        abscisse = self.abscisse_curviligne()
        n2i = {n:i for i,n in enumerate(self.noeuds)}
        positions_apex = np.array([pos[n2i[a]] for a in apex])
        positions[:,:2] = (positions_apex[:-1]+positions_apex[1:])*.5
        vecteursV = (positions_apex[1:]-positions_apex[:-1])
        cart_normes = np.linalg.norm(vecteursV,axis=1)
        curv_normes = np.array([abscisse[n2i[a1]]-abscisse[n2i[a2]]
                                for a1,a2 in zip(apex[1:],apex[:-1])])
        filtre = np.where(cart_normes>0)
        vecteursV[filtre,0] *= curv_normes[filtre]/cart_normes[filtre]
        vecteursV[filtre,1] *= curv_normes[filtre]/cart_normes[filtre]
        #Rolling mean 
        #positions = (positions[1:,:]+positions[:-1,:])/2
        #vecteursV = (vecteursV[1:,:]+vecteursV[:-1,:])/2
        return positions,vecteursV
    
    def apex(self)->list[int]:
        """
        Returns the list of successive apexes along the branch
        """
        temps = [self.n2t[n] for n in self.noeuds]
        apex = [self.noeuds[i] for i in range(1,len(self.noeuds)-1) if temps[i] < temps[i+1]]
        apex.append(self.noeuds[-1])
        return apex

