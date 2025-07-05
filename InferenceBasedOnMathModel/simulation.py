"""
This file contains the definitions of the hypha and mycelium classes.
These classes are used to encode the mycelium in the simulation of the process.
"""

import numpy as np

#=============================================================================#

class hypha:
    """
    Definition of the hypha class. 
    
    Attributes
    ----------
    _len: float
          length of the hypha
    _e: bool
        True if open, False if closed
    """
    def __init__(self, x=0, e=True):
        self._len = x
        self._e = e

    def elongation(self, T=1):
        """
        Updates the length of the hypha after a duration T
        """
        if self._e : 
            self._len += T #open hyphae lengthen at speed 1
        
    def length(self):
        """
        Returns the length of the hypha
        """
        return self._len
    
    #=========================================================================#
    #Update the hypha after branching and create the two new hyphae that result from the branching event
    def apical_branching(self):
        """
        Updates the hypha after apical branching
        """
        self._e = False #Close the hypha
        new1 = hypha(0, True)
        new2 = hypha(0, True) #Create two open hyphae of length 0
        return(new1, new2)
    
    def lateral_branching(self):
        """
        Updates the hypha after lateral branching
        """
        if self.length() != 0:
            i = np.random.uniform(0, self.length()-1e-5) #Draw the branching point
            
            new2 = hypha(self.length()-i, self._e) #Create a hypha of the same type with length x-i
           
            new3 = hypha() #Create an open hypha of length 0
            
            self._e = False #Close the hypha
            self._len = i #Update the length
            
            return(new2, new3)
                
#=============================================================================#    

class mycelium:
    """
    Definition of the mycelium class

    Attributes
    ----------
    _hyphae: list[hypha]
             list of hyphae
    _b1: float
         apical branching rate
    _b2: float
         lateral branching rate
        
    _T: float
        lifetime
    """
    def __init__(self, b1, b2):
        self._hyphae = []
        
        for i in range(3):
            self._hyphae.append(hypha(np.random.uniform(0,0.1)))            
        
        self._b1 = b1
        self._b2 = b2
        
        self._T = 0 
    
    def run(self, Tmax):
        """
        Updates the mycelium up to time Tmax
        """
        from time import time

        t = time()
        while self._T < Tmax :
            self.actualisation()            
            
        t = time() - t
        
        return(t)
    
    def run_to_n(self, n_max):
        """
        Updates the mycelium until the number of hyphae is greater than n_max
        """
        from time import time
                
        t = time()
        while len(self) <= n_max :
            self.actualisation()            
            
        t = time() - t
        
        return(t)
        
    def __len__(self):
        """
        Returns the number of hyphae
        """
        return len(self._hyphae)
    
    def __getitem__(self, i):
        """
        Returns the ith hypha
        """
        return self._hyphae[i]
    #=========================================================================#
    def n_apex(self):
        """
        Returns the number of apex
        """
        return sum([self[i]._e for i in range(len(self))])
    
    def total_length_open(self):
        """
        Returns the total length of the open hyphae
        """
        return sum([self[i].length() for i in range(len(self)) if self[i]._e])
    
    def total_length(self):
        """
        Returns the total length of the mycelium
        """
        return sum([self[i].length() for i in range(len(self))])
    
    def length_open(self): 
        """
        Returns a list of the lengths of each open hypha
        """
        return [self[i].length() for i in range(len(self)) if self[i]._e]
    
    def length_closed(self):
        """
        Returns a list of the lengths of each closed hypha
        """
        return [self[i].length() for i in range(len(self)) if not self[i]._e]
    
    def length_hyphae(self):
        """
        Returns a list of the lengths of each hypha
        """
        return [self[i].length() for i in range(len(self))]
    #=========================================================================#

    def apical_branching(self):
        """
        Draws the waiting time to the next lateral branching event and the index of the hypha that branches
        """
        n = self.n_apex()

        self.__T1 = np.random.exponential(1/(n*self._b1)) #Draw of the waiting time

        id_apex = [i for i in range(len(self)) if self[i]._e] #List of indices of open hyphae   
        self.__id1 = np.random.choice(id_apex) #Uniform draw of the index of the branching hypha
        
    def lateral_branching(self):
        """
        Draws the waiting time to the next lateral branching event and the index of the hypha that branches
        """
        def F_inv(u): #Inverse of the cdf of the waiting time to the next lateral branching event
            L = self.total_length()
            A = self.n_apex()
        
            delta = (self._b2*L)**2 - 2*self._b2*A * np.log(1-u)
        
            return (-self._b2*L + np.sqrt(delta)) / (self._b2*A)
            
        self.__T2 = F_inv(np.random.uniform()) #Draw of the waiting time using inverse cdf method

        p = [self[i].length() + self[i]._e*self.__T2 for i in range(len(self))] #List of updated lengths of hyphae
        p = np.array(p) / sum(p) #List of probabilities of branching

        a = list(range(len(self))) #List of indices of all hyphae

        self.__id2 = np.random.choice(a, p=p) #Draw of the index of the branching hypha according to p
        
    def actualisation(self):
        """
        Updates the mycelium up to the next branching event
        """
        self.apical_branching()
        self.lateral_branching()
            
        T = min(self.__T1, self.__T2) #Waiting time to next branching event
        for i in range(len(self)):
            self[i].elongation(T) #Update of the lengths of the hyphae
    
        self._T += T #Update of the lifetime of the mycelium
    
        if self.__T1 < self.__T2: #Apical branching occurs first
            new1, new2 = self[self.__id1].apical_branching() #Hyphae created by apical branching
        else: #Lateral branching occurs first
            new1, new2 = self[self.__id2].lateral_branching() #Hyphae created by lateral branching
    
        self._hyphae.append(new1)
        self._hyphae.append(new2)

#=============================================================================#

def simulation(b1, b2, n_run, T_max, print_time = True):
    """
    Simulates the mycelium up to time T_max
    
    Args: 
        b1 (float): apical branching rate
        b2 (float): lateral branching rate
        n_run (int): number of simulations
        T_max (float): lifetime at the end of the simulation
        print_time (bool): print time of simulation
            
    Returns:
        data_open (list[float]): list of the lengths of open segments
        data_closed (list[float]): list of the lengths of closed segments
    """
    if print_time:
        print('start of simulation')
    
    data_open = []
    data_closed = []

    t_total = 0
    
    for i in range(n_run):  
        
        m = mycelium(b1, b2)
        t = m.run(T_max)
        
        data_open_ = m.length_open()
        if data_open_[-2] == 0: #remove the open segments of length 0 created after the last branching event
            data_open_ = data_open_[:-2]
        else:
            data_open_ = data_open_[:-1]
            
        data_open += [data_open_]
        data_closed += [m.length_closed()]
        
        t_total += t
    if print_time:
        print("time of sim = " + str(t_total) + 's')
        print("mean time = " + str(t_total/n_run) + 's')
    
    return(data_open, data_closed)

def simulation_to_n(b1, b2, n_run, n_max, print_time = True):
    """
    Simulates the mycelium until the number of hyphae is greater than n_max
    
    Args: 
        b1 (float): apical branching rate
        b2 (float): lateral branching rate
        n_run (int): number of simulations
        n_max (float): number of hyphae at the end of the simulation 
        print_time (bool): print time of simulation
            
    Returns:
        data_open (list[float]): list of the lengths of open segments
        data_closed (list[float]): list of the lengths of closed segments
    """
    
    if print_time:
        print('start of simulation')
    
    data_open = []
    data_closed = []

    t_total = 0
    
    for i in range(n_run): 
        
        m = mycelium(b1, b2)
        t = m.run_to_n(n_max)
        
        data_open_ = m.length_open()
        if data_open_[-2] == 0: #remove the open segments of length 0 created after the last branching event
            data_open_ = data_open_[:-2]
        else:
            data_open_ = data_open_[:-1]
        
        data_open += [data_open_]
        data_closed += [m.length_closed()]
        
        t_total += t
    
    if print_time:
        print("time of sim = " + str(t_total) + 's')
        print("mean time = " + str(t_total/n_run) + 's')
    
    return(data_open, data_closed)

