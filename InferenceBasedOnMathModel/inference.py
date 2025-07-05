"""
This file contains functions used to extract data from a graph of a mycelial network 
"""

import numpy as np
import networkx as nx
from statistics import mean
from math import dist

def gph2xy(g):
    """
    Extract (x, y) coordinates for all nodes in a graph.

    Args:
        g (Any): a NetworkX graph

    Returns:
        x (float): x coordinates
        y (float): y coordinates
    """
    xa = nx.get_node_attributes(g, "x")
    x = [v for (k, v) in xa.items()]
    ya = nx.get_node_attributes(g, "y")
    y = [v for (k, v) in ya.items()]

    return x, y

def prune(g):
    """
    Prune a full-featured networkx graph comprised of nodes Vi, 1 <= i <= 3.

    Args:
        g (Any): a NetworkX graph

    Returns:
        The same networkx graph without node of degree 2
    """
    out = g.copy()
    idx = [n for n in out.nodes if len(list(out.neighbors(n))) == 2]

    for node in idx:
        inodes = list(out.neighbors(node))
        if len(inodes) > 1:  # fix edge case
            wgt = out[inodes[0]][node]["weight"]
            wgt += out[inodes[1]][node]["weight"]
            out.add_edge(inodes[0], inodes[1], weight=wgt)
            out.remove_node(node)

    return out

def nx_dist(g, euclidean=True, min_diameter=0, plot=False, verbose=False):
    """
    Compute the euclidean or curvilinear distance between apex and the closest
    node of degree 3.

    Args:
        g (Any): a NetworkX graph
        idx (int): frame number
        euclidean (bool): euclidean (default) or curvilinear distance
        r1 (int): inward radius (distance wrt the center of the graph) crown
        r2 (int): outward radius (distance wrt the center of the graph) crown
        inward (bool): candidate is located to the near center
        min_dist (int): minimum distance to consider
        min_diameter (int): minimum distance of apex to center
        plot (bool): show graphical output (mostly for debugging purpose)
        verbose (bool): verbose mode; return coordinates of node and distance

    Returns:
        A list[float] of all V1-V3 distances
    """
    gp = prune(g)
    apex = [n for n, d in nx.degree(gp) if d == 1]
    xs, ys = gph2xy(gp)
    cx, cy = mean(xs), mean(ys)

    out = []
    apexes = []

    for e in apex:
        a, b = [gp.nodes[e]["x"], gp.nodes[e]["y"]], [cx, cy]
        apexes.append(a)
        diameter = dist(a, b)
        if diameter > min_diameter:
            nearest = next(nx.neighbors(gp, e))

            car, cdr = e, nearest

            if euclidean:
                a = [gp.nodes[car]["x"], gp.nodes[car]["y"]]
                b = [gp.nodes[cdr]["x"], gp.nodes[cdr]["y"]]
                d = dist(a, b)
            else:
                d = gp[car][cdr]["weight"]
            if verbose:
                out.append((a, b, d))
            else:
                out.append(d)

    if plot:
        import matplotlib.pyplot as plt
        plt.axis("off")
        plt.axis("square")
        plt.axis([cx-2800,cx+2800,cy-2800,cy+2800])
        xo, yo = gph2xy(g)
        x_, y_ = [apexes[i][0] for i in range(len(apexes))],[apexes[i][1] for i in range(len(apexes))]
        plt.scatter(xo, yo, s=1, c="0.7", linewidth=0)
        plt.scatter(x_, y_, s=2, c="cornflowerblue", linewidth=0)
        if verbose:
            for i in range(len(out)):
                xso, xt = out[i][0][0], out[i][1][0]
                yso, yt = out[i][0][1], out[i][1][1]
                plt.scatter(xt, yt, s=1, c="tomato", linewidth=0)
                
                plt.plot([xso, xt], [yso, yt], c="tomato", linestyle="-", linewidth=0.5)
        plt.show()

    return out

def nx_dist_closed(g, min_diameter=None):
    """
    Compute the euclidean distance between the nodes of degree 3

    Args:
        g: a NetworkX graph
        min_diameter (int): minimum distance of node to center

    Returns:
        A list[float] of all V3-V3 distances
    """
    gp = prune(g)
    xs, ys = gph2xy(gp)
    cx, cy = mean(xs), mean(ys)
    
    out = []
    
    for edge in gp.edges():
        e0 = edge[0]
        e1 = edge[1]
        if gp.degree[e0]==3 and gp.degree[e1]==3:
            a, b, c = [gp.nodes[e0]["x"], gp.nodes[e0]["y"]], [gp.nodes[e1]["x"], gp.nodes[e1]["y"]], [cx, cy]
            da = dist(a,c)
            db = dist(b,c)
            if da > min_diameter or db > min_diameter:
                out.append(gp[e0][e1]['weight'])
    return out

def l_tot(g, n=1, k=0, min_diameter=0):
    """
    Compute the length of the mycelium in an
    
    Args:
        g: a NetworkX graph
        n (int): number of sectors 
        k (int): 
        min_diameter (int): minimum distance of node to center
    
    Returns:
        l (float): total length

    """
    gp = prune(g)
    xs, ys = gph2xy(gp)
    cx, cy = mean(xs), mean(ys)
    
    l = 0
    
    for (a,b) in gp.edges():
        da = dist([gp.nodes()[a]['x'],gp.nodes()[a]['y']],[cx,cy])
        db = dist([gp.nodes()[b]['x'],gp.nodes()[b]['y']],[cx,cy])
        
        ta = np.arctan2(gp.nodes()[a]['y']-cy,gp.nodes()[a]['x']-cx)
        if ta < 0:
            ta += 2*np.pi  
        tb = np.arctan2(gp.nodes()[b]['y']-cy,gp.nodes()[b]['x']-cx)
        if tb < 0:
            tb += 2*np.pi
        
        if da > min_diameter or db > min_diameter:
            if (ta >= 2*np.pi*k/n and ta < 2*np.pi*(k+1)/n) or (tb >= 2*np.pi*k/n and tb < 2*np.pi*(k+1)/n):
                l += gp[a][b]['weight']

    return l/2 #each segment corresponds to two edges
    
def nb_apex(g, n=1, k=0, min_diameter=0):
    """
    Compute the number of apexes in an
    
    Args:
        g: a NetworkX graph
        n (int): number of sectors 
        k (int): 
        min_diameter (int): minimum distance of node to center

    Returns:
        nb (int): number of apexes
    """
    gp = prune(g)
    apex = [n for n, d in nx.degree(gp) if d == 1]
    xs, ys = gph2xy(gp)
    cx, cy = mean(xs), mean(ys)
    
    nb = 0
    
    for e in apex:
        a, b = [gp.nodes[e]["x"], gp.nodes[e]["y"]], [cx, cy]
        diameter = dist(a, b)
        theta = np.arctan2(a[1]-b[1], a[0]-b[0])
        if theta < 0:
            theta += 2*np.pi
        if diameter > min_diameter:
            if theta >= 2*np.pi*k/n and theta < 2*np.pi*(k+1)/n:
                nb+=1
    return nb

def diam(g):
    """
    Compute furthest euclidian distance from the centre
    
    Args:
        g (Any): a NetworkX graph
    """
    gp = prune(g)
    xs, ys = gph2xy(gp)
    cx, cy = mean(xs), mean(ys)
    d = []
    for i in range(len(xs)):
        d.append(dist([xs[i], ys[i]], [cx, cy]))
    return max(d)

