# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 12:04:22 2022

@author: Ahmad Al Musawi
"""
import networkx as nx
import math

def DAGI(G, x, y):
    """Global Influence: abstract value of the difference in influence between x, y in comparison to network average degree"""
    deg = sum(d for _, d in G.degree()) / float(G.number_of_nodes())   
    tu = len(x)/deg
    tv = len(y)/deg
    potential  = abs(tu-tv)
    return potential

def DALI(G, x, y):
    """Local Influence: abstract value of the difference in influence between x, y in comparison to nodes' neighbor average degree"""
    ax = set(x)
    ay = set(y)
    if len(ax)!= 0:
        sum1 = sum([G.degree(i) for i in ax])/len(ax)
        tu = len(x)/(sum1)
    else:
        tu = 0
    if len(ay)!= 0 :
        sum2 = sum([G.degree(i) for i in ay])/len(ay)
        tv = len(y)/(sum2)
    else:
        tv = 0

    potential  = abs(tu - tv)
    return potential


def inDAGI(G, x, y):
    """Global Influence: abstract value of the difference in influence between x, y in comparison to network average degree"""
    deg = sum(d for _, d in G.degree()) / float(G.number_of_nodes())   
    tu = len(x)/deg
    tv = len(y)/deg
    potential  = abs(tu-tv)
    if potential==0:
        return 1
    else:
        return (1/potential)

def inDALI(G, x, y):
    """Local Influence: abstract value of the difference in influence between x, y in comparison to nodes' neighbor average degree"""
    ax = set(x)
    ay = set(y)
    if len(ax)!= 0:
        sum1 = sum([G.degree(i) for i in ax])/len(ax)
        tu = len(x)/(sum1)
    else:
        tu = 0
    if len(ay)!= 0 :
        sum2 = sum([G.degree(i) for i in ay])/len(ay)
        tv = len(y)/(sum2)
    else:
        tv = 0
    potential  = abs(tu - tv)
    if potential==0:
        return 1
    else:
        return (1/potential)



def PAGI(G, x, y):
    """Pereferential Attachment based abstract value of the difference in influence between x, y 
    in comparison to network average degree"""
    deg = sum(d for n, d in G.degree()) / float(G.number_of_nodes())   
    tu = len(x)/deg
    tv = len(y)/deg
    potential  = tu * tv
    return (potential)

def PALI(G, x, y):
    """Pereferential Attachment based abstract value of the difference in influence between x, y 
    in comparison to nodes' neighbor average degree"""
   
    ax = set(x)
    ay = set(y)
    if len(ax)!= 0:
        sum1 = sum([G.degree(i) for i in ax])/len(ax)
        tu = len(x)/(sum1)
    else:
        tu = 0
    if len(ay)!= 0 :
        sum2 = sum([G.degree(i) for i in ay])/len(ay)
        tv = len(y)/(sum2)
    else:
        tv = 0
    potential  = tu * tv
    return (potential)


    

def distance(G, s, t):
    try:
        dd =  nx.shortest_path_length(G, s, t)
        if dd > 0 :
            return len(G.nodes())/nx.shortest_path_length(G, s, t)
        else:
            return 0
    except (nx.NetworkXNoPath, ValueError):
        return 0

def CN(G, x, y):
    'common neighbor'
    ax = set(x)
    ay = set(y)
    return  abs(len(ax.intersection(ay)))

def AA(G,x,y):
    'Adamic-Adar index'
    ax = set(x)
    ay = set(y)
    az = ax.intersection(ay)
    sum = 0
    for z in az:
        L = math.log(len(list(G.neighbors(z))))
        # print (L)
        if L != 0 :
            sum = sum + (1/L)
    return sum 

def RA(G, x, y):
    ax = set(x)
    ay = set(y)
    sum = 0 
    for z in (ax.intersection(ay)):
        sum = sum + abs(1/len(list(G.neighbors(z))))
    return sum 

    
def PA(G, x, y):
    'Preferential Attachment'
    ax = set(x)
    ay = set(y)
    return  len(ax)*len(ay)

def JA(G, x, y):
    'Jaccard Index'
    ax = set(x)
    ay = set(y)
    if len(ax.union(ay))!=0 :
        return  len(ax.intersection(ay))/len(ax.union(ay))
    else:
        return 0

def SA(G, x, y):
    'Salton Index'
    ax = set(x)
    ay = set(y)
    if len(ax)!=0 and len(ay)!=0:
        return len(ax.intersection(ay))/math.sqrt(len(ax)*len(ay))
    else:
        return 0

def SO(G, x, y):
    'Sorensen Index'
    ax = set(x)
    ay = set(y)
    if (len(ax)+len(ay))!=0:
        return  2* len(ax.intersection(ay))/(len(ax)+len(ay))
    else:
        return 0

def HPI(G, x, y):
    'Hub Pronoted Index'
    ax = set(x)
    ay = set(y)
    if len(ax)!=0 and len(ay)!=0:
        return  len(ax.intersection(ay))/min(len(ax), len(ay))
    else:
        return 0

def HDI(G, x, y):
    'Hub Depressed Index'
    ax = set(x)
    ay = set(y)
    if max(len(ax), len(ay))!=0:
        return  len(ax.intersection(ay))/max(len(ax), len(ay))
    else:
        return 0
    
def LLHN(G, x, y):
    'Local Leicht-Homle-Newman Index'
    ax = set(x)
    ay = set(y)
    if len(ax)!=0 and len(ay)!=0:
        return  len(ax.intersection(ay))/len(ax)*len(ay)
    else:
        return 0

def CAR(G, x, y):
    ax = set(x)
    ay = set(y)
    sum = 0 
    for z in (ax.intersection(ay)):
        az = G.neighbors(z)
        if len(list(az)) != 0:
            dom = len(ax.intersection(ay.intersection(set(G.neighbors(z)))))
            nom = len(list(G.neighbors(z)))
            sum = sum + (dom/nom)
    return sum
