# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:59:15 2022

@author: Ahmad Al Musawi
"""
import networkx as nx

base_path = 'D:/Documents/Research Projects/Complex Networks Researches/Motif Based Link Prediction/Networks/'
pp = base_path + 'Networks'

def start():
    G = nx.DiGraph()
    G.add_edges_from([('c','b'),('b','c'),('a','b'),('b','a'),('c','a'),('a','c'),('c','d'),
                      ('a','d'),('d','e'),('f','d'),('d','f'),('f','e'),('e','f'),('f','g'),('g','f'),('f','h'),('h','f')])
    d = pattern_freq(G)
    return {i: d[i] for i in d if d[i]!=0}

def pattern_freq(G):
    print('Calculating Patterns...')
    patterns = [i for i in range(40)]
    F = {i: 0 for i in patterns}
    for x,y in G.edges():
        Nx = set(list(G.successors(x)) + list(G.predecessors(x)))
        Ny = set(list(G.successors(y)) + list(G.predecessors(y)))
        Nz = sorted(list(Nx.intersection(Ny)))
        for z in Nz:
            r = T2(G, x, y, z) 
            for i in r:
                F[i] = F[i] + 1
    print('Pattern calculation complete..')
    return F    

def TC(G, u, v, F):
    """Triadic closeness"""
    Nu = list(G.successors(u)) + list(G.predecessors(u))
    Nv = list(G.successors(v)) + list(G.predecessors(v))

    Nuv = sorted(list(set(Nu).intersection(set(Nv))))
    return sum([wP(G, u,v,z, F)  for z in Nuv])

def wP(G, u, v, z, F):
    if F[T(G, u, v, z)] != 0:
        return (F[T(G, u, v, z) + 10] + F[T(G, u, v, z) + 30])/F[T(G, u, v, z)]
    else:
        return 0

def T(G, u, v, z):
    """Pattern ID"""
    Patterns = []
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v)]):
        Patterns.append(1)
    if has_edges(G, [(u,z),(z,u),(z,v)]):
        Patterns.append( 2)
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        Patterns.append( 3)
    if has_edges(G, [(u,z),(z,v)]):
        Patterns.append( 4)
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        Patterns.append( 5)
    if has_edges(G, [(u,z),(v,z)]):
        Patterns.append( 6)
    if has_edges(G, [(z,u),(z,v),(v,z)]):
        Patterns.append( 7)
    if has_edges(G, [(v,z),(z,u)]):
        Patterns.append( 8)
    if has_edges(G, [(z,u),(z,v)]):
        Patterns.append( 9)

    if has_edges(G, [(u,z), (z,u),(v,z),(z,v),(u,v)]):
        Patterns.append( 11)
    if has_edges(G, [(u,z),(z,u),(z,v),(u,v)]):
        Patterns.append( 12)
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        Patterns.append( 13)
    if has_edges(G, [(u,z),(z,v),(u,v)]):
        Patterns.append( 14)
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        Patterns.append( 15)
    if has_edges(G, [(u,z),(v,z),(u,v)]):
        Patterns.append( 16)
    if has_edges(G, [(z,u),(z,v),(v,z),(u,v)]):
        Patterns.append( 17)
    if has_edges(G, [(v,z),(z,u),(u,v)]):
        Patterns.append( 18)
    if has_edges(G, [(z,u),(z,v),(u,v)]):
        Patterns.append( 19)

    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u)]):
        Patterns.append( 21)
    if has_edges(G, [(u,z),(z,u),(z,v),(v,u)]):
        Patterns.append( 22)
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        Patterns.append( 23)
    if has_edges(G, [(u,z),(z,v),(v,u)]):
        Patterns.append( 24)
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        Patterns.append( 25)
    if has_edges(G, [(u,z),(v,z),(v,u)]):
        Patterns.append( 26)
    if has_edges(G, [(z,u),(z,v),(v,z),(v,u)]):
        Patterns.append( 27)
    if has_edges(G, [(v,z),(z,u),(v,u)]):
        Patterns.append( 28)
    if has_edges(G, [(z,u),(z,v),(v,u)]):
        Patterns.append( 29)

    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u), (u,v)]):
        Patterns.append( 31)
    if has_edges(G, [(u,z),(z,u),(z,v),(v,u), (u,v)]):
        Patterns.append( 32)
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        Patterns.append( 33)
    if has_edges(G, [(u,z),(z,v),(v,u), (u,v)]):
        Patterns.append( 34)
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        Patterns.append( 35)
    if has_edges(G, [(u,z),(v,z),(v,u), (u,v)]):
        Patterns.append( 36)
    if has_edges(G, [(z,u),(z,v),(v,z),(v,u), (u,v)]):
        Patterns.append( 37)
    if has_edges(G, [(v,z),(z,u),(v,u), (u,v)]):
        Patterns.append( 38)
    if has_edges(G, [(z,u),(z,v),(v,u), (u,v)]):
        Patterns.append( 39)
    _ = input(f'{Patterns}')
    return Patterns

def T2(G, u, v, z):
    """Pattern ID"""
    Patterns = []
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u), (u,v)]):
        Patterns.append(  31)
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u)]):
        Patterns.append(  21)
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v),(u,v)]):
        Patterns.append(  11)
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v)]):
        Patterns.append(  1)

    if has_edges(G, [(u,z),(z,u),(z,v),(v,u), (u,v)]):
        Patterns.append(  32)
    if has_edges(G, [(u,z),(z,u),(z,v),(v,u)]):
        Patterns.append(  22)
    if has_edges(G, [(u,z),(z,u),(z,v),(u,v)]):
        Patterns.append(  12)
    if has_edges(G, [(u,z),(z,u),(z,v)]):
        Patterns.append(  2)

    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        Patterns.append(  33)
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        Patterns.append(  23)
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        Patterns.append(  13)
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        Patterns.append(  3)


    if has_edges(G, [(u,z),(z,v),(v,u), (u,v)]):
        Patterns.append(  34)
    if has_edges(G, [(u,z),(z,v),(v,u)]):
        Patterns.append(  24)
    if has_edges(G, [(u,z),(z,v),(u,v)]):
        Patterns.append(  14)
    if has_edges(G, [(u,z),(z,v)]):
        Patterns.append(  4)

    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        Patterns.append(  35)
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        Patterns.append(  25)
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        Patterns.append(  15)
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        Patterns.append(  5)

    if has_edges(G, [(u,z),(v,z),(v,u), (u,v)]):
        Patterns.append(  36)
    if has_edges(G, [(u,z),(v,z),(v,u)]):
        Patterns.append(  26)
    if has_edges(G, [(u,z),(v,z),(u,v)]):
        Patterns.append(  16)
    if has_edges(G, [(u,z),(v,z)]):
        Patterns.append(  6)

    if has_edges(G, [(z,u),(z,v),(v,z),(v,u), (u,v)]):
        Patterns.append(  37)
    if has_edges(G, [(z,u),(z,v),(v,z),(v,u)]):
        Patterns.append(  27)
    if has_edges(G, [(z,u),(z,v),(v,z),(u,v)]):
        Patterns.append(  17)
    if has_edges(G, [(z,u),(z,v),(v,z)]):
        Patterns.append(  7)

    if has_edges(G, [(v,z),(z,u),(v,u), (u,v)]):
        Patterns.append(  38)
    if has_edges(G, [(v,z),(z,u),(v,u)]):
        Patterns.append(  28)
    if has_edges(G, [(v,z),(z,u),(u,v)]):
        Patterns.append(  18)
    if has_edges(G, [(v,z),(z,u)]):
        Patterns.append(  8)

    if has_edges(G, [(z,u),(z,v),(v,u), (u,v)]):
        Patterns.append(  39)
    if has_edges(G, [(z,u),(z,v),(v,u)]):
        Patterns.append(  29)
    if has_edges(G, [(z,u),(z,v),(u,v)]):
        Patterns.append(  19)
    if has_edges(G, [(z,u),(z,v)]):
        Patterns.append(  9)
    return Patterns

def T1(G, u, v, z):
    """Pattern ID"""
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v)]):
        return 1
    if has_edges(G, [(u,z),(z,u),(z,v)]):
        return 2
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        return 3
    if has_edges(G, [(u,z),(z,v)]):
        return 4
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        return 5
    if has_edges(G, [(u,z),(v,z)]):
        return 6
    if has_edges(G, [(z,u),(z,v),(v,z)]):
        return 7
    if has_edges(G, [(v,z),(z,u)]):
        return 8
    if has_edges(G, [(z,u),(z,v)]):
        return 9

    if has_edges(G, [(u,z), (z,u),(v,z),(z,v),(u,v)]):
        return 11
    if has_edges(G, [(u,z),(z,u),(z,v),(u,v)]):
        return 12
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        return 13
    if has_edges(G, [(u,z),(z,v),(u,v)]):
        return 14
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        return 15
    if has_edges(G, [(u,z),(v,z),(u,v)]):
        return 16
    if has_edges(G, [(z,u),(z,v),(v,z),(u,v)]):
        return 17
    if has_edges(G, [(v,z),(z,u),(u,v)]):
        return 18
    if has_edges(G, [(z,u),(z,v),(u,v)]):
        return 19

    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u)]):
        return 21
    if has_edges(G, [(u,z),(z,u),(z,v),(v,u)]):
        return 22
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        return 23
    if has_edges(G, [(u,z),(z,v),(v,u)]):
        return 24
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        return 25
    if has_edges(G, [(u,z),(v,z),(v,u)]):
        return 26
    if has_edges(G, [(z,u),(z,v),(v,z),(v,u)]):
        return 27
    if has_edges(G, [(v,z),(z,u),(v,u)]):
        return 28
    if has_edges(G, [(z,u),(z,v),(v,u)]):
        return 29

    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u), (u,v)]):
        return 31
    if has_edges(G, [(u,z),(z,u),(z,v),(v,u), (u,v)]):
        return 32
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        return 33
    if has_edges(G, [(u,z),(z,v),(v,u), (u,v)]):
        return 34
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        return 35
    if has_edges(G, [(u,z),(v,z),(v,u), (u,v)]):
        return 36
    if has_edges(G, [(z,u),(z,v),(v,z),(v,u), (u,v)]):
        return 37
    if has_edges(G, [(v,z),(z,u),(v,u), (u,v)]):
        return 38
    if has_edges(G, [(z,u),(z,v),(v,u), (u,v)]):
        return 39
    else:
        return 0



def T0(G, u, v, z):
    """Pattern ID"""
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u), (u,v)]):
        return 31
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v), (v,u)]):
        return 21
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v),(u,v)]):
        return 11
    if has_edges(G, [(u,z), (z,u),(v,z),(z,v)]):
        return 1

    if has_edges(G, [(u,z),(z,u),(z,v),(v,u), (u,v)]):
        return 32
    if has_edges(G, [(u,z),(z,u),(z,v),(v,u)]):
        return 22
    if has_edges(G, [(u,z),(z,u),(z,v),(u,v)]):
        return 12
    if has_edges(G, [(u,z),(z,u),(z,v)]):
        return 2

    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        return 33
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        return 23
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        return 13
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        return 3


    if has_edges(G, [(u,z),(z,v),(v,u), (u,v)]):
        return 34
    if has_edges(G, [(u,z),(z,v),(v,u)]):
        return 24
    if has_edges(G, [(u,z),(z,v),(u,v)]):
        return 14
    if has_edges(G, [(u,z),(z,v)]):
        return 4

    if has_edges(G, [(u,z),(z,v),(v,z),(v,u), (u,v)]):
        return 35
    if has_edges(G, [(u,z),(z,v),(v,z),(v,u)]):
        return 25
    if has_edges(G, [(u,z),(z,v),(v,z),(u,v)]):
        return 15
    if has_edges(G, [(u,z),(z,v),(v,z)]):
        return 5

    if has_edges(G, [(u,z),(v,z),(v,u), (u,v)]):
        return 36
    if has_edges(G, [(u,z),(v,z),(v,u)]):
        return 26
    if has_edges(G, [(u,z),(v,z),(u,v)]):
        return 16
    if has_edges(G, [(u,z),(v,z)]):
        return 6

    if has_edges(G, [(z,u),(z,v),(v,z),(v,u), (u,v)]):
        return 37
    if has_edges(G, [(z,u),(z,v),(v,z),(v,u)]):
        return 27
    if has_edges(G, [(z,u),(z,v),(v,z),(u,v)]):
        return 17
    if has_edges(G, [(z,u),(z,v),(v,z)]):
        return 7

    if has_edges(G, [(v,z),(z,u),(v,u), (u,v)]):
        return 38
    if has_edges(G, [(v,z),(z,u),(v,u)]):
        return 28
    if has_edges(G, [(v,z),(z,u),(u,v)]):
        return 18
    if has_edges(G, [(v,z),(z,u)]):
        return 8

    if has_edges(G, [(z,u),(z,v),(v,u), (u,v)]):
        return 39
    if has_edges(G, [(z,u),(z,v),(v,u)]):
        return 29
    if has_edges(G, [(z,u),(z,v),(u,v)]):
        return 19
    if has_edges(G, [(z,u),(z,v)]):
        return 9
    else:
        return 0






def has_edges(G, E):
    for v, u in E:
        if not G.has_edge(v,u):
            return False
    return True


def neighbors(G, x):
    return sorted(list(G.successors(x)) + list(G.predecessors(x)))