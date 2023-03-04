# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:29:16 2022
@author: Ahmad Al Musawi
From Paper 1, we will use RA and AA (from Sec. 2.1.) and CNO, CNI and CNIO (from Sec. 2.4.). 
From Paper 2, we will use triadic closeness (Sec. 2.4.3). 
"""

"""U should be carefull passing the right neighbours
DiGraph.neighbors(n)
DiGraph.predecessors(n)

"""

import networkx as nx
import math
import pickle
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from itertools import combinations
from pandas import DataFrame
import matplotlib.pyplot as plt
import itertools

base_path = 'D:/Documents/Research Projects/Complex Networks Researches/Motif Based Link Prediction/Networks/'
pp = base_path + 'Networks'
# pp = project path
networks = ['Citation_', 'Ecoli', 'Email_', 'Human', 'Metabolic_', 'Mouse', 'Recommendation_', 'Twitter_', 'Wiki', 'Yeast']
cnetworks= ['citation' , 'biological', 'friendship','human', 'metabolic','mouse','reco', 'twitter','wiki','biological-2']
# LP_models = ['CNI', 'CNO', 'CNIO', 'AA', 'RA', 'TC', 'MS', 'MyMS']
LP_models = [ 'TC']

def delete_me():
    for M in LP_models:
        dd = []
        for i in networks:
            df = pd.read_csv(base_path + f'6548{i} results.csv')[M].mean()
            dd.append(df)
        for i in range(len(networks)):
            print(f'{M} {networks[i]} :\t {dd[i]}')
       


def start():
    title = input('Enter a distinguished name for storing...')
    for net in range(len(networks)):
        results = []        
        for i in range(3):
            gT, gP = nx.DiGraph(), nx.DiGraph()
            gT = pickle.load(open(f'{pp}/{networks[net]}/{cnetworks[net]}_train{i}.gml','rb'))
            gP = pickle.load(open(f'{pp}/{networks[net]}/{cnetworks[net]}_test{i}.gml','rb'))
            results.append(experiment(gT, gP))
            print(f'{cnetworks[net]}_test{i} ... is completed')
        df = DataFrame(results, columns=LP_models)
        df.to_csv(f'D:/Documents/Research Projects/Complex Networks Researches/Motif Based Link Prediction/Networks/{title}{networks[net]} results.csv')



def plotting():
    for i in networks:
        df = pd.read_csv(f'D:/Documents/Research Projects/Complex Networks Researches/Motif Based Link Prediction/Networks/total - results.csv')
        # plot_df(df, i)
        Plotting(df, i)

def plot_df(df, title):
    data = [(m, df[m].mean(), df[m].sem()) for m in LP_models]    
    data = sorted([[x[0], x[1], x[2]] for x in data] , key=lambda x: x[1], reverse=True)
    plotting_c(data,title, 1, 'Metrics', 'ROC AUC, SEM')
# ------------------------------------------------------------------------------
def plotting_c(data, title , st=1, xlabel='', ylabel='', path=''):
    diff =  [i[0] for i in data]
    aver  = [i[1] for i in data]
    stdev = [i[2] for i in data]
    #---------------------------------
    """Colored plotting"""
    # Create lists for the plot
    x_pos = np.arange(len(diff))
    fig, ax = plt.subplots()
    if st==1:
        ax.bar(x_pos, aver, yerr=stdev, align='center', alpha=0.5, ecolor='black', capsize=10)
    else:
        # removing error bar
        ax.bar(x_pos, aver,  align='center', alpha=0.5, ecolor='black', capsize=10)


    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(diff, fontsize=8)
    ax.set_title(title)
    ax.yaxis.grid(True)
    
    xlocs, xlabs = plt.xticks()
    plt.xticks(rotation=90)

    # Save the figure and show
    plt.tight_layout()
    print(path+title+'.png')
    plt.savefig(path+'.png', dpi = 300)
    plt.show()

def Plotting(df, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    width = 0.4

    ax.bar([i - width / 2 for i in range(len(LP_models))], df[LP_models].mean(), color = 'blue', width = width, alpha = 0.65)
    ax2.bar([i + width / 2 for i in range(len(LP_models))], df[LP_models].std(), color = 'red', width = width, alpha = 0.65)

    plt.xticks([i for i in range(len(LP_models))], LP_models, fontsize = 2)    
    ax.set_ylabel('Mean ROC AUC', fontsize = 12, color = 'blue')
    ax2.set_ylabel('Standard Deviation', fontsize = 12, color = 'red')
    ax.set_title(title)
    
    plt.tight_layout()
    
    plt.savefig(f'{title}.png', dpi = 300)
    plt.show()
    
def lines_plot2():
    df = pd.read_csv(base_path + 'Results 2.csv')
    fig, axe = plt.subplots(dpi=450)
    axe.plot(df[['FFL', 'MS2']], linewidth = 3)
    axe.plot(df[['MS', 'CNI', 'CNO', 'CNIO', 'AA', 'RA']], linewidth = 1)
    axe.legend(['FFL','MS','CNI', 'CNO', 'CNIO', 'AA', 'RA', 'MS2'], bbox_to_anchor = (1.05, 0.6))
    # df.plot(kind = 'line')
    plt.xlabel('Networks')
    plt.ylabel('AV AUC ROC')
    plt.xticks(rotation=90)
    ticks = df.index.tolist()
    plt.xticks(ticks, df.Networks)
    plt.tight_layout()
    plt.savefig(base_path+'Results 2.png', dpi = 450)
    plt.show()


def lines_plot():
    df = pd.read_csv(base_path + 'total - Results.csv')
    fig, axe = plt.subplots(dpi=450)
    axe.plot(df[['FFL']], linewidth = 3)
    axe.plot(df[['MS', 'CNI', 'CNO', 'CNIO', 'AA', 'RA', 'Deep']], linewidth = 0.91)
    axe.legend(['FFL','MS','CNI', 'CNO', 'CNIO', 'AA', 'RA', 'Deep'], bbox_to_anchor = (1.05, 0.6))
    # df.plot(kind = 'line')
    plt.xlabel('Networks')
    plt.ylabel('Mean AUC-ROC')
    plt.xticks(rotation=90)
    ticks = df.index.tolist()
    plt.xticks(ticks, df.Networks)
    plt.tight_layout()
    plt.savefig(base_path+'Results.png', dpi = 450)
    plt.show()






def experiment(gT, gP):
    """gT: train Graph, 
       gP: test Graph"""
    # ------------------------------------------------
    real_y = []
    S_E = []       
    if LP_models[0] == 'MS':
        F = motif_freq(gT)
    else:
        F = pattern_freq(gT)
    print(f'F = {F}')
    # ------------------------------------------------
    for u in gT.nodes():
        for v in gT.nodes():
            if (u,v) in gT.edges() or u==v:
                continue
            Nv_in, Nv_out = list(gT.neighbors(v)), list(gT.predecessors(v))
            Nu_in, Nu_out = list(gT.neighbors(u)), list(gT.predecessors(u))
            Suv = [LP(m, gT, u, v, Nu_in, Nu_out, Nv_in, Nv_out, F) for m in LP_models]
            S_E.append(Suv)
            if gP.has_edge(u,v):
                real_y.append(1)
            else:
                real_y.append(0) 
    newSxy = np.array(S_E)
    results1 = get_AUC(real_y, newSxy)
    return results1

def get_AUC(real_y, newSxy):
    results = []
    for c in newSxy.T:
        try:
            rr = metrics.roc_auc_score(real_y, c)
        except ValueError:
            rr = 0
        results.append(rr)
    return results


def find_auc(Gp, S):
    # Satyaki Code   
    # G is the test graph
    # S <- dictionary where S[(x, y)] is the score for link x -> y
    Y = []
    pred = []
    for key in S.keys():

        if key in Gp.edges():
            Y.append(1)
        else:
            Y.append(0)

        pred.append(S[key])

    return metrics.roc_auc_score(Y, pred)



def LP(a, gT, u, v, Nu_in, Nu_out, Nv_in, Nv_out, F):
    if a=="CNI":
        return CNI(gT, Nu_in, Nv_in)
    if a=="CNO":
        return CNO(gT, Nu_out, Nv_out)
    if a=="CNIO":
        return CNIO(gT, Nu_in + Nu_in, Nv_out + Nv_out)
    if a=="AA":
        return AA(gT,  Nu_in + Nu_in, Nv_out + Nv_out)
    if a=="RA":
        return RA(gT,  Nu_in + Nu_in, Nv_out + Nv_out)
    if a=='TC':
        return TC(gT, u, v, F)
    if a=='MS':
        return MS(gT, u, v, F)
    if a=='MyMS':
        return MyMS(gT, u, v)


def CNI(G,Nx,Ny):
    'common neighbor in'
    ax, ay = set(Nx), set(Ny)
    return  abs(len(ax.intersection(ay)))

def CNO(G,Nx,Ny):
    'common neighbor out'
    ax, ay = set(Nx), set(Ny)
    return  abs(len(ax.intersection(ay)))
    
def CNIO(G,Nx,Ny):
    'common neighbor in out'
    return  CNI(G, Nx, Ny) + CNO(G, Nx, Ny)

def AA(G,Nx,Ny):
    'Adamic-Adar index'
    ax, ay = set(Nx), set(Ny)
    az = ax.intersection(ay)
    sum = 0
    for z in az:
        L = math.log(len(list(G.neighbors(z))))
        # print (L)
        if L != 0 :
            sum = sum + (1/L)
    return sum 

def RA(G, Nx, Ny):
    ax, ay = set(Nx), set(Ny)
    sum = 0 
    for z in (ax.intersection(ay)):
        sum = sum + abs(1/len(list(G.neighbors(z))))
    return sum 



def TC(G, u, v, F):
    """Triadic closeness"""
    Nu = list(G.successors(u)) + list(G.predecessors(u))
    Nv = list(G.successors(v)) + list(G.predecessors(v))

    Nuv = sorted(list(set(Nu).intersection(set(Nv))))
    return sum([wP(G, u,v,z, F)  for z in Nuv])

def wP(G, u, v, z, F):
    # print(T(G, u, v, z))
    # print(F[T(G, u, v, z)])
    if F[T(G, u, v, z)] != 0:
        return (F[T(G, u, v, z) + 10] + F[T(G, u, v, z) + 30])/F[T(G, u, v, z)]
    else:
        return 0

def T(G, u, v, z):
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

def pattern_freq(G):
    print('Calculating Patterns...')
    patterns = [i for i in range(40)]
    F = {i: 0 for i in patterns}
    for x,y in G.edges():
        Nx = list(G.successors(x)) + list(G.predecessors(x))
        Ny = list(G.successors(y)) + list(G.predecessors(y))
        Nz = sorted(list(set(Nx).intersection(set(Ny))))
        for z in Nz:
            r = T(G, x, y, z) 
            F[r] = F[r] + 1
    print('Pattern calculation complete..')
    return F


# ---------------------------------------------------------------
def MS(G, u, v, F):
    """Motif Score"""
    Nu = list(G.successors(u)) + list(G.predecessors(u))
    Nv = list(G.successors(v)) + list(G.predecessors(v))

    Nuv = sorted(list(set(Nu).intersection(set(Nv))))
    if len(Nuv) != 0:
        return sum([F[mt(G, u, v, z)]/13 for z in Nuv])/len(Nuv)
    else:
        return 0
    
def motif_freq(G):
    print('Calculating Patterns...')
    patterns = [i for i in range(0,14)]
    F = {i: 0 for i in patterns}
    for x,y in G.edges():
        Nx = list(G.successors(x)) + list(G.predecessors(x))
        Ny = list(G.successors(y)) + list(G.predecessors(y))
        Nz = sorted(list(set(Nx).intersection(set(Ny))))
        for z in Nz:
            r = mt(G, x, y, z) 
            F[r] = F[r] + 1
    print('Pattern calculation complete..')
    return F
    
def mt(G, x, y, z):
    '''Motif type'''
    if has_edges(G, [(x,y), (y,x), (x,z),(z,x), (y,z),(z,y)]):
        return 13
    if has_edges(G, [ (y,x), (x,z),(z,x), (y,z),(z,y)]):
        return 12
    if has_edges(G, [ (y,x), (x,z),(z,x), (y,z)]):
        return 11
    if has_edges(G, [ (y,x), (z,x), (y,z),(z,y)]):
        return 10
    if has_edges(G, [ (y,x), (x,z),(z,x), (y,z)]):
        return 9
    if has_edges(G, [ (x,z),(z,x), (y,z),(z,y)]):
        return 8
    if has_edges(G, [ (y,x), (x,z), (z,y)]):
        return 7
    if has_edges(G, [ (y,x), (z,x), (z,y)]):
        return 6
    if has_edges(G, [ (z,x), (y,z),(z,y)]):
        return 5
    if has_edges(G, [ (x,z), (y,z),(z,y)]):
        return 4
    if has_edges(G, [ (y,x), (x,z)]):
        return 3
    if has_edges(G, [ (x,z), (y,z)]):
        return 2
    if has_edges(G, [(z,x), (z,y)]):
        return 1
    else :
        return 0

# -------------------------------------------------------------
def MyMS(G, u, v):
    """my Motif based similarity"""
    patterns = [i for i in range(0,14)]
    Fu = {i: 0 for i in patterns}
    Fv = {i: 0 for i in patterns}
    Nu = neighbors(G, u)
    Nv = neighbors(G, v)
    Nz = sorted(list(set(Nu).intersection(set(Nv))))

    for z in Nz:
        for a in neighbors(G, z):

            r = mt(G, u, z, a) 
            Fu[r] = Fu[r] + 1

            r = mt(G, v, z, a) 
            Fv[r] = Fv[r] + 1

    return sum([abs(Fu[i]-Fv[i]) for i in patterns])

def neighbors(G, x):
    return sorted(list(G.successors(x)) + list(G.predecessors(x)))