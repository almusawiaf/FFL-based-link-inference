import pickle
import random
import networkx as nx
from pandas import DataFrame


import Motif_Based_Link_Prediction as MBLP

import networkx as nx
import pickle
import random
import numpy as np

from scipy import spatial
from sklearn import metrics
from triad import *
from karateclub import DeepWalk

def gen(G, si):

    H = G.to_undirected()
    dsum = float(sum([H.degree(u) for u in H.nodes()]))

    # New subgraph
    g = nx.DiGraph()

    # Pick random seed as a starting node
    ra = np.random.choice(G.nodes(), 10, p = [float(H.degree(u))/float(dsum) for u in G.nodes()])
    r = random.choice(ra)
    # print (ra, r)

    g.add_node(r)

    while len(g) < si:
        nset = []
        # List of all neighbors of nodes in g
        for u in g.nodes():
            l = H.neighbors(u)
            nset.extend(l)

        # The nodes in 'nset' are not already present in g
        nset = [u for u in nset if u not in g.nodes()]

        if len(nset) > 0:
            r = random.choice(nset)
        else:
            r = random.choice(list(G.nodes()))

        N = list(g.nodes())
        for u in N:
            if G.has_edge(u, r):
                g.add_edge(u, r)

            if G.has_edge(r, u):
                g.add_edge(r, u)

    h = g.to_undirected()
    if nx.number_connected_components(h) > 1:
        print ("ALARM---")

    return g


def find_auc(G, S):

    # S <- dictionary
    # S[(x, y)] <- link prediction score for x -> y
    Y = []
    pred = []
    for key in S.keys():

        if key in G.edges():
            Y.append(1)
        else:
            Y.append(0)

        pred.append(S[key])

    return metrics.roc_auc_score(Y, pred)


def find_acc(G, S):

    S = sorted(S.items(), key = lambda x: x[1], reverse = True)
    # print (S)
    # input('')

    V = 0
    for i in range(len(S)):
        e = (int(S[i][0][0]), int(S[i][0][1]))

        if i < len(G.edges()):
            if e in G.edges():
                V = V + 1
        else:
            if e not in G.edges():
                V = V + 1

    return float(V) / float(len(S))


def sample(G, perc):

    E = list(G.edges())

    Et, Ep = [], []
    for i in range(len(E)):
        if random.uniform(0, 1) < perc:
            Et.append(E[i])
        else:
            Ep.append(E[i])

    Gt = create_graph(G, Et)
    Gp = create_graph(G, Ep)
    return Gt, Gp


def create_graph(G, E):
    H = nx.DiGraph()
    H.add_nodes_from(list(G.nodes()))
    H.add_edges_from(E)
    return H


def walks(G, wl = 10, wn = 50, dim = 16, ws = 5):
    print (list(G.nodes()))

    model = DeepWalk(walk_length = wl, walk_number = wn, dimensions = dim, window_size = ws)
    model.fit(G)

    embedding = model.get_embedding()
    return embedding


def cosine(dataSetI, dataSetII):
    result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
    return result




def create_graph(G, E):
    H = nx.DiGraph()
    H.add_nodes_from(list(G.nodes()))
    H.add_edges_from(E)
    return H


def sample(G, perc):

    E = list(G.edges())
    Et, Ep = [], []

    for i in range(len(E)):

        if random.uniform(0, 1) < perc:
            Et.append(E[i])
        else:
            Ep.append(E[i])

    Gt = create_graph(G, Et)
    Gp = create_graph(G, Ep)
    return Gt, Gp


GList = pickle.load(open('GList.p', 'rb'))
repeat = 10
title = input('Enter a distinguished name for storing...')

results = []
#-----------------------------------------------------------------------------
#                             OLD Implementation
#-----------------------------------------------------------------------------
# for G in GList:
#     A1, A2, A3, A4 = [], [], [], []
#     for r in range(repeat):
#         Gt, Gp = sample(G, 0.9)
#         print (len(Gt), len(Gt.edges()), len(Gp), len(Gp.edges()))
# #         results.append(MBLP.experiment(Gt, Gp))

# # df = DataFrame(results, columns=MBLP.LP_models)
# # df.to_csv(f'Networks/{title} results.csv')
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#                             DeepWalk Implementation
#-----------------------------------------------------------------------------
A1, A2, A3, A4 = [], [], [], []
kappa = 0.001
k = 10.0

for G in GList:
    G = nx.convert_node_labels_to_integers(G, first_label = 0)

    Gt, Gp = sample(G, 0.9)
    print (len(Gt), len(Gp), len(Gt.edges()), len(Gp.edges()))

    S2 = {}
    I = Gt.to_undirected()
    embeds = walks(I)
    nL = [(u, v) for u in Gp.nodes() for v in Gp.nodes() if u != v and (u, v) not in Gt.edges()]

    for (u, v) in nL:
        S2[(u, v)] = cosine(embeds[u], embeds[v])

    a2 = find_auc(Gp, S2)
    A2.append(a2)
    print (np.mean(A2), np.std(A2), '\n')

df = DataFrame(A2)
df.to_csv(f'Networks/{title} results.csv')
