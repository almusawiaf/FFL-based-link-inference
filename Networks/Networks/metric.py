import networkx as nx
import pickle
import random
import numpy as np
from sklearn import metrics
from triad import *


def find_auc(G, S):

    # G is the test graph
    # S <- dictionary where S[(x, y)] is the score for link x -> y
    Y = []
    pred = []
    for key in S.keys():

        if key in G.edges():
            Y.append(1)
        else:
            Y.append(0)

        pred.append(S[key])

    return metrics.roc_auc_score(Y, pred)
