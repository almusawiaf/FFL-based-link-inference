import networkx as nx
import pickle
from pandas import DataFrame

networks = ['Citation_', 'Ecoli', 'Email_', 'Human', 'Metabolic_', 'Mouse', 'Recommendation_', 'Twitter_', 'Wiki', 'Yeast']
cnetworks= ['citation' , 'biological', 'friendship','human', 'metabolic','mouse','reco', 'twitter','wiki','biological-2']


def start(net):
    G = nx.DiGraph()
    for i in range(10):
        gT, gP = nx.DiGraph(), nx.DiGraph()
        gT = pickle.load(open(f'Networks/Networks/{networks[net]}/{cnetworks[net]}_train{i}.gml','rb'))
        gP = pickle.load(open(f'Networks/Networks/{networks[net]}/{cnetworks[net]}_test{i}.gml','rb'))
        
        G.add_nodes_from(gT.nodes())
        G.add_nodes_from(gP.nodes())        
        G.add_edges_from(gT.edges())
        G.add_edges_from(gP.edges())
        
    return G
        


def find_motifs(G):
    M = {}
    C = {u: 0 for u in G.nodes()}

    ind = len(G)

    for u in G.nodes():
        if G.out_degree(u) < 2:
            continue

        for v in G.nodes():
            if u == v or G.out_degree(v) < 1 or G.in_degree(v) < 1:
                continue

            for w in G.nodes():
                if w == u or w == v or G.in_degree(w) < 2:
                    continue

                if G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(u, w):
                    M[ind] = [u, v, w]
                    ind = ind + 1
                    C[u], C[v], C[w] = C[u] + 1, C[v] + 1, C[w] + 1

    return M, C


# GList = pickle.load(open('GList.p', 'rb'))
# results = {}
# for g in range(len(GList)):
#     G = GList[g]
#     _ , ffls = find_motifs(G)
#     results[g] = [len(G), len(G.edges()), sum(ffls.values())]


results = {}
for net in range(len(cnetworks)):
    G = start(net)
    print(cnetworks[net], len(G.nodes()), len(G.edges()))
    # _ , ffls = find_motifs(G)
    # results[cnetworks[net]] = [len(G.nodes()), len(G.edges()), sum(ffls.values())]
    results[cnetworks[net]] = [len(G.nodes()), len(G.edges())]
    print()

df = DataFrame(results)
df.to_csv('Networks/networks0 information.csv')
