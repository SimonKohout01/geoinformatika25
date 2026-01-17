from collections import *
from queue import *
from numpy import *
import math
  
def loadEdges(file_name):
    #Convert list of lines to the graph
    PS = []
    PE = []
    W = []
    with open(file_name,'r', encoding='utf-8-sig') as f:
        for line in f:
            #Split
            x1, y1, x2, y2, w = line.split()
            
            #Add start, end points and weights to the list
            PS.append((float(x1), float(y1)))
            PE.append((float(x2), float(y2)))
            W.append(float(w))
    return PS, PE, W

def pointsToIDs(P):
    #Create a map: key = coordinates, value = id
    D = {}
    for i in range(len(P)):
        D[(P[i][0], P[i][1])] = i
        
    return D

def edgesToGraph(D, PS, PE, W):
    #Convert edges to undirected graph
    G = defaultdict(dict)

    #Add weight to the edge
    for i in range(len(PS)):
        G[D[PS[i]]][D[PE[i]]] = W[i]
        G[D[PE[i]]][D[PS[i]]] = W[i]

    return G

# Added a function for matching created nods with corresponding city
# We can then select a city for the graph algorithms
# AI was used here for syntax error and a help what should be in the function
def cities(load_cities, D):
    city_to_id = {}

    node_list = []
    for coord, node_id in D.items():
        node_list.append((coord[0], coord[1], node_id))
    
    with open(load_cities, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
                
            name = parts[0]
            x = float(parts[1])
            y = float(parts[2])
                
            min_dist = float('inf')
            nearest_id = None
            
            # Calculating difference between all nodes, saving the lowest
            for ux, uy, uid in node_list:
                dist = math.sqrt((x - ux)**2 + (y - uy)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = uid
            
            # Applying the tolerance (1 m) for the coordinates, then assigns city to the node
            if nearest_id is not None and min_dist < 1.0:
                city_to_id[name] = nearest_id
            else:
                print(f"City '{name}' does not exist in a node .")
    
    return city_to_id

# This is a test for this skript
if __name__ == "__main__":
# Load edges
# Choose what calculation you want to use
    # file = 'silnice_usti_vzdal.txt'                 # For calculating Euclid distance 
    # file = 'silnice_usti_cas.txt'                 # For calculating fastest time without any consideration of curvature of the roads
    file = 'silnice_usti_cas_klikatost.txt'       # For calculating fastest time considering curvature of the roads -> slowing down

    PS, PE, W = loadEdges(file)

#Merge lists and remove unique points
    PSE = PS + PE
    PSE=unique(PSE,axis=0).tolist()
    PSE.insert(0, [1000000, 1000000])

#Edges to graph
    D = pointsToIDs(PSE)
    G = edgesToGraph(D, PS, PE, W)
    print(G)

# Loading data

    load_cities = 'mesta_usti.txt'

    list_cities = cities(load_cities, D)

# Listing the cities with corresponding ID
 
    print(list_cities)

