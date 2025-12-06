# Bonus 2: RTree

# Autoři: Fuchsig, Kohout

# Prohlášení:
# Kód byl z většiny vypracován samostatně. Nejistoty či jiné problémy
# byly konzultovány s umělou inteligencí (Google Gemini). To se vztahuje
# na opravy syntaxe, ladění chyb či hrubý návrh postupu práce.
# AI byla využita jako nástroj k pochopení problematiky a konzultování myšlenek.

# Documentation used: https://rtree.readthedocs.io/en/latest/tutorial.html

from math import *
from time import *
from matplotlib.pyplot import *
from numpy import *


def loadPoints(file):
    #Load file
    X, Y, Z = [], [], [] ;
    
    with open(file) as f:
        #Process lines
        for line in f:
            #Split line
            x, y, z = line.split('\t')
            
            #Add coordinates
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
    
    return X, Y, Z


def drawPoints(X, Y, Z, bx, transp = 0.2):
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal')

    #Compute sphere scale: 1 pix = 25.4 mm
    scale = 1
    if bx > 0:
        scale = int(bx * bx * 40 * 40)
        
    #Plot points
    ax.scatter(X, Y, Z, s=scale, alpha = transp)

    show()


#Load points
X, Y, Z = loadPoints('tree_18.txt')

#Draw points
drawPoints(X, Y, Z, 0, 0.2)


# Initialization
n = len(X)
print("Number of points:", n)


# Setup R-tree for 3D
# By default, rtree works in 2D. We must enable 3D manually.
# Source: https://rtree.readthedocs.io/en/latest/class.html#rtree.index.Property.dimension
from rtree import index 
from time import process_time

p = index.Property()
p.dimension = 3
idx = index.Index(properties=p)

t_start_rtree = process_time()

# Build Tree (Insert points)
# The library is designed for rectangles (bounding boxes), not points.
# To insert a point, we must create a "box" with zero size (min_coords = max_coords).
# Syntax: insert(id, (left, bottom, right, top)) -> for 3D we duplicate coords
# Source: https://rtree.readthedocs.io/en/latest/tutorial.html#inserting-data
for i in range(n):
    idx.insert(i, (X[i], Y[i], Z[i], X[i], Y[i], Z[i]))

# Search NN
sum_dist_rtree = 0
points_processed_rtree = 0

for i in range(n):
    # We query using a box that represents our point
    query_box = (X[i], Y[i], Z[i], X[i], Y[i], Z[i])
    
    # We ask for 2 nearest items because the 1st one is the point itself (dist 0)
    #Source: https://rtree.readthedocs.io/en/latest/tutorial.html#querying
    nearest_ids = list(idx.nearest(query_box, 2))
    
    d_min = 1000000000.0
    
    for cand_id in nearest_ids:
        # Library returns IDs, we must calculate distance manually
        d = ((X[i]-X[cand_id])**2 + (Y[i]-Y[cand_id])**2 + (Z[i]-Z[cand_id])**2)**0.5
        
        # Check: Is closer AND is not zero (ignore duplicates)
        if d < d_min and d > 0:
            d_min = d
            
    # Check if found
    if d_min < 1000000000.0:
        sum_dist_rtree += d_min
        points_processed_rtree += 1

# Calculate Density
if points_processed_rtree > 0:
    d_aver_rtree = sum_dist_rtree / points_processed_rtree
    rho_rtree = 1 / (d_aver_rtree**3)
else:
    rho_rtree = 0

t_end_rtree = process_time()

print("Average distance (R-tree):", d_aver_rtree)
print("Density (R-tree):", rho_rtree)
print("Time elapsed (R-tree):", t_end_rtree - t_start_rtree, "s")