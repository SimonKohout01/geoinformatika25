# Method 3: KD-Tree

# Autoři: Fuchsig, Kohout

# Prohlášení:
# Kód byl z většiny vypracován samostatně. Nejistoty či jiné postupy
# byly konzultovány s umělou inteligencí (Google Gemini). To se vztahuje
# na opravy syntaxe, ladění chyb, hrubý návrh postupu práce či implementace výpočtu PCA.
# AI byla využita jako nástroj k pochopení problematiky a konzultování myšlenek.


from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 

def loadPoints(file):
    #Load file
    X, Y, Z = [], [], [] 
    
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


class KDNode:
    def __init__(self, point, axis, left, right, point_id):
        self.point = point       # [x, y, z]
        self.axis = axis         # 0=x 1=y 2=z
        self.left = left         # Left wing
        self.right = right       #Right wing
        self.point_id = point_id # Original index

class KDTree:
    def __init__(self, X, Y, Z):
        # Prepare list of points with index
        # We need original index to know whch point is which
        self.node_list = []
        for i in range(len(X)):
            self.node_list.append( [[X[i],Y[i], Z[i]], i] )
            
        # Start building the tree
        self.root = self.build_tree(self.node_list, 0)

    def build_tree(self, points, depth):
        # Base case: no points left
        if not points:
            return None

        # Axis changes with depth
        axis = depth%3

        # Sort points by coordinate (x, y or z)
        #Lambda is used for sorting key
        points.sort(key=lambda p: p[0][axis])

        # Find middle indx - median
        median_idx = len(points) // 2
        median_val = points[median_idx]

        # Recursive build
        return KDNode(
            point=median_val[0],
            axis=axis,
            left=self.build_tree(points[:median_idx], depth + 1),
            right=self.build_tree(points[median_idx+1:], depth + 1),
            point_id=median_val[1])
        
    # AI used here for suggesting and explaining the workflow
    def get_nn(self, query_point, node=None, best_node=None, best_dist=1000000000.0):
        # Start at start
        if node is None:
            node = self.root
            
        #Calculate distance to current node
        dx = node.point[0] - query_point[0]
        dy = node.point[1] - query_point[1]
        dz = node.point[2] - query_point[2]
        dist = (dx*dx + dy*dy +dz*dz)**(1/2)

        # Update best
        if dist <best_dist and dist > 0:
            best_dist = dist
            best_node = node

        # If the result is negative, point is on the left side
        axis = node.axis
        diff = query_point[axis] - node.point[axis]

        if diff < 0:
            near = node.left
            far = node.right
        else:
            near = node.right
            far = node.left

        # Search near side
        if near:
            # Going deeper into the tree
            candidate_node, candidate_dist = self.get_nn(query_point, near, best_node, best_dist)
            if candidate_dist < best_dist:
                best_dist = candidate_dist
                best_node = candidate_node

        # Search far side same as in near side 
        if far and abs(diff) < best_dist:
            # We only look there if the distance to the splitting axis is smaller than our current best distance
            candidate_node, candidate_dist = self.get_nn(query_point, far, best_node, best_dist)
            if candidate_dist < best_dist:
                best_dist = candidate_dist
                best_node = candidate_node

        return best_node, best_dist
    
    # Fucntion for searching k NN for aproximating curvature in each point
    # AI used here for suggesting and explaining the workflow and debugging
    
    def get_knn(self, query_point, k, node=None, best_list=None):
        if node is None:
            node = self.root
        if best_list is None:
            best_list = []

        # Calculate distance to current node
        dx = node.point[0] - query_point[0]
        dy = node.point[1] - query_point[1]
        dz = node.point[2] - query_point[2]
        dist = (dx*dx + dy*dy + dz*dz)**0.5
        
        # Add to list
        best_list.append((dist, node.point_id))
        
        # Sort and get best 30
        best_list.sort(key=lambda x: x[0]) 
        if len(best_list) > (k + 1):
            best_list.pop() 
            
        if len(best_list) == (k + 1):
            worst_dist = best_list[-1][0]
        else:
            worst_dist = float('inf')

        # If the result is negative, point is on the left side
        axis = node.axis
        diff = query_point[axis] - node.point[axis]

        if diff < 0:
            near = node.left
            far = node.right
        else:
            near = node.right
            far = node.left
        # Search near side
        if near:
            self.get_knn(query_point, k, near, best_list)
            if len(best_list) == (k + 1):
                worst_dist = best_list[-1][0]

        if far and abs(diff) < worst_dist:
            self.get_knn(query_point, k, far, best_list)

        return best_list
    
    
#Compute density - Method 3: KD-Tree

print("Computing density and curvature of points with KDTree Method")

t_start_kd = process_time()

# Building Tree
tree = KDTree(X, Y, Z)

# Search for NN
sum_dist_kd = 0
points_processed_kd = 0


for i in range(n):
    query = [X[i], Y[i], Z[i]]
    
    # Get NN from tree
    nn_node, dist = tree.get_nn(query)
    
    # Check if valid
    if dist < 1000000000.0:
        sum_dist_kd += dist
        points_processed_kd += 1

# Average
if points_processed_kd > 0:
    d_aver_kd = sum_dist_kd / points_processed_kd
    rho_kd = 1 / (d_aver_kd**3)
else:
    rho_kd = 0
    

# Search curvature in each point for 30 NN
curvature = []
knn = 30

# AI used here for debugging and correcting syntax
# and later helping with usage of PCA

for i in range(len(X)):
    # searching NN, same as before
    query = [X[i], Y[i], Z[i]]
    
    knn_node = tree.get_knn(query, knn)
    
    # List of knn
    knn_list = []
    
    for dist, idx in knn_node:
        # KD Tree finds the closest point as the point itself, the distance is 0, it skips this point
        if idx != i:
            point = [X[idx], Y[idx], Z[idx]]        # Creates coordinates of the point
            knn_list.append(point)
            
    # Edge of point cloud -> low number of neighbours, then just add 0
    if len(knn_list) < 3:
        curvature.append(0)
        continue

    
    # Calculating the covariance     
    knn_matrix = cov(knn_list, rowvar=False)
    
    # From the calculation we get eigenvalues and vectors, whic we won't be using
    eigenval, vectors = linalg.eig(knn_matrix)
    
    # Sorting eigenvalues, the lowest will be L1
    lambda_c = sorted(abs(eigenval))
    
    # Calculating kappa from given equation of curvature
    l1 = lambda_c[0] 
    l2 = lambda_c[1]
    l3 = lambda_c[2]
    
    sum_l = l1 + l2 + l3
    
    if sum_l == 0:          # 0 means flat plane (light color)
        kappa = 0
    else:
        kappa = l1 / sum_l  # >0 means terrain or noise (darker color)
        
    curvature.append(kappa)
    

# End time of calculation
t_end_kd = process_time()

# CHARACTERISTICS - DESNSITY

print("Average distance:", d_aver_kd)
print("Density:", rho_kd)
print("Time elapsed:", t_end_kd - t_start_kd, "s") 

# CHARACTERISTICS - APPROXIMATED CURVATURE FOR EACH POINT

print("Generating visualisation of point cloud where color of each point is determined by its approximated curvature.")

# Visualisation
fig = figure()
ax = axes(projection='3d')

vis = ax.scatter(X, Y, Z, c=curvature, cmap='magma_r', s=1)

fig.colorbar(vis, label='Approximated Curvature')
show()

# CREATE PLOT OF CALCULATION TIME ON NUMBER OF POINTS

print("Generating the plot of calculation time on number of points.")

# List of test number of points (50 to all points)
test = [50,100,250,500,1000,2500,5000,7500,10000,15000,25737]
# Blank list of calculated time fo each sample
test_time = []

for sample in test:
    X_sample = X[:sample]
    Y_sample = Y[:sample]
    Z_sample = Z[:sample]

    t_start_sample = process_time()

    sample_tree = KDTree(X_sample,Y_sample,Z_sample)

    knn = 30
    for i in range(sample):
        query = [X_sample[i], Y_sample[i], Z_sample[i]]
    # Density
        sample_tree.get_nn(query)
    # Curvature
        sample_tree.get_knn(query, knn)

    t_end_sample = process_time()

    time = t_end_sample - t_start_sample
    test_time.append(time)

# Generating the plot

# fig = figure()

plot(test, test_time, '-', label='KD-Tree')

xlabel("Number of points")
ylabel("Time [s]")
title("Plot of computing time on the number of points")
grid(True)
legend()

show()

# Results:
# Distance = 0.05539943701880381
# Density = 5881.443724827224
# Time of calculation (density + curvature) = on my machine 20-30 seconds
# Time of calculating all samples = on my machine max 2 minutes