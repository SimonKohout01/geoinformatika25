# Method 2: Voxelization (Grid)

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

    
def drawVoxels(x_min, y_min, z_min, dx, dy, dz, V):
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal') 
    
    #Create meshgrid
    xedges = linspace(x_min, x_min + dx, n_r+1) # type: ignore
    yedges = linspace(y_min, y_min + dy, n_r+1) # type: ignore
    zedges = linspace(z_min, z_min + dz, n_r+1) # type: ignore
    
    VX, VY, VZ = meshgrid(xedges, yedges, zedges,  indexing="ij") 
    
    #Draw voxels
    ax.voxels(VX, VY, VZ, V, edgecolor='k')
    
    show()
        

#Load points
X, Y, Z = loadPoints('tree_18.txt')

#Draw points
drawPoints(X, Y, Z, 0, 0.2)


def init_spat_index(X, Y, Z, n_xyz):
    # Compute bounding box
    x_min = min(X)
    x_max = max(X)
    y_min = min(Y)
    y_max = max(Y)
    z_min = min(Z)
    z_max = max(Z)
    
    # Min_max box edges
    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min
    
    # Size of one cell
    bx = dx / n_xyz
    by = dy / n_xyz
    bz = dz / n_xyz

    return x_min, y_min, z_min, dx, dy, dz, bx, by, bz 

def get3D_index(x, y, z, x_min, y_min, z_min, dx, dy, dz, n_xyz):
    # Rounding constant (to avoid index out of bounds)
    c = 0.9999999999999
    
    # Reduced coordinates 
    xr = (x - x_min) / dx
    yr = (y - y_min) / dy
    zr = (z - z_min) / dz
    
    #Compute indices
    jx = int(xr * n_xyz * c)
    jy = int(yr * n_xyz * c)
    jz = int(zr * n_xyz * c)
    
    return jx, jy, jz

def get1D_index(jx, jy, jz, n_xyz):
    # Convert 3D index to unique 1D number
    return jx + jy * n_xyz + jz * (n_xyz**2)

def create3Dindex(X, Y, Z, x_min, y_min, z_min, dx, dy, dz, n_xyz):
    # Dictionary for spatial index
    H = {}
    
    for i in range(len(X)): 
        # Get address of the point
        jx, jy, jz = get3D_index(X[i], Y[i], Z[i], x_min, y_min, z_min, dx, dy, dz, n_xyz)
        
        # Get hash key
        idx = get1D_index(jx, jy, jz, n_xyz)
        
        # Create list if not exists
        if idx not in H:
            H[idx] = []
            
        # Save point ID 'i' into the cell
        H[idx].append(i)
        
    return H



# Initialization
n = len(X)
print("Number of points:", n)


# Compute Density - Method 2: Voxelization (Grid)

print("Computing density and curvature of points with Voxelization Method")


#Amount of bins/rows
n = len(X)
#n_r = int(n**(1/3))   # Computing, so it could be just about right
n_r = 15 
if n_r < 1: n_r = 1

print(f"Grid size: {n_r} x {n_r} x {n_r}")

#Initialize index variables
x_min, y_min, z_min, dx, dy, dz, bx, by, bz = init_spat_index(X, Y, Z, n_r)


t_start_grid = process_time()

# Build the index
H = create3Dindex(X, Y, Z, x_min, y_min, z_min, dx, dy, dz, n_r)


# Search for Nearest neighbors 
sum_dist_grid = 0
points_processed = 0

# AI used here for debugging and correcting syntax

for i in range(n):
    # Find which cell the point is in
    jx, jy, jz = get3D_index(X[i], Y[i], Z[i], x_min, y_min, z_min, dx, dy, dz, n_r)
    idx = get1D_index(jx, jy, jz, n_r)
    
    #Get candidate points from the same cell 
    candidates = H[idx]
    
    # Find nearest in candidates
    d_min = float('inf')  # Infinit limit
    
    # We use simple search within candidates
    for cand_id in candidates:
        if i == cand_id: continue 
        
        # Distances
        d_x = X[i] - X[cand_id]
        d_y = Y[i] - Y[cand_id]
        d_z = Z[i] - Z[cand_id]
        d = (d_x*d_x + d_y*d_y + d_z*d_z)**(1/2)  #Euclidean distance
        
        if d < d_min:
            d_min = d
            
    # If we found a neighbor 
    if d_min < float('inf'):    # If d_min is still infinity, the point was alone in the cell
        sum_dist_grid += d_min  # Add to total distance
        points_processed += 1

# Calculate average and use the fprmula
if points_processed > 0:
    d_aver_grid = sum_dist_grid / points_processed
    rho_grid = 1 / (d_aver_grid**3)
else:
    rho_grid = 0
    
    
# Approximating curvature in each point using voxels (points can also exist in neighbouring voxels)
# AI used here for suggesting and explaining a workflow, also correcting syntax

curvature = []
knn = 30

for i in range(n):
    jx_orig, jy_orig, jz_orig = get3D_index(X[i], Y[i], Z[i], x_min, y_min, z_min, dx, dy, dz, n_r)
    
    candidates = []
    
    # Need to search also neighbouring voxels
    for kx in range(jx_orig - 1, jx_orig + 2):
        for ky in range(jy_orig - 1, jy_orig + 2):
            for kz in range(jz_orig - 1, jz_orig + 2):
                
                # Check bounds
                if kx < 0 or kx > n_r or ky < 0 or ky > n_r or kz < 0 or kz > n_r: continue
                
                idx_neighbor = get1D_index(kx, ky, kz, n_r)
                if idx_neighbor in H:
                    candidates.extend(H[idx_neighbor])

# Search curvature in each point for 30 NN
    distances = []
    for c in candidates:
        if c == i: continue
        d = ((X[i]-X[c])**2 + (Y[i]-Y[c])**2 + (Z[i]-Z[c])**2)**0.5
        distances.append((d, c))
        
    distances.sort(key=lambda x: x[0])
    
    best_candidates = distances[:knn+1]
    
    knn_list = []
    
    for dist, idx in best_candidates:
        # KD Tree finds the closest point as the point itself, the distance is 0, it skips this point
        if idx != i:
            point = [X[idx], Y[idx], Z[idx]]         # Creates coordinates of the point
            knn_list.append(point)
    
    # Edge of point cloud -> low number of neighbours, then just add 0
    if len(knn_list) < 3:
        curvature.append(0)
        continue
    
    # Calculating the covariance
    # AI used here for helping with usage and calculating with PCA
         
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


t_end_grid = process_time()

# CHARACTERISTICS - DENSITY

print("Average distance:", d_aver_grid)
print("Density:", rho_grid)
print("Time elapsed:", t_end_grid - t_start_grid, "s")

# CHARACTERISTICS - APPROXIMATED CURVATURE FOR EACH POINT

print("Generating visualisation of point cloud where color of each point is determined by its approximated curvature.")

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
test_time_sample = []
n_r = 15    # Fixed number of boxes

# AI used here for debugging and correcting syntax

for sample in test:
    X_sample = X[:sample]
    Y_sample = Y[:sample]
    Z_sample = Z[:sample]

    t_start_sample = process_time()

    xm, ym, zm, ddx, ddy, ddz, bbx, bby, bbz = init_spat_index(X_sample, Y_sample, Z_sample, n_r)
    H = create3Dindex(X_sample, Y_sample, Z_sample, xm, ym, zm, ddx, ddy, ddz, n_r)
    
# Search for Nearest neighbors - same as in main code
    for i in range(sample):
        jx, jy, jz = get3D_index(X_sample[i], Y_sample[i], Z_sample[i], xm, ym, zm, ddx, ddy, ddz, n_r)
        idx = get1D_index(jx, jy, jz, n_r)
    
        candidates = H.get(idx,[])
    
        d_min = float('inf')
    
        for cand_id in candidates:
            if i == cand_id: continue 
        
            d_x = X_sample[i] - X_sample[cand_id]
            d_y = Y_sample[i] - Y_sample[cand_id]
            d_z = Z_sample[i] - Z_sample[cand_id]
            d = (d_x*d_x + d_y*d_y + d_z*d_z)**(1/2)
        
            if d < d_min:
                d_min = d
                   
    
    t_end_sample = process_time()

    time = t_end_sample - t_start_sample
    test_time_sample.append(time)

fig = figure()
plot(test, test_time_sample, '-', label='Voxel Grid (n_r=15)')
xlabel("Number of points")
ylabel("Time [s]")
title("Plot of computing time on the number of points")
grid(True)
legend()

show()


print("Generating the plot of calculation time on number of voxels.")

# List of test voxel sizes
test_grid = [5, 10, 15, 20, 25, 30]
# Blank list of calculated time fo each sample
test_time_grid = []
point_cloud = len(X)

# AI used here for debugging and correcting syntax

for n_r in test_grid:
    t_start_sample = process_time()

    xm, ym, zm, ddx, ddy, ddz, bbx, bby, bbz = init_spat_index(X, Y, Z, n_r)
    H = create3Dindex(X, Y, Z, xm, ym, zm, ddx, ddy, ddz, n_r)
    
# Search for Nearest neighbors - same as in main code
    for i in range(point_cloud):
        jx, jy, jz = get3D_index(X[i], Y[i], Z[i], xm, ym, zm, ddx, ddy, ddz, n_r)
        idx = get1D_index(jx, jy, jz, n_r)
    
        candidates = H.get(idx,[])
    
        d_min = float('inf')
    
        for cand_id in candidates:
            if i == cand_id: continue 
        
            d_x = X[i] - X[cand_id]
            d_y = Y[i] - Y[cand_id]
            d_z = Z[i] - Z[cand_id]
            d = (d_x*d_x + d_y*d_y + d_z*d_z)**(1/2)
        
            if d < d_min:
                d_min = d
                   
    
    t_end_sample = process_time()

    time = t_end_sample - t_start_sample
    test_time_grid.append(time)

fig = figure()
plot(test_grid, test_time_grid, '-', label='Voxel Grid (all points)')
xlabel("Size of voxels")
ylabel("Time [s]")
title("Plot of computing time on the number of voxels")
grid(True)
legend()

show()


# Results:
# Distance = 0.06184842111343027
# Density = 4226.823838628724
# Time of calculation (density + curvature) = on my machine cca 30 seconds
# Time of calculating all samples = on my machine around 1 minute