# Method 1: Naive Search

# Autoři: Fuchsig, Kohout

# Prohlášení:
# Kód byl z většiny vypracován samostatně. Nejistoty či jiné problémy
# byly konzultovány s umělou inteligencí (Google Gemini). To se vztahuje
# na opravy syntaxe, ladění chyb či hrubý návrh postupu práce.
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


def getNN(xq, yq, zq, X, Y, Z):
    #Find nearest point and its distance
    dmin = inf
    xn, yn, zn = X[0], Y[0], Z[0]
    
    #Process all points
    for i in range(len(X)):
        #Compute distance
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]
        d = (dx*dx + dy*dy + dz * dz)**0.5
        
        #Actualize minimum: distance + coordinates
        if d < dmin and d > 0:
            dmin = d
            xn, yn, zn = X[i], Y[i], Z[i]
    return xn, yn, zn, dmin


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
print("Computing density with Naive Method")

# Compute Density - Method 1: Naive Search

#  Starting timer
t_start = process_time() 

sum_dist = 0

# Go through all points
for i in range(n):
    # Using function getNN to find nearest neighbour for point i
    # It returns coordinates (xn, yn, zn) and distance (d)
    xn, yn, zn,d = getNN(X[i], Y[i], Z[i], X, Y, Z)
    
    # Add distance to sum
    sum_dist += d
    
# Calculate average distance d_aver
d_aver = sum_dist / n

# Calculate density: rho = 1 / d**3
# Check for zero to avoid dividing by zero 
if d_aver > 0:
    rho= 1 / (d_aver**3)
else:
    rho = 0

#Stop timer
t_end = process_time()

# CHARACTERISTICS - DENSITY

print("Average distance:", d_aver)
print("Density:", rho)
print("Time elapsed:", t_end - t_start, "s")

# CREATE PLOT OF CALCULATION TIME ON NUMBER OF POINTS
# AI used here for debugging and correcting syntax

print("Generating the plot of calculation time on number of points.")

points_list = []
# List of test number of points (50 to 10000) because it takes long
test = [50,100,250,500,1000,2500,5000,7500,10000]
# Blank list of calculated time fo each sample
test_time = []


for sample in test:
    X_sample = X[:sample]
    Y_sample = Y[:sample]
    Z_sample = Z[:sample]

    t_start_sample = process_time()

    sum_dist_test = 0
    
    # Go through all points
    for i in range(sample):
        xn, yn, zn,d = getNN(X_sample[i], Y_sample[i], Z_sample[i], X_sample, Y_sample, Z_sample)
    
        sum_dist_test += d

    d_aver = sum_dist_test / sample

    if d_aver > 0:
        rho= 1 / (d_aver**3)
    else:
        rho = 0

    t_end_sample = process_time()

    time = t_end_sample - t_start_sample
    test_time.append(time)


# Generating the plot

fig = figure()

plot(test, test_time, '-', label='Naive Method')

xlabel("Number of points")
ylabel("Time [s]")
title("Plot of computing time on the number of points")
grid(True)
legend()

show()


# Results:
# Distance = 0.05539943701880381
# Density = 5881.443724827224
# Time of calculation density = on my machine around 3 minutes
# Time of calculating all samples = on my machine max 3 minutes