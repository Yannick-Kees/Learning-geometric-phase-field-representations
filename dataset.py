from shapemaker import *


####################
# Metaballs ########
####################

def create_8D():
    # Create dataset of metaballs from 2 balls
    # 2 . 3 coordinates + 2 radii = 8 dimensions
    
    points = []

    f = open("dataset_8D.npy", "wb")
    
    for i in range(50):
        points.append(shape_maker8D(2000))
        print(i)
        
        
    np.save(f, points)
    return
    
def eval_8D(i):
    # Plot metaballs of 2 balls from the dataset
    
    f = np.load(open("dataset_8D.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))
    print(f[i][1])


def create():
    # Create dataset of metaballs
    
    points = []

    f = open("dataset1k.npy", "wb")
    
    for i in range(50):
        points.append(shape_maker1(3,1000))
        print(i)
        
    np.save(f, points)
    return
    
def eval(i):
    # visualize items from the metaball dataset
    
    f = np.load(open("dataset1k.npy", "rb"))
    draw_point_cloud(Tensor(f[i]))
    
    
    
####################
# Ellipsoid ########
####################    
    
    
def create_ell():
    # Create dataset of ellipsoids
    points = []

    f = open("dataset_ellipsoid.npy", "wb")
    
    for i in range(50):
        points.append(make_ellipse())
        print(i)
        
        
    np.save(f, points)
    return
    
def eval_ell(i):
    # Plot ellipsoids from the dataset
    
    f = np.load(open("dataset_ellipsoid.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))
    
    
 
eval_8D(1)