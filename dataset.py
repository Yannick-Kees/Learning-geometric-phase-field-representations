from shapemaker import *


####################
# Metaballs ########
####################

def create_8D():
    # Create dataset of metaballs from 2 balls
    # 2 . 3 coordinates + 2 radii = 8 dimensions
    
    points = []

    f = open(r"dataset/dataset_8D.npy", "wb")
    
    for i in range(50):
        points.append(shape_maker8D(2000, 4))
        print(i)
        
        
    np.save(f, points)
    return
    
def eval_8D(i):
    # Plot metaballs of 2 balls from the dataset
    
    f = np.load(open(r"dataset/dataset_16D.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))
    print(f[i][1])


def create():
    # Create dataset of metaballs
    
    points = []

    f = open(r"dataset/dataset1k.npy", "wb")
    
    for i in range(50):
        points.append(shape_maker1(3,1000))
        print(i)
        
    np.save(f, points)
    return
    
def eval(i):
    # visualize items from the metaball dataset
    
    f = np.load(open(r"dataset/dataset1k.npy", "rb"))
    draw_point_cloud(Tensor(f[i]))
    
    
    
####################
# Ellipsoid ########
####################    
    
    
def create_ell():
    # Create dataset of ellipsoids
    points = []

    f = open(r"dataset/dataset_ellipsoid.npy", "wb")
    
    for i in range(50):
        points.append(make_ellipse())
        print(i)
        
        
    np.save(f, points)
    return
    
def eval_ell(i):
    # Plot ellipsoids from the dataset
    
    f = np.load(open(r"dataset/dataset_ellipsoid.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))
    
    
####################
# Faces   ##########
####################  

# 383 faces in total

#import open3d as o3d

def load_face(index):
    # Load face 'index' from .ply file
    # Need to enable the import open3d part for this

    pcd = o3d.io.read_point_cloud("faces/face_" + str(index) + ".ply")
    x = np.asarray(pcd.points)
    x = Tensor(normalize(x))
    x = x - np.array([-0.22,-0.22,0.24])
    x = Tensor(normalize(x))
    x = x + np.array([0.0,.15,-0.15])
    x = np.array(normalize(x))
    
    if x.shape[0] != 23725:
        print("NOT THE RIGHT SIZE!!")
        print(index)
        
    return x

def make_face_dataset():
    # Convert all .ply files into one numpy array and save it
    points = []

    f = open(r"dataset/dataset_faces.npy", "wb")
    
    for i in range(100):
        print(i)
        x = (load_face(i),0)
        points.append(x)
        
    np.save(f, points)
    return

def show_face(i):
    # Plot face from the dataset

    f = np.load(open(r"dataset/dataset_faces.npy", "rb"),allow_pickle=True)
    draw_point_cloud(Tensor(f[i][0]))     


show_face(5)