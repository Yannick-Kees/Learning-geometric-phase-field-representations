from shapemaker import *




def create():
    points = []

    f = open("dataset1k.npy", "wb")
    for i in range(50):
        points.append(shape_maker1(3,1000))
        print(i)
        
        
    np.save(f, points)
    return
    
def eval(i):
    f = np.load(open("dataset1k.npy", "rb"))
    draw_point_cloud(Tensor(f[i]))
    
    
    
def create_ell():
    points = []

    f = open("dataset_ellipsoid.npy", "wb")
    for i in range(50):
        points.append(make_ellipse())
        print(i)
        
        
    np.save(f, points)
    return
    
def eval_ell(i):
    f = np.load(open("dataset_ellipsoid.npy", "rb"),allow_pickle=True)
    print(f[i][1])
    draw_point_cloud(Tensor(f[i][0]))
    
    

    
eval_ell(5)