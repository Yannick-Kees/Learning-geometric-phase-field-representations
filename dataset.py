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
    
eval(32)