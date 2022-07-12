from shapemaker import *
"""
points = []

f = open("dataset.npy", "wb")
for i in range(50):
    points.append(shape_maker1(3,400))
    print(i)
    
    
np.save(f, points)
  
"""

f = np.load(open("dataset.npy", "rb"))
draw_point_cloud(Tensor(f[22]))