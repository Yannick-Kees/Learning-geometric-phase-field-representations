

from networks import *
""" 
file = open("3dObjects/bigcube.off")    
pc = read_off(file)
print(pc)
cloud = torch.tensor(normalize(pc) )
draw_point_cloud(cloud)
""" 
def f(x):
    return x[1]**2+x[0]**2


"""
file = open("3dObjects/cow.off")    
pc = read_off(file)
 
cloud = torch.tensor(normalize(pc) )

#draw_point_cloud(cloud)
cloud += torch.tensor([0.15,-.15,.1]).repeat(cloud.shape[0],1)
cloud = torch.tensor(normalize(cloud) )
print(len(cloud))

pointcloud = Variable(torch.tensor(normalize(produce_pan(50)))  , requires_grad=True).to(device)
draw_point_cloud(pointcloud)
"""

def read_ply_file(file):
    
    data = file.readlines()
    num_vertices = int(data[4].split(" ")[2].replace("\n",""))

 

    vertices = [   [float(x)  for x in row.split(" ")[0:3] ] for row in data[11:11+num_vertices]]
    return vertices
    
Test = False
if Test:
        file = open("3dObjects/bunny_0.ply")
        pc = read_ply_file(file)
        cloud = torch.tensor(normalize(pc))
        #cloud = torch.tensor( flat_circle(2000) )

        cloud += torch.tensor([0.15,-.15,.1]).repeat(cloud.shape[0],1)
        cloud = torch.tensor(normalize(cloud) )


        pc = Variable( cloud , requires_grad=True).to(device)
        indices = np.random.choice(len(pc), 1000, False)
        pointcloud = pc[indices]
        #cloud = torch.tensor(flat_circle(8000) )

        print(len(pc))


print(17*512+513*512*4+513)