

from networks import *
""" 
file = open("3dObjects/bigcube.off")    
pc = read_off(file)
print(pc)
cloud = torch.tensor(normalize(pc) )
draw_point_cloud(cloud)
""" 

def read_ply_file(file):
    
    data = file.readlines()
    num_vertices = int(data[4].split(" ")[2].replace("\n",""))

    vertices = [   [float(x)  for x in row.split(" ")[0:3] ] for row in data[11:11+num_vertices]]
    return vertices
    
Test = False
if Test:
        file = open("3dObjects/nefertiti.obj")
        pc =  read_obj_file(file)

        #pc = read_off(file)
        cloud = torch.tensor(cut_hole(normalize(pc)))
        #cloud = torch.tensor( flat_circle(2000) )

        #cloud += torch.tensor([0.15,-.15,.1]).repeat(cloud.shape[0],1)
        #cloud += torch.tensor([0.15,-.15,.1]).repeat(cloud.shape[0],1)
        #cloud = torch.tensor(normalize(cloud) )


        pc = Variable( cloud , requires_grad=True).to(device)
        indices = np.random.choice(len(pc), 2000, False)
        pointcloud = pc[indices]
        #cloud = torch.tensor(flat_circle(8000) )

        draw_point_cloud(pointcloud)
        

dataset = np.load(open("dataset_ellipsoid.npy", "rb"),allow_pickle=True)

network =  FeatureSpaceNetwork(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
network.load_state_dict(torch.load(r"shape_space_64.pth", map_location=device))
network.to(device) 

network.eval()
x = np.array([dataset[1][1].detach().cpu().numpy()])
latent = Tensor( x ) 