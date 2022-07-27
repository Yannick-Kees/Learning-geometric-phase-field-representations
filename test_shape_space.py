from shapemaker import *

autoencoder = PCAutoEncoder2(3, 400)
autoencoder.load_state_dict(torch.load(r"autoencoder.pth", map_location=device))
autoencoder.to(device) 
autoencoder.eval()


dataset = np.load(open("dataset.npy", "rb"))

network =  ParkEtAl(512+3, [520]*7 , [4], FourierFeatures=3, num_features = 8, sigma = 3 )
network.load_state_dict(torch.load(r"shape_space.pth", map_location=device))
network.to(device) 
network.eval()


index = 36
point = [.5,  -.5,  -.5 ]

point = Tensor(point)
points = [dataset[index]]

# points = points.cuda()
points = np.array(points)
points = Variable( Tensor(points) , requires_grad=True).to(device)


inputs = torch.transpose(points, 1, 2)
reconstructed_points, global_feat = autoencoder(inputs)

x = torch.cat((point, global_feat[0]))
# print(network(x))

shape_space_toParaview(network, 64, 36, global_feat)