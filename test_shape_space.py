from shapemaker import *


def test_ellipse(index):
 


    dataset = np.load(open("dataset_ellipsoid.npy", "rb"),allow_pickle=True)

    network =  ParkEtAl(3+3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"shape_space_ellipse.pth", map_location=device))
    network.to(device) 

    network.eval()
    x = np.array([dataset[index][1].detach().cpu().numpy()])
    latent = Tensor( x ) 

    

    shape_space_toParaview(network, 127, index, latent)
    return

def test_8D(index):
 


    dataset = np.load(open("dataset_8D.npy", "rb"),allow_pickle=True)

    # network =  ParkEtAl(3+3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network = FeatureSpaceNetwork(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3, feature_space=8 )
    network.load_state_dict(torch.load(r"shape_space_8D_NoEncoder.pth", map_location=device))
    network.to(device) 

    network.eval()
    x = np.array([dataset[index][1].detach().cpu().numpy()])
    latent = Tensor( x ) 

    

    shape_space_toParaview2(network, 127, index, latent)
    return


def test_shape(index):
    autoencoder = PCAutoEncoder64(3, 1000)
    autoencoder.load_state_dict(torch.load(r"autoencoder64.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()
    
    


    dataset = np.load(open("dataset1k.npy", "rb"))

    network =  FeatureSpaceNetwork(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"shape_space_64e.pth", map_location=device))
    network.to(device) 
    network.eval()



    point = [.5,  -.5,  -.5 ]

    point = Tensor(point)
    points = [dataset[index]]

    # points = points.cuda()
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)


    #x = torch.cat((point, global_feat[0]))
    # print(network(x))

    shape_space_toParaview2(network, 131, index, global_feat)
    return

test_8D(1)
print("HEy")
test_8D(3)
