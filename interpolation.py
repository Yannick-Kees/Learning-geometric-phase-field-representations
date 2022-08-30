from networks import *


def interpol_2d():
    # Simply interpolate function values of SDF's
    f = ParkEtAl(2, [128]*3, [], geometric_init=True, FourierFeatures=False, num_features = 6, sigma = 2 )
    f.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\circle.pth", map_location=device))


    g = ParkEtAl(2, [128]*3, [], geometric_init=True, FourierFeatures=False, num_features = 6, sigma = 2 )
    g.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\square.pth", map_location=device))


    def Interpolate(f, g, t, i):
        color_plot_interpolate(f,g,t, i, True)
        draw_phase_field_interpolate(f, g, t, .5, .5, i, True)

    for i in range(201):
        Interpolate(f, g, i/200, i)
        
    return
        
        
        
        
def interploate_3d(start_shape, end_shape):
    # Interpolat between shapes of shape space
    #
    #   start_shape:    Index of beginning shape in the dataset
    #   end_shape:      Index of final shape in the dataset
    
    #   Load autoencoder
    autoencoder = PCAutoEncoder64(3, 1000)
    autoencoder.load_state_dict(torch.load(r"autoencoder64.pth", map_location=device))
    autoencoder.to(device) 
    autoencoder.eval()
    
    #   Load dataset
    dataset = np.load(open("dataset1k.npy", "rb"))
    
    #   Load shape space network
    network =  FeatureSpaceNetwork(3, [520]*7 , [4], FourierFeatures=True, num_features = 8, sigma = 3 )
    network.load_state_dict(torch.load(r"shape_space_64.pth", map_location=device))
    network.to(device) 
    network.eval()


    points = [dataset[start_shape], dataset[end_shape]]

    # points = points.cuda()
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)


    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)


    for t in [0,.2,.4,.6,.8,1.0]:
        
        feature = Tensor(np.array([t* global_feat[1].detach().cpu().numpy() + (1-t) * global_feat[0].detach().cpu().numpy()]))

        shape_space_toParaview2(network, 128, t * 10, feature)
        
    return
        
        

interploate_3d(45, 44)