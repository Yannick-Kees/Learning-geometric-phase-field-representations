from visualizing import *

class small_MLP(nn.Module):
    # Neuronal Network
    def __init__(self, NUM_NODES):
        super(small_MLP, self).__init__()
        self.lin1 = nn.Linear(2,NUM_NODES)
        self.lin2 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin3 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin4 = nn.Linear(NUM_NODES,1)

    def forward(self, x):
        # Feed forward function
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        # In the end tanh, such that values are between -1 and 1
        x = torch.tanh(x)
      
        return x
    
class small_MLP3D(nn.Module):
    # Neuronal Network
    def __init__(self, NUM_NODES):
        super(small_MLP3D, self).__init__()
        self.lin1 = nn.Linear(3,NUM_NODES)
        self.lin2 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin3 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin4 = nn.Linear(NUM_NODES,1)

    def forward(self, x):
        # Feed forward function
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        # In the end tanh, such that values are between -1 and 1
        x = torch.tanh(x)
      
        return x


  
class Siren_Network(nn.Module):
    # Neuronal Network (SIREN-Network)
    # Sin- function as activation function
    
    def __init__(self, NUM_NODES):
        super(Siren_Network, self).__init__()
        self.lin1 = nn.Linear(3,NUM_NODES)
        self.lin2 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin3 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin4 = nn.Linear(NUM_NODES,1)
        torch.nn.init.uniform_(self.lin1.weight, -np.sqrt(1.5) , np.sqrt(1.5))
        torch.nn.init.uniform_(self.lin2.weight, -np.sqrt(1.5) , np.sqrt(1.5))
        torch.nn.init.uniform_(self.lin3.weight, -np.sqrt(1.5) , np.sqrt(1.5))
        torch.nn.init.uniform_(self.lin4.weight, -np.sqrt(1.5) , np.sqrt(1.5))


    def forward(self, x):
        # Feed forward function
        x = torch.sin( self.lin1(x))
        x = torch.sin(self.lin2(x))
        x = torch.sin(self.lin3(x))
        x = self.lin4(x)
        # In the end tanh, such that values are between -1 and 1
        x = torch.tanh(x)
      
        return x




  
class big_MLP(nn.Module):
    # do not use on regular computer/CPU
    def __init__(self, NUM_NODES):
        super(small_MLP, self).__init__()
        self.lin1 = nn.Linear(2,NUM_NODES)
        self.lin2 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin3 = nn.Linear(NUM_NODES,NUM_NODES)
        self.lin4 = nn.Linear(NUM_NODES,1)

    def forward(self, x):
        # Feed forward function
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        # In the end tanh, such that values are between -1 and 1
        x = torch.tanh(x)
        
        return x
      

class ParkEtAl(nn.Module):
    
    #   !! COPYRIGHT for this network class : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!
    #   Added Fourier features

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=.3,
        beta=100,
        FourierFeatures = False,
        num_features = 0,
        sigma = .5
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:               Neuronal Network
        #   d_in:               dimension of points in point cloud
        #   dims:               array, where dims[i] are the number of neurons in layer i
        #   skip_in:            array containg the layer indices for skipping layers
        #   geometric_init:     Geometric initialisation
        #   radius_init:        Radius for Geometric initialisation
        #   beta:               Value for softmax activation function
        #   FourierFeatures:    Use Fourier Features
        #   num_features:       Number of Fourier Features
        #   sigma:              Sigma value for Frequencies in FF
        
        super().__init__()
        
        self.FourierFeatures = FourierFeatures
        self.d_in = d_in
        
        if FourierFeatures:
            # Use Fourier Features

            self.d_in = d_in * num_features * 2         # Dimension of Fourier Features
            self.original_dim = d_in                    # Original Dimension
            self.FFL = rff.layers.GaussianEncoding(sigma=sigma, input_size=self.original_dim, encoded_size=self.d_in//2) # Fourier Feature layer


        dims = [self.d_in] + dims + [1]     # Number of neurons in each layer
        
        self.num_layers = len(dims)         # Number of layers
        self.skip_in = skip_in              # Skipping layers

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
                
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim) # Affine linear transformation

            
            if geometric_init:
                # if true preform preform geometric initialization
                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin) # Save layer

        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # Softplus activation function
            #self.activation = nn.Sigmoid()

        # vanilla ReLu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):
        # forward pass of the NN

        x = input
        if self.FourierFeatures:
            # Fourier Layer
            x = self.FFL(x)

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # Skipping layer
                
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x) # Apply layer

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
    

class FeatureSpaceNetwork(nn.Module):
    
    #   !! COPYRIGHT for this network class : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!
    #   Added Fourier features and a feature vector gets concatenated after passing FF-layer

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=.3,
        beta=100,
        FourierFeatures = False,
        num_features = 0,
        sigma = .5,
        feature_space = 64
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:               Neuronal Network
        #   d_in:               dimension of points in point cloud
        #   dims:               array, where dims[i] are the number of neurons in layer i
        #   skip_in:            array containg the layer indices for skipping layers
        #   geometric_init:     Geometric initialisation
        #   radius_init:        Radius for Geometric initialisation
        #   beta:               Value for softmax activation function
        #   FourierFeatures:    Use Fourier Features
        #   num_features:       Number of Fourier Features
        #   sigma:              Sigma value for Frequencies in FF
        
        super().__init__()
        
        self.FourierFeatures = FourierFeatures
        self.d_in = d_in
        if FourierFeatures:
            # Use Fourier Features

            self.d_in = d_in * num_features * 2         # Dimension of Fourier Features
            self.original_dim = d_in                    # Original Dimension
            self.FFL = rff.layers.GaussianEncoding(sigma=sigma, input_size=self.original_dim, encoded_size=self.d_in//2) # Fourier Feature layer


        dims = [self.d_in] + dims + [1]     # Number of neurons in each layer
        dims[0] += feature_space
        
        self.num_layers = len(dims)         # Number of layers
        self.skip_in = skip_in              # Skipping layers

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim) # Affine linear transformation

            
            if geometric_init:
                # if true preform preform geometric initialization
                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin) # Save layer

        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # Softplus activation function
            #self.activation = nn.Sigmoid()

        # vanilla ReLu
        else:
            self.activation = nn.ReLU()

    def forward(self, input, ft):
        # forward pass of the NN
        #
        # input:    Point in R^3
        # ft:       Feature Vector   

        x = input
        if self.FourierFeatures:
            # Fourier Layer
            x = self.FFL(x)         # Pass Fourier Features
            x = torch.cat((x,ft),1) # Concatenate Feature vector

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # Skipping layer
                
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x) # Apply layer

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
    
    
class FeatureSpaceNetwork2(nn.Module):
    
    #   !! COPYRIGHT for this network class : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!
    #   Added Fourier features and a feature vector gets concatenated after passing FF-layer

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=.3,
        beta=100,
        FourierFeatures = False,
        num_features = 0,
        sigma = .5,
        feature_space = 64
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:               Neuronal Network
        #   d_in:               dimension of points in point cloud
        #   dims:               array, where dims[i] are the number of neurons in layer i
        #   skip_in:            array containg the layer indices for skipping layers
        #   geometric_init:     Geometric initialisation
        #   radius_init:        Radius for Geometric initialisation
        #   beta:               Value for softmax activation function
        #   FourierFeatures:    Use Fourier Features
        #   num_features:       Number of Fourier Features
        #   sigma:              Sigma value for Frequencies in FF
        
        super().__init__()
        
        self.FourierFeatures = FourierFeatures
        self.d_in = d_in
        if FourierFeatures:
            # Use Fourier Features

            self.d_in = d_in * num_features * 2         # Dimension of Fourier Features
            self.original_dim = d_in                    # Original Dimension
            self.FFL = rff.layers.GaussianEncoding(sigma=sigma, input_size=self.original_dim, encoded_size=self.d_in//2) # Fourier Feature layer


        dims = [self.d_in] + dims + [1]     # Number of neurons in each layer
        dims[0] += feature_space
        
        self.num_layers = len(dims)         # Number of layers
        self.skip_in = skip_in              # Skipping layers

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim) # Affine linear transformation

            
            if geometric_init:
                # if true preform preform geometric initialization
                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin) # Save layer

        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # Softplus activation function
            #self.activation = nn.Sigmoid()

        # vanilla ReLu
        else:
            self.activation = nn.ReLU()

    def forward(self, input, ft):
        # forward pass of the NN
        #
        # input:    Point in R^3
        # ft:       Feature Vector   

        x = input
        if self.FourierFeatures:
            # Fourier Layer
            x = self.FFL(x)         # Pass Fourier Features
            x = torch.cat((x,ft),1) # Concatenate Feature vector

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # Skipping layer
                
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x) # Apply layer

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
    
 
class PCAutoEncoder(nn.Module):
    #  !! COPYRIGHT: https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/model/model.py !!
    """ Point-Net Autoencoder for Point Cloud 

    """
    def __init__(self, point_dim, num_points):
        super(PCAutoEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1)

        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_points*3)

        #batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points , global_feat
    
    
    
    
    


class PCAutoEncoder2(nn.Module):
    #  !! COPYRIGHT: https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/model/model.py !!
    """ Point-Net Autoencoder for Point Cloud 
    Input: 
    Output: 
    """
    def __init__(self, point_dim, num_points):
        super(PCAutoEncoder2, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=512, kernel_size=1)


        self.fc1 = nn.Linear(in_features=512, out_features=512)

        self.fc3 = nn.Linear(in_features=512, out_features=num_points*3)

    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder
        x = F.relu(self.conv1(x))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.fc1(x))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points , global_feat
    
    
class PCAutoEncoder64(nn.Module):
    #  !! COPYRIGHT: https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/model/model.py !!
    #
    # 64 dimensional Feature space

    def __init__(self, point_dim, num_points):
        super(PCAutoEncoder64, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)


        self.fc1 = nn.Linear(in_features=64, out_features=512)

        self.fc3 = nn.Linear(in_features=512, out_features=num_points*3)

    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder
        x = F.relu(self.conv1(x))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 64)
        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.fc1(x))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points , global_feat
    
class PointNetAutoEncoder(nn.Module):
    #  !! COPYRIGHT: https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/model/model.py !!
    #
    # 64 dimensional Feature space

    def __init__(self, point_dim, num_points, ft_dimension):
        super(PointNetAutoEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=ft_dimension, kernel_size=1)


        self.fc1 = nn.Linear(in_features=ft_dimension, out_features=512)

        self.fc3 = nn.Linear(in_features=512, out_features=num_points*3)
        self.ft_dimension = ft_dimension

    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder
        x = F.relu(self.conv1(x))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.ft_dimension)
        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.fc1(x))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points , global_feat
    
    
