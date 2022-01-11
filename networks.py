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
        # Am Ende tanh, damit alles zwischen -1 und +1 ist
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
        # Am Ende tanh, damit alles zwischen -1 und +1 ist
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
        # Am Ende tanh, damit alles zwischen -1 und +1 ist
        x = torch.tanh(x)
        
        return x
    
    

class ParkEtAl(nn.Module):

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=1,
        beta=100
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:           Neuronal Network
        #   d_in:           dimension of points in point cloud
        #   dims:           array, where dims[i] are the number of neurons in layer i
        #   skip_in:        array containg the layer indices for skipping layers
        #   geometric_init: Geometric initialisation
        #   radius_init:    Radius for Geometric initialisation
        #   beta:           Value for softmax activation function
        
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            
            if geometric_init:
                # if true preform preform geometric initialization
                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x