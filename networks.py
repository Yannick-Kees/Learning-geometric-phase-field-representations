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