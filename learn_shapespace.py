from torch import double
from shapemaker import *

####################
# Settings #########
####################

# Training Parameters
NUM_TRAINING_SESSIONS = 70000
START_LEARNING_RATE = 0.01
PATIENCE = 1500
NUM_NODES = 512
FOURIER_FEATUERS = True
SIGMA = 3.0
MONTE_CARLO_SAMPLES = 2000
SHAPES_EACH_STEP = 16
EPSILON = .0001
CONSTANT = 40. if FOURIER_FEATUERS else 10.0 



####################
# Main #############
####################

autoencoder = PCAutoEncoder2(3, 400)
autoencoder.load_state_dict(torch.load(r"autoencoder.pth", map_location=device))
autoencoder.to(device) 
autoencoder.eval()


dataset = np.load(open("dataset.npy", "rb"))

network =  ParkEtAl(512+3, [520]*7 , [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA )
network.to(device) 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)


for i in range(NUM_TRAINING_SESSIONS+1):
    
    network.zero_grad()
    loss = 0
    for _ in range(SHAPES_EACH_STEP):
        index = np.random.randint(50)
        shape = dataset[index]#[:,:num_points]
        pointcloud = Variable( Tensor(shape) , requires_grad=False).to(device)

        cloudT = Tensor( np.array([ np.array(shape).T]))
        pointcloudT = Variable( cloudT , requires_grad=True).to(device)

        rec, latent = autoencoder(pointcloudT)
        latent = torch.ravel(latent)
        loss +=  AT_loss_shapespace(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES,  CONSTANT, latent )
        
    if (i%10==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
        
        # backpropagation
        
    loss.backward(retain_graph= True )
    optimizer.step()
    scheduler.step(loss)
    

torch.save(network.state_dict(), "shape_space.pth")
print("Finished")

