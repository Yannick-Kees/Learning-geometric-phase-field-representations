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

#   Load dataset
dataset = np.load(open(r"dataset/dataset_8D.npy", "rb"),allow_pickle=True)

#   Setup Network
network =  ParkEtAl(3+8, [520]*7 , [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA )
#network = FeatureSpaceNetwork(3, [520]*7 , [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA, feature_space=8 )
network.to(device) 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)


for i in range(NUM_TRAINING_SESSIONS+1):
    
    network.zero_grad()
    loss = 0
    shape_batch = np.random.choice(50, SHAPES_EACH_STEP, replace=False)
    
    for index in shape_batch:

        shape = dataset[index][0]#[:,:num_points]
        latent = Tensor(dataset[index][1]).to(device)
        pointcloud = Variable( Tensor(shape) , requires_grad=False).to(device)

        latent = torch.ravel(latent)
        loss +=  AT_loss_shapespace(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES,  CONSTANT, latent )
        
    if (i%10==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
        
    # backpropagation
        
    loss.backward(retain_graph= True )
    optimizer.step()
    scheduler.step(loss)
    

torch.save(network.state_dict(), "shape_space_8D_NoEncoder_AFF.pth")
print("Finished")

