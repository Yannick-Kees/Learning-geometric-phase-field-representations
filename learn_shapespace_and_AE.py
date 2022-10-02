from shapemaker import *

####################
# Settings #########
####################

# Training Parameters
NUM_TRAINING_SESSIONS = 100     
START_LEARNING_RATE = 0.01      
PATIENCE = 1500
NUM_NODES = 512         
FOURIER_FEATUERS = True
SIGMA = 3.0
MONTE_CARLO_SAMPLES = 2000
SHAPES_EACH_STEP = 1
EPSILON = .0001
CONSTANT = 40. if FOURIER_FEATUERS else 10.0 

# Network Design
FEATURE_DIMENSION = 12
SIZE_POINTCLOUD = 23725
TOTAL_SHAPES = 100



####################
# Main #############
####################

#   Load autoencoder
autoencoder = PointNetAutoEncoder(3,SIZE_POINTCLOUD,FEATURE_DIMENSION)
autoencoder.to(device) 


#   Load dataset
dataset = np.load(open(r"dataset/dataset_faces100.npy", "rb"),allow_pickle=True)

#   Setup Shape Space Learning Network
network =  FeatureSpaceNetwork2(3, [520]*2 , [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA, feature_space=FEATURE_DIMENSION, geometric_init=False )

network.to(device) 

all_params = chain(network.parameters(), autoencoder.parameters())
optimizer = torch.optim.Adam(all_params, START_LEARNING_RATE)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

# Check if it really trains both networks at the same time | Part 1
#print(autoencoder(Variable( Tensor( np.array([ np.array(dataset[1][0]).T])) , requires_grad=True).to(device)))


for i in range(NUM_TRAINING_SESSIONS+1):
    
    network.zero_grad()
    autoencoder.zero_grad()
    loss = 0
    shape_batch = np.random.choice(TOTAL_SHAPES, SHAPES_EACH_STEP, replace=False)
    
    for index in shape_batch:

        shape = dataset[index][0]#[:,:num_points]
        pointcloud = Variable( Tensor(shape) , requires_grad=False).to(device)

        cloudT = Tensor( np.array([ np.array(shape).T]))
        pointcloudT = Variable( cloudT , requires_grad=True).to(device)

        rec, latent = autoencoder(pointcloudT)
        latent = torch.ravel(latent)
        loss +=  AT_loss_shapespace2(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES,  CONSTANT, latent )
        
    if (i%10==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
        
        # backpropagation
        
    loss.backward( )
    optimizer.step()
    scheduler.step(loss)


# Check if it really trains both networks at the same time | Part 2   
#print(autoencoder(Variable( Tensor( np.array([ np.array(dataset[1][0]).T])) , requires_grad=True).to(device)))

#torch.save(network.state_dict(), r"models/face_space.pth")
#torch.save(autoencoder.state_dict(), r"models/face_ae.pth")
print("Finished")
