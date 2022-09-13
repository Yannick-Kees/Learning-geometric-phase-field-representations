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

#   Load autoencoder
autoencoder = PCAutoEncoder64(3, 1000)
autoencoder.load_state_dict(torch.load(r"models/autoencoder64.pth", map_location=device))
autoencoder.to(device) 
autoencoder.eval()

#   Load dataset
dataset = np.load(open(r"dataset/dataset1k.npy", "rb"))

#   Setup Shape Space Learning Network
network =  FeatureSpaceNetwork(3, [520]*7 , [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA )
network.to(device) 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)


for i in range(NUM_TRAINING_SESSIONS+1):
    
    network.zero_grad()
    loss = 0
    shape_batch = np.random.choice(50, SHAPES_EACH_STEP, replace=False)
    
    for index in shape_batch:

        shape = dataset[index]#[:,:num_points]
        pointcloud = Variable( Tensor(shape) , requires_grad=False).to(device)

        cloudT = Tensor( np.array([ np.array(shape).T]))
        pointcloudT = Variable( cloudT , requires_grad=True).to(device)

        rec, latent = autoencoder(pointcloudT)
        latent = torch.ravel(latent)
        loss +=  AT_loss_shapespace2(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES,  CONSTANT, latent )
        
    if (i%10==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
        
        # backpropagation
        
    loss.backward(retain_graph= True )
    optimizer.step()
    scheduler.step(loss)
    

torch.save(network.state_dict(), "shape_space_64e.pth")
print("Finished")
