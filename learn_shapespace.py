from train_autoencoder import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 70000
START_LEARNING_RATE = 0.0001
PATIENCE = 1500
NUM_NODES = 512
FOURIER_FEATUERS = True
SIGMA = 3.0
BATCHSIZE = 10000 #16k zu viel

# Phase-Loss
MONTE_CARLO_SAMPLES = 2000
EPSILON = .0001
CONSTANT = 40. if FOURIER_FEATUERS else 10.0 # SIREN



####################
# Main #############
####################

#experiments = [ 0.01,0.1,.5,1,2,3,4,5,6,7,8,9,10]


autoencoder = PCAutoEncoder(3, 400)
autoencoder.load_state_dict(torch.load(r"autoencoder.pth", map_location=device))
autoencoder.train(mode=False)

network =  ParkEtAl(512+3, [512]*7 , [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA )
network.to(device) 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)




for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2
    
    network.zero_grad()
    
    cloud = torch.tensor( shape_maker1(3,400))
    pointcloud = Variable( cloud , requires_grad=True).to(device)
    
    rec, latent = autoencoder(cloud)


    loss =  AT_loss_shapespace(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES,  CONSTANT, latent )
    if (i%50==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )

        # report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
        
        # backpropagation
        
        loss.backward(retain_graph= True )
    optimizer.step()
    scheduler.step(loss)
    

torch.save(network.state_dict(), "shape_space.pth")
print("Finished")

