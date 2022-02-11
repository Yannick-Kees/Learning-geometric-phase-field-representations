from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 50000
START_LEARNING_RATE = 0.01
PATIENCE = 1500
NUM_NODES = 512
FOURIER_FEATUERS = True
SIGMA = 1.0

# Phase-Loss
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_BALL_SAMPLES = 60
EPSILON = .0001
CONSTANT = 70.0 if not FOURIER_FEATUERS else 100.0
MU = 0.0


####################
# Main #############
####################

network = ParkEtAl(3, [NUM_NODES]*7, [4], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA )
network.to(device) 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

file = open("3dObjects/bunny.off")    
pc = read_off(file)
cloud = torch.tensor(normalize(pc) )
cloud += torch.tensor([0.15,-.15,.1]).repeat(cloud.shape[0],1)
cloud = torch.tensor(normalize(cloud) )


pointcloud = Variable( cloud , requires_grad=True).to(device)

for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2
    
    loss = Phase_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, MU)
    #report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
    if (i%10==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
    # backpropagation
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    

torch.save(network.state_dict(), "bunny3.pth")
print("Finished")
