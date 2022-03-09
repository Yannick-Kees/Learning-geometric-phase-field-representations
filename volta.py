from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 10000
START_LEARNING_RATE = 0.01
PATIENCE = 1500
NUM_NODES = 256
FOURIER_FEATUERS = False
SIGMA = 1.3
BATCHSIZE = 15000 #16k zu viel

# Phase-Loss
LOSS = "AT"
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_BALL_SAMPLES = 60
EPSILON = .0001
if LOSS == "MM":
    CONSTANT = 140.0 if not FOURIER_FEATUERS else 140.0 # 14, Modica Mortola
else:
    CONSTANT = 10. if FOURIER_FEATUERS else 10.0 # 14, Constante höher bei FF
MU = 0.3


####################
# Main #############
####################

network = ParkEtAl(3, [NUM_NODES]*5, [], FourierFeatures=FOURIER_FEATUERS, num_features = 8, sigma = SIGMA )
network.to(device) 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

file = open("3dObjects/bunny_0.ply")
pc = read_ply_file(file)
cloud = torch.tensor(flat_circle(4000) )

#cloud += torch.tensor([0.15,-.15,.1]).repeat(cloud.shape[0],1)
#cloud = torch.tensor(normalize(cloud) )


pc = Variable( cloud , requires_grad=True).to(device)
use_batch = (len(pc) > BATCHSIZE )

for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2
    if use_batch:
        
        indices = np.random.choice(len(pc), BATCHSIZE, False)
        pointcloud = pc[indices]
    else:
        pointcloud = pc
    
    if LOSS == "AT":
            loss = AT_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT )
    else:
        loss = Phase_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, MU)
    # report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
    if (i%10==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().cpu().numpy() )
    # backpropagation
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    

torch.save(network.state_dict(), "scheibeAT.pth")
print("Finished")
