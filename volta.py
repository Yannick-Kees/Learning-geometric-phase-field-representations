from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 10000
START_LEARNING_RATE = 0.01
PATIENCE = 1500
NUM_NODES = 128

# Phase-Loss
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_BALL_SAMPLES = 60
EPSILON = .001
CONSTANT = 50.0
MU = 0.1


####################
# Main #############
####################

network = ParkEtAl(3, [NUM_NODES]*4, [2] )
network.to(device) 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

file = open("3dObjects/bigcube.off")    
pc = read_off(file)
cloud = torch.tensor(normalize(pc) )


pointcloud = Variable( cloud , requires_grad=True).to(device)

for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2
    
    loss = Phase_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, MU)
    #report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )
    if (i%10==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )
    # backpropagation
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    

torch.save(network.state_dict(), r"C:\Users\Yannick\Desktop\MA\Programming part\models\bigcubePARK.pth")
print("Finished")