from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 20000
START_LEARNING_RATE = 0.01
PATIENCE = 1500
NUM_NODES = 128

# Phase-Loss
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_BALL_SAMPLES = 50
EPSILON = .01
CONSTANT = 14.0
MU = 0.0


####################
# Main #############
####################

network = ParkEtAl(2, [NUM_NODES]*4, [2] )
network.to(device)
 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

pointcloud = Variable( g_quadrath  , requires_grad=True).to(device)

for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2
    
    loss = Phase_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, MU)
    #loss = at_loss(network, network2, EPSILON**3, EPSILON, pointcloud, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT )
    if (i%50==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )
    # backpropagation
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
draw_point_cloud(pointcloud)
torch.save(network.state_dict(), r"C:\Users\Yannick\Desktop\MA\Programming part\models\bla.pth")
ff =lambda x: network(x) + gradient(x, network(x))
color_plot(network)
draw_phase_field(network, 1.0, 1.0)
