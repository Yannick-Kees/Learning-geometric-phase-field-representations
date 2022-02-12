from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 5000
START_LEARNING_RATE = 0.01
PATIENCE = 1000
NUM_NODES = 128
FOURIER_FEATUERS = True
SIGMA = 1.3

# LOSS
LOSS = "MM" # Either AT or MM
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_BALL_SAMPLES = 20
EPSILON = .001
if LOSS == "MM":
    CONSTANT = 14.0 if FOURIER_FEATUERS else 14 # 14, Modica Mortola
else:
    CONSTANT = .65 if FOURIER_FEATUERS else 5.5 # 14, Constante höher bei FF
MU = 1.0


####################
# Main #############
####################

#network = ParkEtAl(2, [NUM_NODES]*3, [2],   geometric_init=False, FourierFeatures=FOURIER_FEATUERS, num_features = 6, sigma = SIGMA )
network = ParkEtAl(2, [NUM_NODES]*3, [], geometric_init=True, FourierFeatures=False, num_features = 6, sigma = SIGMA )
#network = small_MLP(NUM_NODES)
network.to(device)
 
optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

pointcloud = Variable(torch.tensor(normalize(g_quadrath))  , requires_grad=True).to(device)

for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2
    
    if LOSS == "AT":
        loss = AT_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT )
    else:
        loss = Phase_loss(network, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, MONTE_CARLO_BALL_SAMPLES, CONSTANT, MU)
        
    if (i%50==0):
        report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )
        
    # backpropagation
    
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
draw_point_cloud(pointcloud)
torch.save(network.state_dict(), r"C:\Users\Yannick\Desktop\MA\Programming part\models\bla.pth")

color_plot(network)
draw_phase_field(network, .5, .5)
