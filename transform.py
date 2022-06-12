from transformations import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 5000
START_LEARNING_RATE = 0.01
PATIENCE = 1000
NUM_NODES = 512
FOURIER_FEATUERS = False
SIGMA = 1.7
BATCHSIZE = 200

# LOSS
LOSS = "AT" # Either AT or MM
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_BALL_SAMPLES = 20
EPSILON = .01
CONSTANT = 2.0 if FOURIER_FEATUERS else 5.5 # 14, Constante höher bei FF





####################
# Main #############
####################

f =  ParkEtAl(2, [NUM_NODES]*2, [], geometric_init=False, FourierFeatures=False, num_features = 6, sigma = SIGMA )
network = ParkEtAl(2, [NUM_NODES]*2, [], geometric_init=False, FourierFeatures=False, num_features = 6, sigma = SIGMA )
f.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\CUBE.pth", map_location=device))

optimizer = optim.Adam(network.parameters(), START_LEARNING_RATE )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

for i in range(NUM_TRAINING_SESSIONS+1):
    # training the network
    # feed forward
    # Omega = [0,1]^2

    loss = sharpening(f, network, .5, .5, MONTE_CARLO_SAMPLES, 2, EPSILON)
        
    network.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)



color_plot(network, 2, False)
draw_phase_field(network, .5, .5, i, False)
#draw_height(network)
torch.save(network.state_dict(), r"C:\Users\Yannick\Desktop\MA\Programming part\models\circle.pth")


