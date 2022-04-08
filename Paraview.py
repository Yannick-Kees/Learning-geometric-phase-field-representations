from networks import *

# Look up in 3dvisualization.py for the configurations of the different models
f  = ParkEtAl(3, [512]*7, [4], FourierFeatures=True, num_features = 8, sigma = 1.7 )
f.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\bunny5.pth", map_location=device))

toParaview(f, 32)



    
    
