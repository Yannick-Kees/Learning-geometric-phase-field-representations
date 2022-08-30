from networks import *

# Measure size of Neural Network
measure = False

if measure:

    # Create Network models, to measure the size
    for i in [1,2,4,8,16,32,64,128,256,512]:
        network = ParkEtAl(3, [i] , [], FourierFeatures=8, num_features = 8, sigma = 3 )  
        torch.save(network.state_dict(), str(i)+"cube.pth")
        
else:
    
    x = [ 1,2,4,8,16,32,64,128,256,512]

    y = [3,3,3,4,6,9,15,27,52,102]

    h = np.arange(1,512)
    plt.xlabel("Number of Neurons")
    plt.ylabel("Size of Neural Network in KB")
    plt.plot(x,y, 'p-')
    

    plt.show()