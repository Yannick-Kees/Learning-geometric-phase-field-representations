

from networks import *
""" 
file = open("3dObjects/bigcube.off")    
pc = read_off(file)
print(pc)
cloud = torch.tensor(normalize(pc) )
draw_point_cloud(cloud)
""" 
def f(x):
    return x[1]**2+x[0]**2


def Zero_recontruction_loss_Lip(f, pc,  m,  d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    """" 
    loss = 0    # loss in the sum
    
    for point in pc:
        # loop over all points in pointcloud, i.e. x\in X
        
        variation  = torch.normal(mean = torch.full(size=( m* d,1), fill_value=0.0) , std= torch.full(size=(m*d,1), fill_value=.001) )  # Random points in B_\delta(0)
        start_point = point.repeat(m,1)+   torch.reshape( variation, (m, d) )                                                             # Random points [ x_i ] in B_\delta(x)
        start_point = start_point.to(device)
        loss +=  torch.abs(f(start_point).mean())                                                                                           # Estimate \sum_{x\in X} |\dashint_{B_delta} u(x) dx|

    """
    n = len(pc)
    matrix = pc.repeat(m,1)
    matrix = torch.reshape(matrix, (m,n,d))
    print(matrix)
    variation  = torch.normal(mean = torch.full(size=( n*m *d,1), fill_value=0.0) , std= torch.full(size=(m*n*d,1), fill_value=.001) )
    error = torch.reshape( variation, (m,n, d) )
    matrix += error
    print(matrix)
    matrix = matrix.reshape(m*n,d)
    print(matrix)
    #matrix = f(matrix)
    print(matrix)
    matrix = torch.reshape( matrix, (m,n, d) )
    matrix  = matrix.mean(0)
    print(matrix)
    matrix = torch.abs(matrix)
    loss = matrix.mean()
        
    


    return      # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} u(x) dx|



file = open("3dObjects/cow.off")    
pc = read_off(file)
cloud = torch.tensor(normalize(pc) )
draw_point_cloud(cloud)
cloud += torch.tensor([0.15,-.15,.1]).repeat(cloud.shape[0],1)
cloud = torch.tensor(normalize(cloud) )
