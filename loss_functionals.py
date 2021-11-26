from networks import *


def gradient(inputs, outputs):
    # Returns:
    #   Pointwise gradient estimation [ Df(x_i) ]
    
    # Parameters:
    #   inputs:     [ x_i ]
    #   outputs:    [ f(x_i) ] 
    
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True)[0][:,-3:]
    return points_grad

#############################
# PHASE - Loss ##############
#############################

# double well potential
W = lambda s: s**2 - 2.0*torch.abs(s) + torch.tensor([1.0])

def ModicaMortola(f, eps, n):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    
    start_points = Variable(torch.rand(n, 2), requires_grad =True)  # Create random points [ x_i ]
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return (W(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2


def Zero_recontruction_loss(f, pc, eps, n, c):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    
    loss = 0    # loss in the sum
    dim = 2     # Dimension of points in point cloud
    
    for point in pc:
        # loop over all point in pointcloud, i.e. x\in X
        
        variation  = torch.normal(mean = torch.full(size=( n* dim,1), fill_value=0.0) , std= torch.full(size=(n*dim,1), fill_value=.001) )  # Random points in B_\delta(0)
        start_point = point.repeat(n,1)+   torch.reshape( variation, (n, dim) )                                                             # Random points [ x_i ] in B_\delta(x)
        loss +=  torch.abs(f(start_point).mean())                                                                                           # Estimate \sum_{x\in X} |\dashint_{B_delta} u(x) dx|

    return  c*eps**(1.0/3.0)/(len(pc)) *  loss      # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} u(x) dx|


def Eikonal_loss(f, pc, eps):
    # Returns:
    #   Eikonal loss around the points of point cloud
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    
    gradients = gradient( pc, f(pc) )                                               # calculates [ Du(x) ]
    norms = np.sqrt(eps) *  gradients.norm(2,dim=-1)                                # calculates [ |Dw(x)| ] = [ sqrt(eps) * |Du(x)| ]
    eikonal = torch.abs(torch.full(size=(len(pc) ,1), fill_value=1.0)-norms)**2     # calculates [ | 1-|Dw(x)| |^2 ] 
    
    return eikonal.mean()    # return \sum_{x\in X} |1-|Dw(u) | |^2                 # returns [ | 1-|Dw(x)| |^2 ] 

def Phase_loss(f, pointcloud, eps, n, m, c, mu ):
    return eps**(-.5)*(ModicaMortola(f, eps, n) +  Zero_recontruction_loss(f, pointcloud, eps, m, c))+mu * Eikonal_loss(f, pointcloud, eps )


#############################
# Ambrosio Tortorelli #######
#############################